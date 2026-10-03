"""
Swar-Laya Studio — Raga list API (FastAPI + PostgreSQL), deployed on Render.

Endpoints
  GET    /health                 -> {"ok": true, "ragas": <count>}
  GET    /ragas                  -> {"<key>": {...raga...}, ...}  (sorted by order)
  GET    /ragas/{key}            -> {...raga...}
  PUT    /ragas/{key}            -> create/replace one raga          (admin)
  DELETE /ragas/{key}            -> delete one raga                  (admin)
  POST   /ragas/bulk             -> {"<key>": {...}, ...} upsert     (admin)

Admin = a signed-in Firebase user whose email is listed in ADMIN_EMAILS.
The browser sends its Firebase ID token as "Authorization: Bearer <token>";
the token is verified against Google's public keys for FIREBASE_PROJECT_ID.

First start with an empty table seeds it: from the existing Firestore
"ragas" collection (public read) when reachable, otherwise from the bundled
ragas_all.json.

Environment
  DATABASE_URL          Postgres connection string (Render injects it)
  FIREBASE_PROJECT_ID   default "swar-laya-studio"
  ADMIN_EMAILS          comma-separated admin emails
  ALLOWED_ORIGINS       comma-separated CORS origins, default "*"
  SEED_FROM_FIRESTORE   "1" (default) to try Firestore first when seeding
"""
import json
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import psycopg
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from google.auth.transport import requests as g_requests
from google.oauth2 import id_token as g_id_token

PROJECT_ID = os.environ.get("FIREBASE_PROJECT_ID", "swar-laya-studio")
ADMIN_EMAILS = {e.strip().lower() for e in os.environ.get("ADMIN_EMAILS", "").split(",") if e.strip()}
ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
SEED_FROM_FIRESTORE = os.environ.get("SEED_FROM_FIRESTORE", "1") == "1"
SEED_FILE = Path(__file__).with_name("ragas_all.json")

DB_URL = os.environ.get("DATABASE_URL", "")
if not DB_URL:
    raise RuntimeError("DATABASE_URL is not set")

pool = ConnectionPool(DB_URL, min_size=1, max_size=5, open=False)
_g_request = g_requests.Request()
_seed_lock = threading.Lock()

@asynccontextmanager
async def lifespan(_app):
    startup()
    yield
    pool.close()


app = FastAPI(title="Swar-Laya Raga API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_methods=["GET", "PUT", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

ALLOWED_FIELDS = {
    "name", "thaat", "order", "allowed", "vadi", "samvadi", "aroha", "avaroha",
    "pakkad", "jaati", "time", "rasa", "skippedAaroh", "skippedAvaroh",
}


# ── Firestore → plain JSON (used only for the first seed) ────────────────────
def _fs_value(v):
    t, x = next(iter(v.items()))
    if t == "integerValue":
        return int(x)
    if t == "arrayValue":
        return [_fs_value(i) for i in x.get("values", [])]
    if t == "mapValue":
        return {k: _fs_value(i) for k, i in x.get("fields", {}).items()}
    if t == "nullValue":
        return None
    return x  # stringValue, doubleValue, booleanValue, timestampValue


def _ragas_from_firestore():
    base = (f"https://firestore.googleapis.com/v1/projects/{PROJECT_ID}"
            f"/databases/(default)/documents/ragas?pageSize=300")
    out, token = {}, None
    with httpx.Client(timeout=30) as c:
        while True:
            r = c.get(base + (f"&pageToken={token}" if token else ""))
            r.raise_for_status()
            d = r.json()
            for doc in d.get("documents", []):
                key = doc["name"].rsplit("/", 1)[-1]
                out[key] = {k: _fs_value(v) for k, v in doc.get("fields", {}).items()}
            token = d.get("nextPageToken")
            if not token:
                return out


def _ragas_from_file():
    data = json.loads(SEED_FILE.read_text(encoding="utf-8"))
    return {r["key"]: {k: v for k, v in r.items() if k != "key"} for r in data}


def _clean(raga: dict) -> dict:
    return {k: v for k, v in raga.items() if k in ALLOWED_FIELDS}


def _upsert_many(cur, ragas: dict):
    for key, raga in ragas.items():
        raga = _clean(raga)
        cur.execute(
            """INSERT INTO ragas (key, data, ord, updated_at) VALUES (%s, %s, %s, now())
               ON CONFLICT (key) DO UPDATE
               SET data = EXCLUDED.data, ord = EXCLUDED.ord, updated_at = now()""",
            (key, Jsonb(raga), int(raga.get("order") or 0)),
        )


def startup():
    pool.open()
    pool.wait()
    with _seed_lock, pool.connection() as conn, conn.cursor() as cur:
        cur.execute("""CREATE TABLE IF NOT EXISTS ragas (
                           key        text PRIMARY KEY,
                           data       jsonb NOT NULL,
                           ord        integer NOT NULL DEFAULT 0,
                           updated_at timestamptz NOT NULL DEFAULT now())""")
        cur.execute("SELECT count(*) FROM ragas")
        if cur.fetchone()[0] > 0:
            return
        ragas, source = None, None
        if SEED_FROM_FIRESTORE:
            try:
                ragas, source = _ragas_from_firestore(), "firestore"
            except Exception as e:  # network/permission problems -> use the file
                print(f"[seed] Firestore read failed ({e}); using bundled file")
        if not ragas:
            ragas, source = _ragas_from_file(), "ragas_all.json"
        _upsert_many(cur, ragas)
        print(f"[seed] inserted {len(ragas)} ragas from {source}")


# ── Auth ─────────────────────────────────────────────────────────────────────
def require_admin(authorization: str | None):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Sign in required")
    token = authorization.split(" ", 1)[1].strip()
    try:
        claims = g_id_token.verify_firebase_token(token, _g_request, audience=PROJECT_ID)
    except Exception:
        raise HTTPException(401, "Invalid or expired sign-in token")
    email = (claims or {}).get("email", "").lower()
    if not email or not claims.get("email_verified", False) or email not in ADMIN_EMAILS:
        raise HTTPException(403, "Admin access required")
    return email


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM ragas")
        return {"ok": True, "ragas": cur.fetchone()[0]}


@app.get("/ragas")
def list_ragas():
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT key, data FROM ragas ORDER BY ord, key")
        return {k: d for k, d in cur.fetchall()}


@app.get("/ragas/{key}")
def get_raga(key: str):
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT data FROM ragas WHERE key = %s", (key,))
        row = cur.fetchone()
    if not row:
        raise HTTPException(404, "Raga not found")
    return row[0]


@app.put("/ragas/{key}")
async def put_raga(key: str, request: Request, authorization: str | None = Header(None)):
    require_admin(authorization)
    raga = await request.json()
    if not isinstance(raga, dict) or not raga.get("name"):
        raise HTTPException(400, "Body must be a raga object with a name")
    with pool.connection() as conn, conn.cursor() as cur:
        _upsert_many(cur, {key: raga})
    return {"ok": True, "key": key}


@app.delete("/ragas/{key}")
def delete_raga(key: str, authorization: str | None = Header(None)):
    require_admin(authorization)
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM ragas WHERE key = %s", (key,))
        if cur.rowcount == 0:
            raise HTTPException(404, "Raga not found")
    return {"ok": True, "key": key}


@app.post("/ragas/bulk")
async def bulk_ragas(request: Request, authorization: str | None = Header(None)):
    require_admin(authorization)
    ragas = await request.json()
    if not isinstance(ragas, dict):
        raise HTTPException(400, "Body must be an object keyed by raga key")
    with pool.connection() as conn, conn.cursor() as cur:
        _upsert_many(cur, ragas)
    return {"ok": True, "count": len(ragas)}
