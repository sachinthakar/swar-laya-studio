// seed_ragas.mjs
// Seeds all ragas from ragas_all.json into Firestore
// Usage: node seed_ragas.mjs
//
// This uses the Firebase web SDK (v12 modular) in Node.js.
// Requires: npm install firebase (already installed)
//
// The script writes each raga as a document in the "ragas" Firestore collection,
// using the raga's key as the document ID.

import { initializeApp } from 'firebase/app';
import {
  getFirestore, doc, setDoc, writeBatch,
  collection, getDocs, deleteDoc
} from 'firebase/firestore';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));

// ── Firebase config (same as in the HTML) ───────────────────────────────────
const firebaseConfig = {
  apiKey:            "AIzaSyCsVf2UuPEgexC7WoQSK0ONxuiOUxo0a88",
  authDomain:        "swar-laya-studio.firebaseapp.com",
  projectId:         "swar-laya-studio",
  storageBucket:     "swar-laya-studio.firebasestorage.app",
  messagingSenderId: "508121720546",
  appId:             "1:508121720546:web:61c8535e222ac0916ae0bb"
};

// ── Load ragas ───────────────────────────────────────────────────────────────
const ragasPath = join(__dirname, 'ragas_all.json');
const ragas = JSON.parse(readFileSync(ragasPath, 'utf8'));
console.log(`Loaded ${ragas.length} ragas from ${ragasPath}`);

// ── Init Firebase ────────────────────────────────────────────────────────────
const app  = initializeApp(firebaseConfig);
const db   = getFirestore(app);
const COLL = 'ragas';

// ── Seed function ─────────────────────────────────────────────────────────────
async function seedRagas() {
  console.log(`\nSeeding ${ragas.length} ragas to Firestore collection "${COLL}"…\n`);

  // Firestore writeBatch supports max 500 ops per batch
  const BATCH_SIZE = 400;
  let batchCount = 0, docCount = 0, errorCount = 0;

  for (let i = 0; i < ragas.length; i += BATCH_SIZE) {
    const chunk = ragas.slice(i, i + BATCH_SIZE);
    const batch = writeBatch(db);

    for (const raga of chunk) {
      const key = raga.key;
      if (!key) {
        console.warn(`  SKIP: raga with no key at index ${i}`);
        continue;
      }
      // Remove 'key' field from the document (it's the doc ID)
      const { key: _k, ...ragaData } = raga;

      const ref = doc(db, COLL, key);
      batch.set(ref, ragaData);
      docCount++;
    }

    try {
      await batch.commit();
      batchCount++;
      console.log(`  Batch ${batchCount}: wrote ${chunk.length} docs (total: ${docCount})`);
    } catch (err) {
      errorCount++;
      console.error(`  Batch ${batchCount + 1} FAILED:`, err.message);
    }
  }

  console.log(`\n✅ Done. ${docCount} ragas seeded in ${batchCount} batch(es). Errors: ${errorCount}`);
}

seedRagas().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
