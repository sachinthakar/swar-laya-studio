// Swar-Laya Studio v5 — cloud: Firebase sign-in, save/load compositions, toast
// (split from the single-file studio.html; scripts load in order and share globals)

// ═══════════════════════════════════════════════════════════════════════════════
// PHASE 1 — CLOUD FEATURES (Firebase Auth + Firestore)
// ═══════════════════════════════════════════════════════════════════════════════
//
// Firebase is auto-initialized by /__/firebase/init.js (loaded in <head>)
// when the app is served from Firebase Hosting. This keeps the API key out
// of source code entirely. When opening locally (file://), cloud features
// are simply unavailable — use the hosted URL for full functionality.
//

// ── Initialise Firebase ───────────────────────────────────────────────────────
let auth, db;
try {
    if (typeof firebase !== 'undefined' && firebase.apps && firebase.apps.length > 0) {
        auth = firebase.auth();
        db   = firebase.firestore();
    } else {
        console.warn('Firebase not auto-initialized — cloud features disabled (local mode).');
    }
} catch(e) {
    console.warn('Firebase not configured. Cloud features disabled.', e.message);
}

// Load ragas from Firestore (must be after db is initialised above)
loadRagasFromFirestore();

// ── App state ─────────────────────────────────────────────────────────────────
let currentUser   = null;
let isPremium     = false;
let isAdmin       = false;
let currentCompId = null;   // Firestore document ID of the currently loaded composition

const FREE_COMP_LIMIT = 3;  // max compositions for free-tier accounts

// ── Auth state listener ───────────────────────────────────────────────────────
if (auth) {
    auth.onAuthStateChanged(async user => {
        currentUser = user;
        if (user) {
            await ensureUserProfile(user);
            const profile = await db.collection('users').doc(user.uid).get();
            isPremium = profile.exists ? (profile.data().isPremium || false) : false;
            isAdmin   = profile.exists ? (profile.data().isAdmin   || false) : false;
        } else {
            isPremium = false;
            isAdmin   = false;
        }
        updateAuthBar(user);
        // Rebuild raga dropdown now that isPremium/isAdmin are known
        populateRagaDropdown();
    });
}

// ── Auth UI ───────────────────────────────────────────────────────────────────
function updateAuthBar(user) {
    const bar = document.getElementById('auth-bar');
    const logo = '<span style="font-weight:700;letter-spacing:0.5px;">🎵 Swar-Laya Studio</span>';
    if (user) {
        const avatar  = user.photoURL
            ? `<img src="${user.photoURL}" class="user-avatar" onerror="this.style.display='none'">`
            : '';
        const name    = user.displayName || user.email || 'User';
        const badge   = isPremium
            ? '<span class="premium-badge">⭐ Premium</span>'
            : '<span class="free-badge">Free</span>';
        const adminBtn = isAdmin
            ? `<button class="btn-signout" style="background:#4a148c;color:#fff;" onclick="openRagaAdmin()">🎵 Raga Admin</button>`
            : '';
        bar.innerHTML = `
            ${logo}
            <div class="spacer"></div>
            ${avatar}
            <span class="user-name">${name}</span>
            ${badge}
            ${adminBtn}
            <button class="btn-signout" onclick="doSignOut()">Sign Out</button>
        `;
    } else {
        bar.innerHTML = `
            ${logo}
            <div class="spacer"></div>
            <button class="btn-google" onclick="signInGoogle()">Sign in with Google</button>
            <button class="btn-email-login" onclick="document.getElementById('email-modal').style.display='flex'">
                Sign in with Email
            </button>
        `;
    }
}

// ── Google sign-in ────────────────────────────────────────────────────────────
function signInGoogle() {
    if (!auth) { alert('Firebase not configured yet. Add your Firebase config keys first.'); return; }
    const provider = new firebase.auth.GoogleAuthProvider();
    auth.signInWithPopup(provider)
        .catch(e => alert('Sign-in failed: ' + e.message));
}

// ── Email sign-in / register (auto-creates account if new) ───────────────────
function emailSignIn() {
    if (!auth) { alert('Firebase not configured yet.'); return; }
    const email = document.getElementById('email-input').value.trim();
    const pass  = document.getElementById('pass-input').value;
    if (!email || !pass) { alert('Please enter email and password.'); return; }

    auth.signInWithEmailAndPassword(email, pass)
        .then(() => {
            document.getElementById('email-modal').style.display = 'none';
            document.getElementById('login-modal').style.display = 'none';
        })
        .catch(err => {
            if (err.code === 'auth/user-not-found' || err.code === 'auth/invalid-credential') {
                // Auto-register new user
                auth.createUserWithEmailAndPassword(email, pass)
                    .then(() => {
                        document.getElementById('email-modal').style.display = 'none';
                        document.getElementById('login-modal').style.display = 'none';
                        showToast('Account created and signed in ✓');
                    })
                    .catch(e2 => alert('Registration failed: ' + e2.message));
            } else {
                alert('Sign-in error: ' + err.message);
            }
        });
}

// ── Sign out ──────────────────────────────────────────────────────────────────
function doSignOut() {
    if (auth) auth.signOut();
    currentCompId = null;
}

// ── Ensure user profile doc exists in Firestore ───────────────────────────────
async function ensureUserProfile(user) {
    const ref = db.collection('users').doc(user.uid);
    const doc = await ref.get();
    if (!doc.exists) {
        await ref.set({
            email:       user.email || '',
            displayName: user.displayName || '',
            isPremium:   false,
            createdAt:   firebase.firestore.FieldValue.serverTimestamp()
        });
    }
}

// ── Cloud save ────────────────────────────────────────────────────────────────
async function saveToCloud() {
    if (!currentUser) { document.getElementById('login-modal').style.display = 'flex'; return; }

    try {
        // Free-tier limit check (only when creating a new composition)
        if (!isPremium && !currentCompId) {
            const existing = await db.collection('compositions')
                .doc(currentUser.uid).collection('songs').get();
            if (existing.size >= FREE_COMP_LIMIT) {
                alert(`Free accounts can save up to ${FREE_COMP_LIMIT} compositions.\n\nUpgrade to Premium for unlimited cloud storage.`);
                return;
            }
        }

        const data = buildCompositionData();
        const col  = db.collection('compositions').doc(currentUser.uid).collection('songs');

        if (currentCompId) {
            // Update existing composition
            await col.doc(currentCompId).update({
                ...data,
                updatedAt: firebase.firestore.FieldValue.serverTimestamp()
            });
        } else {
            // Save as new composition
            const newDoc = await col.add({
                ...data,
                createdAt: firebase.firestore.FieldValue.serverTimestamp(),
                updatedAt: firebase.firestore.FieldValue.serverTimestamp()
            });
            currentCompId = newDoc.id;
        }
        showToast('Saved to cloud ✓');
    } catch(e) {
        alert('Cloud save failed: ' + e.message);
    }
}

// ── Composition picker modal ──────────────────────────────────────────────────
async function openCompositions() {
    if (!currentUser) { document.getElementById('login-modal').style.display = 'flex'; return; }

    const modal = document.getElementById('comp-modal');
    const list  = document.getElementById('comp-list');
    list.innerHTML = '<div style="text-align:center;color:#888;padding:24px;">Loading…</div>';
    modal.style.display = 'flex';

    try {
        const snap = await db.collection('compositions')
            .doc(currentUser.uid).collection('songs')
            .orderBy('updatedAt', 'desc').get();

        if (snap.empty) {
            list.innerHTML = `
                <div style="text-align:center;color:#888;padding:24px;line-height:1.7;">
                    No compositions saved yet.<br>
                    Click <strong>Save to Cloud</strong> to save your first one.
                </div>`;
            return;
        }

        list.innerHTML = '';
        snap.forEach(doc => {
            const d    = doc.data();
            const date = d.updatedAt ? d.updatedAt.toDate().toLocaleDateString('en-IN') : '—';
            const div  = document.createElement('div');
            div.className = 'comp-item' + (doc.id === currentCompId ? ' active-comp' : '');
            div.innerHTML = `
                <div class="comp-info">
                    <strong>${d.title || 'Untitled'}</strong>
                    <span class="comp-meta">
                        ${(d.lines || []).length} lines &nbsp;·&nbsp;
                        ${d.bpm || '?'} BPM &nbsp;·&nbsp;
                        saved ${date}
                    </span>
                </div>
                <div class="comp-actions">
                    <button onclick="loadFromCloud('${doc.id}')" class="btn-load-comp">Load</button>
                    <button onclick="deleteComp('${doc.id}', this)" class="btn-del-comp">Delete</button>
                </div>`;
            list.appendChild(div);
        });
    } catch(e) {
        list.innerHTML = `<div style="color:#c62828;padding:16px;">Error loading: ${e.message}</div>`;
    }
}

// ── Load a composition from Firestore into the editor ────────────────────────
async function loadFromCloud(docId) {
    if (!currentUser) return;
    try {
        const doc = await db.collection('compositions')
            .doc(currentUser.uid).collection('songs').doc(docId).get();
        if (!doc.exists) { alert('Composition not found.'); return; }

        const d = doc.data();
        if (d.title) document.getElementById('song-title').value = d.title;
        if (d.scale) {
            const sv = parseFloat(d.scale) < 256 ? parseFloat(d.scale) * 2 : parseFloat(d.scale);
            let closestOpt = '';
            let minDiff = Infinity;
            document.querySelectorAll('#scale-select option').forEach(opt => {
                const diff = Math.abs(parseFloat(opt.value) - sv);
                if (diff < minDiff) {
                    minDiff = diff;
                    closestOpt = opt.value;
                }
            });
            document.getElementById('scale-select').value = closestOpt || sv;
        }
        if (d.bpm)   document.getElementById('bpm').value = d.bpm;
        document.getElementById('song-sheet').innerHTML = '';
        (d.lines || []).forEach(line => addLine(line));

        currentCompId = docId;
        document.getElementById('comp-modal').style.display = 'none';
        showToast('Composition loaded ✓');
    } catch(e) {
        alert('Load failed: ' + e.message);
    }
}

// ── Delete a composition from Firestore ───────────────────────────────────────
async function deleteComp(docId, btn) {
    if (!confirm('Delete this composition permanently? This cannot be undone.')) return;
    try {
        await db.collection('compositions')
            .doc(currentUser.uid).collection('songs').doc(docId).delete();
        if (currentCompId === docId) currentCompId = null;
        btn.closest('.comp-item').remove();
        showToast('Composition deleted');
    } catch(e) {
        alert('Delete failed: ' + e.message);
    }
}

// ── New composition (clears the editor) ───────────────────────────────────────
function newComposition() {
    if (!confirm('Start a new composition? Unsaved changes will be lost.')) return;
    document.getElementById('song-title').value = 'Untitled';
    document.getElementById('song-sheet').innerHTML = '';
    currentCompId = null;
    addLine();
    document.getElementById('comp-modal').style.display = 'none';
    showToast('New composition started');
}

// ── Toast notification helper ─────────────────────────────────────────────────
function showToast(msg) {
    const t = document.getElementById('toast');
    t.textContent = msg;
    t.style.opacity = '1';
    clearTimeout(t._timer);
    t._timer = setTimeout(() => { t.style.opacity = '0'; }, 2800);
}

