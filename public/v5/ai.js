// Swar-Laya Studio v5 — admin + AI: raga admin panel, AI Compose
// (split from the single-file studio.html; scripts load in order and share globals)

// ═══════════════════════════════════════════════════════════════════════════════
// RAGA ADMIN PANEL
// ═══════════════════════════════════════════════════════════════════════════════

function openRagaAdmin() {
    if (!isAdmin) return;
    raAdminRefreshTable();
    document.getElementById('raga-admin-modal').classList.add('open');
}
function closeRagaAdmin() {
    document.getElementById('raga-admin-modal').classList.remove('open');
    raAdminCancelEdit();
    const s = document.getElementById('raga-admin-search');
    if (s) s.value = '';
}

// ── Table ─────────────────────────────────────────────────────────────────────
function raAdminRefreshTable() {
    const tbody = document.getElementById('raga-admin-tbody');
    tbody.innerHTML = '';
    const q = (document.getElementById('raga-admin-search')?.value || '').toLowerCase().trim();
    const allEntries = Object.entries(RagaData)
        .sort((a, b) => ((a[1].thaat||'') + String(a[1].order||0)).localeCompare((b[1].thaat||'') + String(b[1].order||0)));
    const filtered = q
        ? allEntries.filter(([key, r]) =>
            key.includes(q) ||
            (r.name  || '').toLowerCase().includes(q) ||
            (r.thaat || '').toLowerCase().includes(q))
        : allEntries;
    filtered.forEach(([key, r]) => {
            const pakkadText = (r.pakkad || []).map(entry => {
                // Support both {note,oct} (Firestore) and [note,oct] (legacy)
                const sw  = Array.isArray(entry) ? entry[0] : entry.note;
                const oct = Array.isArray(entry) ? entry[1] : entry.oct;
                return oct === 2 ? sw+"''" : oct === 1 ? sw+"'" : oct === -1 ? sw+"," : sw;
            }).join(' ');
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><code>${key}</code></td>
                <td><b>${r.name}</b></td>
                <td>${r.thaat || ''}</td>
                <td style="color:#e65100;font-weight:bold">${r.vadi || ''}</td>
                <td style="color:#1565c0;font-weight:bold">${r.samvadi || ''}</td>
                <td style="font-size:11px;color:#777">${r.jaati || ''}</td>
                <td style="font-size:11px;color:#555">${r.time || ''}</td>
                <td style="font-size:11px;color:#555">${pakkadText}</td>
                <td>
                    <button class="ra-edit-btn"   onclick="raAdminEdit('${key}')">Edit</button>
                    <button class="ra-delete-btn" onclick="raAdminDelete('${key}')">Delete</button>
                </td>`;
            tbody.appendChild(tr);
        });
    const countEl = document.getElementById('raga-admin-count');
    if (countEl) {
        const total = allEntries.length;
        countEl.textContent = q ? `${filtered.length} of ${total} ragas` : `${total} ragas`;
    }
}

// ── Add / Edit form ───────────────────────────────────────────────────────────
function raAdminNew() {
    document.getElementById('ra-edit-key-original').value = '';
    ['ra-key','ra-name','ra-thaat','ra-order','ra-allowed',
     'ra-vadi','ra-samvadi','ra-aroha','ra-avaroha','ra-pakkad',
     'ra-rasa','ra-skipped-aaroh','ra-skipped-avaroh'].forEach(id => {
        document.getElementById(id).value = '';
    });
    document.getElementById('ra-key').readOnly = false;
    document.getElementById('raga-edit-form').classList.add('open');
    document.getElementById('ra-key').focus();
}

function raAdminEdit(key) {
    const r = RagaData[key];
    if (!r) return;
    document.getElementById('ra-edit-key-original').value = key;
    document.getElementById('ra-key').value     = key;
    document.getElementById('ra-key').readOnly  = true;
    document.getElementById('ra-name').value    = r.name    || '';
    document.getElementById('ra-thaat').value   = r.thaat   || '';
    document.getElementById('ra-order').value   = r.order   || '';
    document.getElementById('ra-allowed').value = (r.allowed || []).join(' ');
    document.getElementById('ra-vadi').value    = r.vadi    || '';
    document.getElementById('ra-samvadi').value = r.samvadi || '';
    document.getElementById('ra-aroha').value   = r.aroha   || '';
    document.getElementById('ra-avaroha').value = r.avaroha || '';
    const pakkadText = (r.pakkad || []).map(entry => {
        const sw  = Array.isArray(entry) ? entry[0] : entry.note;
        const oct = Array.isArray(entry) ? entry[1] : entry.oct;
        return oct === 1 ? sw+"'" : oct === -1 ? sw+"," : sw;
    }).join(' ');
    document.getElementById('ra-pakkad').value         = pakkadText;
    document.getElementById('ra-jaati').value          = r.jaati        || '';
    document.getElementById('ra-time').value           = r.time         || '';
    document.getElementById('ra-rasa').value           = r.rasa         || '';
    document.getElementById('ra-skipped-aaroh').value  = r.skippedAaroh || '';
    document.getElementById('ra-skipped-avaroh').value = r.skippedAvaroh|| '';
    document.getElementById('raga-edit-form').classList.add('open');
    document.getElementById('ra-name').focus();
}

function raAdminCancelEdit() {
    document.getElementById('raga-edit-form').classList.remove('open');
}

// Parse pakkad text "Ni, Re Ga ma Sa'" → [["Ni",-1],["Re",0],["Ga",0],["ma",0],["Sa",1]]
function parsePakkadText(text) {
    return text.trim().split(/\s+/).filter(Boolean).map(token => {
        if (token.endsWith("'"))  return [token.slice(0,-1), 1];
        if (token.endsWith(','))  return [token.slice(0,-1), -1];
        return [token, 0];
    });
}

async function raAdminSave() {
    const keyOrig = document.getElementById('ra-edit-key-original').value;
    const key     = document.getElementById('ra-key').value.trim().toLowerCase().replace(/\s+/g,'');
    const name    = document.getElementById('ra-name').value.trim();
    const thaat   = document.getElementById('ra-thaat').value.trim();
    const order   = parseInt(document.getElementById('ra-order').value) || 0;
    const allowed = document.getElementById('ra-allowed').value.trim().split(/\s+/).filter(Boolean);
    const vadi    = document.getElementById('ra-vadi').value.trim();
    const samvadi = document.getElementById('ra-samvadi').value.trim();
    const aroha   = document.getElementById('ra-aroha').value.trim();
    const avaroha = document.getElementById('ra-avaroha').value.trim();
    // Firestore has no nested-array support, so pakkad is stored as an array
    // of {note, oct} objects (the shape engine.js's SEED_RAGAS and the read
    // side here already use) -- not the [swara, octave] tuples
    // parsePakkadText() returns for editing convenience.
    const pakkad  = parsePakkadText(document.getElementById('ra-pakkad').value)
                        .map(([note, oct]) => ({ note, oct }));

    if (!key || !name) { alert('Key and Name are required.'); return; }

    const jaati        = document.getElementById('ra-jaati').value.trim();
    const time         = document.getElementById('ra-time').value.trim();
    const rasa         = document.getElementById('ra-rasa').value.trim();
    const skippedAaroh = document.getElementById('ra-skipped-aaroh').value.trim();
    const skippedAvaroh= document.getElementById('ra-skipped-avaroh').value.trim();

    const ragaDoc = { name, thaat, order, allowed, vadi, samvadi, aroha, avaroha, pakkad,
                      jaati, time, rasa, skippedAaroh, skippedAvaroh };

    // Save to Firestore
    if (db) {
        try {
            await db.collection('ragas').doc(key).set(ragaDoc);
        } catch(e) {
            alert('Firestore save failed: ' + e.message); return;
        }
    }

    // Update in-memory RagaData and refresh UI
    RagaData[key] = ragaDoc;
    populateRagaDropdown();
    raAdminCancelEdit();
    raAdminRefreshTable();
    showToast(`Raga "${name}" saved ✓`);
}

async function raAdminDelete(key) {
    const r = RagaData[key];
    if (!r) return;
    if (!confirm(`Delete raga "${r.name}"? This cannot be undone.`)) return;

    if (db) {
        try {
            await db.collection('ragas').doc(key).delete();
        } catch(e) {
            alert('Firestore delete failed: ' + e.message); return;
        }
    }
    delete RagaData[key];
    populateRagaDropdown();
    raAdminRefreshTable();
    showToast(`Raga "${r.name}" deleted`);
}

// ── Seed built-in data into the Raga API ─────────────────────────────────────
async function raAdminSeed() {
    if (!db) { alert('Firestore not available.'); return; }
    const entries = Object.entries(SEED_RAGAS);
    if (!entries.length) { alert('SEED_RAGAS is empty \u2014 nothing to seed.'); return; }
    if (!confirm('This will write ' + entries.length + ' ragas to Firestore. Proceed?')) return;

    let count = 0, batchErrors = 0;
    const firstErrMsg = [];
    // Firestore writeBatch supports up to 500 ops; use chunks of 400 to be safe
    const CHUNK = 400;
    for (let i = 0; i < entries.length; i += CHUNK) {
        const batch = db.batch();
        entries.slice(i, i + CHUNK).forEach(([key, raga]) => {
            batch.set(db.collection('ragas').doc(key), raga);
        });
        try {
            await batch.commit();
            count += Math.min(CHUNK, entries.length - i);
        } catch(e) {
            batchErrors++;
            firstErrMsg.push(e.message);
            console.error('Batch seed error:', e);
        }
    }
    if (batchErrors > 0) {
        alert(
            'Seed failed!\n\nError: ' + firstErrMsg[0] +
            '\n\nMost likely cause: Firestore security rules block writes.\n' +
            'Deploy firestore.rules via Firebase CLI:\n  firebase deploy --only firestore:rules'
        );
    }
    await loadRagasFromFirestore();
    raAdminRefreshTable();
    const msg = batchErrors
        ? ('Seed failed \u2014 ' + batchErrors + ' batch error(s)')
        : ('Seeded ' + count + ' ragas to Firestore \u2713');
    showToast(msg);
}

// ── AI provider helpers ───────────────────────────────────────────────────────
const AI_PROVIDER_INFO = {
    gemini:     { label: 'Google Gemini API Key',  href: 'https://aistudio.google.com/app/apikey',  placeholder: 'AIza…' },
    mistral:    { label: 'Mistral API Key',         href: 'https://console.mistral.ai/api-keys/',    placeholder: 'your-mistral-key…' },
    openrouter: { label: 'OpenRouter API Key',      href: 'https://openrouter.ai/keys',              placeholder: 'sk-or-…' },
};

const MISTRAL_MODELS = new Set([
    'mistral-small-latest','mistral-medium-latest','mistral-large-latest',
    'open-mistral-7b','open-mixtral-8x7b','open-mixtral-8x22b',
    'codestral-latest','pixtral-large-latest'
]);

function aiGetProvider(model) {
    if (model.startsWith('gemini')) return 'gemini';
    if (MISTRAL_MODELS.has(model))  return 'mistral';
    return 'openrouter'; // models with org/name:free format go via OpenRouter
}

function aiUpdateKeyLabel() {
    const model    = document.getElementById('ai-model').value;
    const provider = aiGetProvider(model);
    const info     = AI_PROVIDER_INFO[provider];
    document.getElementById('ai-key-label-text').textContent = info.label;
    const link = document.getElementById('ai-key-link');
    link.href = info.href;
    document.getElementById('ai-api-key').placeholder = info.placeholder;
    // Restore saved key for this provider
    const saved = localStorage.getItem(`ai_key_${provider}`)
               || (provider === 'gemini' ? localStorage.getItem('gemini_api_key') : '');
    if (saved) document.getElementById('ai-api-key').value = saved;
    else       document.getElementById('ai-api-key').value = '';
}

// Unified LLM call — handles Gemini, Mistral, and OpenRouter
async function aiCallLLM(prompt, model, apiKey, opts = {}) {
    const provider = aiGetProvider(model);
    const temp     = opts.temperature  || 0.75;
    const maxTok   = opts.maxTokens    || 4096;

    let res;
    if (provider === 'gemini') {
        res = await fetch(
            `https://generativelanguage.googleapis.com/v1/models/${model}:generateContent?key=${apiKey}`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    contents: [{ parts: [{ text: prompt }] }],
                    generationConfig: { temperature: temp, maxOutputTokens: maxTok }
                })
            }
        );
    } else {
        // OpenAI-compatible format (Mistral + OpenRouter)
        const endpoint = provider === 'mistral'
            ? 'https://api.mistral.ai/v1/chat/completions'
            : 'https://openrouter.ai/api/v1/chat/completions';
        const headers = {
            'Content-Type':  'application/json',
            'Authorization': `Bearer ${apiKey}`,
        };
        if (provider === 'openrouter') {
            headers['HTTP-Referer'] = window.location.origin;
            headers['X-Title']      = 'Swar-Laya Studio';
        }
        res = await fetch(endpoint, {
            method: 'POST',
            headers,
            body: JSON.stringify({
                model,
                messages: [{ role: 'user', content: prompt }],
                temperature: temp,
                max_tokens: maxTok
            })
        });
    }

    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        const msg = err.error?.message || `HTTP ${res.status}`;
        if (res.status === 429 || msg.includes('quota') || msg.includes('RESOURCE_EXHAUSTED')
                               || msg.includes('rate_limit')) {
            throw new Error(
                `Quota / rate-limit exceeded for "${model}".\n\n` +
                `Try a different model or provider:\n` +
                `• Gemini free: aistudio.google.com/app/apikey\n` +
                `• Mistral free: console.mistral.ai\n` +
                `• OpenRouter free: openrouter.ai (no billing needed)`
            );
        }
        throw new Error(msg);
    }

    const data = await res.json();
    let text;
    if (provider === 'gemini') {
        text = (data.candidates?.[0]?.content?.parts?.[0]?.text || '').trim();
    } else {
        text = (data.choices?.[0]?.message?.content || '').trim();
    }
    // Strip markdown code fences
    text = text.replace(/^```(?:json)?\s*/i, '').replace(/\s*```\s*$/i, '').trim();
    return text;
}

// ── AI Compose ────────────────────────────────────────────────────────────────
function openAICompose() {
    // Populate raga list (all ragas, sorted by order)
    const sel = document.getElementById('ai-raga-select');
    sel.innerHTML = '<option value="">— choose raga —</option>';
    Object.entries(RagaData)
        .sort((a, b) => (a[1].order || 0) - (b[1].order || 0))
        .forEach(([key, r]) => {
            const o = document.createElement('option');
            o.value = key;
            o.textContent = r.name;
            sel.appendChild(o);
        });
    // Pre-select current raga if one is active
    const cur = document.getElementById('raga-select').value;
    if (cur) sel.value = cur;
    // Restore saved API key for the currently selected provider
    aiUpdateKeyLabel();
    document.getElementById('ai-status').textContent = '';
    document.getElementById('ai-generate-btn').disabled = false;
    document.getElementById('ai-compose-modal').style.display = 'flex';
}

function closeAICompose() {
    document.getElementById('ai-compose-modal').style.display = 'none';
}

function toggleAILyricsOptions() {
    const show = document.getElementById('ai-gen-lyrics').checked;
    document.getElementById('ai-lyrics-options').style.display = show ? 'block' : 'none';
}

async function doAICompose() {
    const ragaKey   = document.getElementById('ai-raga-select').value;
    const taalKey   = document.getElementById('ai-taal-select').value;
    const lineCount = parseInt(document.getElementById('ai-lines-count').value);
    const instr     = document.getElementById('ai-instrument').value;
    const model     = document.getElementById('ai-model').value;
    const emotion   = document.getElementById('ai-emotion').value.trim();
    const apiKey    = document.getElementById('ai-api-key').value.trim();

    const provider = aiGetProvider(model);
    if (!ragaKey) { alert('Please choose a Raga.'); return; }
    if (!apiKey)  {
        const info = AI_PROVIDER_INFO[provider];
        alert(`Please enter your ${info.label}.\n\nGet a free key at: ${info.href}`);
        return;
    }

    if (document.getElementById('ai-save-key').checked) {
        localStorage.setItem(`ai_key_${provider}`, apiKey);
        if (provider === 'gemini') localStorage.setItem('gemini_api_key', apiKey); // backward compat
    }

    const raga     = RagaData[ragaKey];
    const beats    = TaalBeats[taalKey] || parseInt(taalKey) || 8;
    const taalName = { '8':'Keherwa', '16':'Teentaal', '12':'Ektaal', '6':'Dadra',
                       '7':'Rupak', '8_bhajani':'Bhajani', '10':'Jhaptaal', '14':'Dhamar' }[taalKey] || taalKey;
    const allowed  = (raga.allowed || ['Sa','Re','Ga','ma','Pa','Dha','Ni']).join(' ');
    const emotionLine = emotion ? `\nEmotion / Intention: "${emotion}"` : '';

    const prompt = `You are an expert in Indian classical music composition. Generate a melodic instrumental composition in Raga ${raga.name} (${raga.thaat || ''} Thaat).

Raga details:
- Aroha: ${raga.aroha || 'Sa Re Ga ma Pa Dha Ni Sa'}
- Avaroha: ${raga.avaroha || 'Sa Ni Dha Pa ma Ga Re Sa'}
- Allowed notes: ${allowed}
- Vadi: ${raga.vadi || 'Sa'}, Samvadi: ${raga.samvadi || 'Pa'}${emotionLine}

Taal: ${taalName} (${beats} beats per line). Generate exactly ${lineCount} lines.

Return ONLY a valid JSON array (no markdown fences, no explanation) of exactly ${lineCount} objects.
Each object must have this exact structure:
{
  "section": "Sthayi",
  "taal_key": "${taalKey}",
  "line_instrument": "${instr}",
  "line_volume": 1.0,
  "lyrics": ["-", "-", ...],
  "notes": [["Sa"], ["Re"], ...],
  "octaves": [[0], [0], ...],
  "meend": [[false], [false], ...]
}

Critical rules:
- "lyrics" must have exactly ${beats} elements, all "-"
- "notes", "octaves", "meend" must each have exactly ${beats} elements
- Each element of notes/octaves/meend is an array with exactly 1 item
- notes[i][0] must be one of: Sa Re Ga ga ma Ma Pa Dha dha Ni ni
- Only use notes from the raga's allowed list: ${allowed}
- octaves: 0=middle octave, 1=upper octave, -1=lower octave
- Use octave 1 for upper Sa and notes reached by ascending beyond Pa
- Use octave -1 for notes below middle Sa
- Lines 1-2: Sthayi (stay in middle/lower octave, around Sa to Pa)
- Lines 3-4: Antara (ascend to upper octave, reach upper Sa)
- Lines 5+: Sanchari (explore full range)
- Follow raga grammar: respect characteristic phrases, avoid forbidden notes
- Add meend:true on 2-3 ascending steps per line for expressiveness
- Create musically coherent, singable melodic phrases`;

    const statusEl = document.getElementById('ai-status');
    const btn      = document.getElementById('ai-generate-btn');
    statusEl.innerHTML = '⏳ Composing… (may take 10–20 seconds)';
    btn.disabled = true;

    try {
        let text = await aiCallLLM(prompt, model, apiKey, { temperature: 0.75, maxTokens: 4096 });

        let lines;
        try {
            lines = JSON.parse(text);
        } catch(e) {
            // Try to extract the JSON array if surrounded by extra text
            const match = text.match(/\[[\s\S]*\]/);
            if (match) {
                lines = JSON.parse(match[0]);
            } else {
                console.error('Raw AI response:', text);
                throw new Error('Could not parse JSON from AI response. Try again.');
            }
        }

        if (!Array.isArray(lines) || lines.length === 0)
            throw new Error('AI returned no lines. Try again.');

        // ── Optional: generate lyrics via a second LLM call ──────────────────
        const genLyrics = document.getElementById('ai-gen-lyrics').checked;
        if (genLyrics) {
            statusEl.innerHTML = '✅ Notation ready — ✍️ Generating lyrics…';
            const lang = document.querySelector('input[name="ai-lyrics-lang"]:checked')?.value || 'Hindi';

            const sections = lines.map((l, i) => l.section || (i < 2 ? 'Sthayi' : 'Antara'));
            const sectionDesc = sections.map((s, i) => `Line ${i+1}: ${s}`).join(', ');

            const lyricsPrompt = `You are a poet and lyricist specializing in Indian classical music.

Write ${lang} song lyrics for a composition in Raga ${raga.name}.
Theme / Intention: ${emotion || 'devotional and melodic, capturing the essence of the raga'}
Structure: ${lines.length} lines — ${sectionDesc}
Each line has exactly ${beats} beat positions.

Rules:
- Return ONLY a valid JSON array (no markdown, no explanation)
- The array has exactly ${lines.length} inner arrays
- Each inner array has exactly ${beats} strings — one syllable or short word per beat
- Use "${lang}" language and script
- Sthayi lines (1-2): the main refrain/chorus — repeated, memorable phrase
- Antara lines (3-4): the verse — elaborates the theme, slightly elevated emotion
- Sanchari lines (5+): free exploration of the theme
- Use "-" for any beat that is a silent rest or held sustain
- Make syllables singable — prefer open vowel sounds (अ, आ, ए, ओ) on long notes
- Lyrics must be poetic, emotionally resonant, and feel natural when sung
- Avoid forced rhyme; prioritise melodic flow and meaning

Example output format for ${beats}-beat lines:
[["शब्द", "शब्द", ...], ["शब्द", ...]]

Return only the JSON array.`;

            try {
                const lyrText = await aiCallLLM(lyricsPrompt, model, apiKey, { temperature: 0.85, maxTokens: 2048 });
                {

                    let lyrArr;
                    try { lyrArr = JSON.parse(lyrText); }
                    catch(e) {
                        const m = lyrText.match(/\[[\s\S]*\]/);
                        if (m) lyrArr = JSON.parse(m[0]);
                    }

                    if (Array.isArray(lyrArr)) {
                        lines.forEach((line, i) => {
                            if (Array.isArray(lyrArr[i])) {
                                // Ensure exact beat count — trim or pad with "-"
                                const raw = lyrArr[i].map(s => (typeof s === 'string' && s.trim()) ? s.trim() : '-');
                                line.lyrics = raw.slice(0, beats);
                                while (line.lyrics.length < beats) line.lyrics.push('-');
                            }
                        });
                        statusEl.innerHTML = '✅ Notation + Lyrics ready — loading into song sheet…';
                    } else {
                        console.warn('Lyrics parse failed — keeping "-" placeholders');
                    }
                }
            } catch(lyricErr) {
                // Non-fatal: notation still loads, just without lyrics
                console.warn('Lyrics generation error (non-fatal):', lyricErr);
            }
        }

        // Load each line into the song sheet (lyrics included if generated)
        lines.forEach(line => addLine(line));

        statusEl.innerHTML = genLyrics
            ? `✅ Added ${lines.length} lines with notation + lyrics!`
            : `✅ Added ${lines.length} lines to the song sheet!`;
        setTimeout(() => closeAICompose(), 1800);

    } catch(e) {
        statusEl.innerHTML = `❌ Error: ${e.message}`;
        console.error('AI Compose error:', e);
        btn.disabled = false;
    }
}

// ── Export JSON ───────────────────────────────────────────────────────────────
function raAdminExport() {
    const json = JSON.stringify(RagaData, null, 2);
    const blob = new Blob([json], { type:'application/json' });
    const a    = document.createElement('a');
    a.href     = URL.createObjectURL(blob);
    a.download = 'ragas.json';
    a.click();
    URL.revokeObjectURL(a.href);
}
