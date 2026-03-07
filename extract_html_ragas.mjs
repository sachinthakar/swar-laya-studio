// extract_html_ragas.mjs
// Extracts RagaData from the HTML file and merges with consolidated batch data
import { readFileSync, writeFileSync } from 'fs';
import { createContext, runInContext } from 'vm';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));

const htmlPath = join(__dirname, 'Swar-Laya Studio_v3.html');
const consolidatedPath = join(__dirname, 'ragas_consolidated.json');
const outputPath = join(__dirname, 'ragas_all.json');

// 1. Extract RagaData from HTML via VM evaluation
const html = readFileSync(htmlPath, 'utf8');

// Find the RagaData block
const startMatch = html.match(/let RagaData\s*=\s*\{/);
if (!startMatch) throw new Error("Could not find let RagaData = {");

let startIdx = startMatch.index + startMatch[0].length - 1; // position of '{'
let depth = 0, endIdx = startIdx;
for (let i = startIdx; i < html.length; i++) {
  if (html[i] === '{') depth++;
  else if (html[i] === '}') {
    depth--;
    if (depth === 0) { endIdx = i + 1; break; }
  }
}

const ragaDataStr = html.slice(startIdx, endIdx);

// Math.pow is used in SwaraMap but not in RagaData - safe to evaluate directly
const sandbox = { Math };
createContext(sandbox);
let htmlRagas = {};
try {
  runInContext(`htmlRagas = ${ragaDataStr};`, sandbox);
  htmlRagas = sandbox.htmlRagas;
  console.log(`Extracted ${Object.keys(htmlRagas).length} ragas from HTML`);
} catch (e) {
  console.error('Failed to parse HTML RagaData:', e.message);
  process.exit(1);
}

// 2. Load consolidated batch data
const consolidatedRagas = JSON.parse(readFileSync(consolidatedPath, 'utf8'));
console.log(`Loaded ${consolidatedRagas.length} ragas from consolidated batch files`);

// 3. Build final set: start with HTML ragas (the original 23), overlay batch ragas
// Keys to remove
const REMOVE_KEYS = new Set(['bilaval', 'raagmalika']);

const allRagas = {};

// Add HTML ragas first (they may have good pakkad/data for original ragas)
for (const [key, raga] of Object.entries(htmlRagas)) {
  if (REMOVE_KEYS.has(key)) { console.log(`  REMOVING from HTML: ${key}`); continue; }
  allRagas[key] = { ...raga, key };
}

// Overlay batch data (batch data is more complete for new ragas)
for (const raga of consolidatedRagas) {
  const key = raga.key;
  if (REMOVE_KEYS.has(key)) continue;
  // Only overlay if raga not already in allRagas OR if batch data has more complete info
  // Prefer batch data for ragas with order >= 110 (the researched ones)
  // Keep HTML data for ragas already in HTML with order < 110 (original 23)
  if (!allRagas[key] || (raga.order >= 110)) {
    allRagas[key] = raga;
  }
}

// 4. Sort by order
const sorted = Object.values(allRagas).sort((a, b) => (a.order || 9999) - (b.order || 9999));

// 5. Validate and clean pakkad notes
const VALID_NOTES = new Set(['Sa','re','Re','ga','Ga','Ma','ma','Pa','dha','Dha','ni','Ni']);

function cleanNote(note) {
  // NFD/NFC normalize and strip combining characters
  const cleaned = note.normalize('NFD').replace(/[\u0300-\u036f\u1dc0-\u1dff\u20d0-\u20ff]/g, '').trim();
  if (cleaned === 'S') return 'Sa'; // lone S = upper Sa
  return cleaned;
}

let warnings = 0;
for (const raga of sorted) {
  if (Array.isArray(raga.pakkad)) {
    raga.pakkad = raga.pakkad.map(([note, oct]) => {
      const cn = cleanNote(String(note));
      if (!VALID_NOTES.has(cn)) {
        console.warn(`  WARN: raga ${raga.key} pakkad note "${note}" → "${cn}" not valid`);
        warnings++;
      }
      return [cn, oct];
    });
  }
}

// 6. Write output
writeFileSync(outputPath, JSON.stringify(sorted, null, 2), 'utf8');
console.log(`\nWrote ${sorted.length} ragas to ${outputPath}`);
if (warnings) console.log(`${warnings} pakkad warnings (check output)`);

// Print summary
console.log('\nFinal raga list:');
for (const r of sorted) {
  console.log(`  ${String(r.order).padStart(5)}  ${r.key.padEnd(40)} ${r.name}`);
}
