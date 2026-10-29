// Swar-Laya Studio v5 — audio engine: state, tuning, samples, tabla/dholak, synth, playback, tanpura
// (split from the single-file studio.html; scripts load in order and share globals)

// ─── State ───────────────────────────────────────────────────────────────────
let audioCtx       = null;
let playbackTimer  = null;
let tanpuraTimer   = null;
let lastPlayingLine = null;   // tracks which line is playing for auto-scroll
let tanpuraNodes   = [];
let isTanpuraOn    = false;

// Monophonic synth nodes (shared, switches instrument per line)
let mainOsc        = null;
let mainGain       = null;
let mainFilter     = null;
let harmonics      = [];   // [{osc, gain, ratio}]

// Per-instrument effects (created in setupSynth, destroyed in stopSynth)
let vibratoLFO     = null;   // pitch-vibrato oscillator (violin, flute)
let vibratoGain    = null;   // controls vibrato depth in cents
let fluteNoiseNode = null;   // looping noise source for bansuri breath
let fluteNoiseGain = null;   // mixes breath noise into the signal chain

// Master bus
let masterBus      = null;
let masterVolNode  = null;   // post-compressor gain — allows boost beyond 1.0×
let sampleBus      = null;   // gain node for all sampled notes — zeroed on Stop
let dryGain        = null;
let wetGain        = null;
let reverbNode     = null;
let compressor     = null;
let noiseBuffer    = null;   // short noise buffer (tabla transients)
let breathBuffer   = null;   // longer loopable noise buffer (flute breath)
let lastPlayedFreq = null;   // tracks prev note freq for meend glide with samples
let legatoEnv      = null;   // last sample env in legato mode — faded out on next note

// ─── Tuning ──────────────────────────────────────────────────────────────────
// Equal-temperament semitone ratios (12th root of 2)
const SwaraMap = {
    "-":  null,
    "~":  null,   // sustain / hold — continue previous note, do not re-attack
    "Sa": 1,
    "re": Math.pow(2, 1/12),   // komal Re  (minor 2nd)
    "Re": Math.pow(2, 2/12),   // shuddha Re (major 2nd)
    "ga": Math.pow(2, 3/12),   // komal Ga  (minor 3rd)
    "Ga": Math.pow(2, 4/12),   // shuddha Ga (major 3rd)
    "Ma": Math.pow(2, 5/12),   // shuddha Ma (perfect 4th)
    "ma": Math.pow(2, 6/12),   // tivra Ma  (tritone)
    "Pa": Math.pow(2, 7/12),   // Pa        (perfect 5th)
    "dha": Math.pow(2, 8/12),  // komal Dha (minor 6th)
    "Dha": Math.pow(2, 9/12),  // shuddha Dha (major 6th)
    "ni": Math.pow(2, 10/12),  // komal Ni  (minor 7th)
    "Ni": Math.pow(2, 11/12)   // shuddha Ni (major 7th)
};

const TaalBols = {
    "4":        ["Dha","Gi","Na","Ti"],
    "8":        ["Dha","Gi","Na","Ti","Na","Ka","Dhi","Na"],
    "8_bhajani":["Dhin","Dha","Dhin","Dhinna","Tin","Ta","Tin","Tina"],
    "6":        ["Dha","Dhi","Na","Dha","Tu","Na"],
    "7":           ["Ti","Ti","Na","Dhi","Na","Dhi","Na"],
    "12":          ["Dhin","Dhin","Dha","Ge","Tirkit","Tirkit","Tu","Na","Kat","Ta","Dhin","Dha"],
    "16":          ["Dha","Dhin","Dhin","Dha","Dha","Dhin","Dhin","Dha","Dha","Tin","Tin","Ta","Ta","Dhin","Dhin","Dha"],
    "10":          ["Dhi","Na","Dhi","Dhi","Na","Ti","Na","Dhi","Dhi","Na"],
    "12_chautaal": ["Dha","Dha","Din","Ta","Kita","Dha","Din","Ta","Tita","Kata","Gadi","Gana"],
    "14":          ["Ka","Da","Hi","Da","Hi","Da","Hi","Ka","Ti","Ta","Ka","Ti","Da","Hi"]
};

// Proper beat count lookup (avoids fragile parseInt("8_bhajani"))
const TaalBeats = {
    "4": 4, "8": 8, "8_bhajani": 8, "6": 6, "7": 7, "12": 12, "16": 16,
    "10": 10, "12_chautaal": 12, "14": 14
};

// Khali (empty/waved-hand) beat indices (0-based) for each taal
const TaalKhali = {
    "4":           [2],         // Keherwa 4: beat 3
    "8":           [4],         // Keherwa 8: beat 5
    "8_bhajani":   [4],         // Bhajani:   beat 5
    "6":           [3],         // Dadra:     beat 4
    "7":           [1, 2],      // Rupak:     vibhag-1 (beats 2-3) is khali
    "12":          [6],         // Ektaal:    beat 7
    "16":          [8],         // Teentaal:  beat 9
    "10":          [5],         // Jhaptaal:  beat 6
    "12_chautaal": [2, 6],      // Chautaal:  beats 3, 7
    "14":          [5],         // Dhamar:    beat 6
};

// Dholak bol patterns (one per beat) for each taal.
// Right head (dagga): Na, Ti, Ka, Te  |  Left head (bayi): Ge
// Combined strokes:  Dha (both open), Dhin (both resonant)
const DholakBols = {
    "4":           ["Dha","Ge","Na","Ge"],
    "8":           ["Dha","Ge","Na","Ge","Dha","Ge","Na","Ge"],
    "8_bhajani":   ["Dha","Ge","Dha","Dha","Ge","Dha","Ge","Na"],
    "6":           ["Dha","Dhin","Na","Dha","Ti","Na"],
    "7":           ["Ti","Ti","Na","Dhin","Na","Dhin","Na"],
    "12":          ["Dha","Dha","Ge","Na","Dha","Ge","Na","Dha","Ge","Na","Ge","Na"],
    "16":          ["Dha","Dhin","Dhin","Dha","Dha","Dhin","Dhin","Dha","Dha","Ti","Ti","Na","Na","Dhin","Dhin","Dha"],
    "10":          ["Dha","Ge","Na","Ge","Na","Dha","Ge","Ti","Na","Na"],
    "12_chautaal": ["Dha","Dha","Ge","Na","Dha","Ge","Na","Ge","Ti","Na","Ge","Na"],
    "14":          ["Ka","Dha","Ge","Ti","Na","Ka","Dha","Ge","Ti","Na","Ka","Ti","Na","Na"],
};

// ─── Raga Definitions ────────────────────────────────────────────────────────
// allowed: swar names that appear in this raga (uppercase = shuddha, lowercase = komal/tivra)
// vadi: most prominent swara; samvadi: second most prominent
// pakkad: [[swarName, octave], ...] — octave: -1=lower, 0=middle, 1=upper
// thaat: parent thaat name (used to group the dropdown)
// order: sort order within thaat group
// jaati: "AROHA-AVAROHA" note-count classification (ODAV=5, SHADAV=6, SAMPURNA=7)
// time: time of day to sing/play; rasa: emotional quality
// skippedAaroh / skippedAvaroh: notes omitted in ascent / descent (vakra/varja)
let RagaData = {};
// Snapshot of built-in ragas (used by "Seed from Built-in" in the admin panel)
// Seed data for Firestore (287 ragas). This is the ONLY raga data in the HTML.
// RagaData is always loaded from Firestore — this is only used by the Seed button.
const SEED_RAGAS = {"yaman":{"name":"Yaman","thaat":"Kalyan","order":10,"allowed":["Sa","Re","Ga","ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa Re Ga ma Pa Dha Ni Ṡa","avaroha":"Ṡa Ni Dha Pa ma Ga Re Sa","pakkad":[{"note":"Ni","oct":-1},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPURNA-SAMPURNA","time":"Evening","rasa":"Shant","skippedAaroh":"","skippedAvaroh":""},"marwa":{"name":"Marwa","thaat":"Kalyan","order":11,"allowed":["Sa","re","Ga","ma","Dha","Ni"],"vadi":"re","samvadi":"Dha","aroha":"Sa re Ga ma Dha Ni Ṡa","avaroha":"Ṡa Ni Dha ma Ga re Sa","pakkad":[{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"ma","oct":0},{"note":"Ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0}],"jaati":"SHADAV-SHADAV","time":"Late Afternoon","rasa":"","skippedAaroh":"S P","skippedAvaroh":"S P"},"bilawal":{"name":"Bilawal","thaat":"Bilawal","order":20,"allowed":["Sa","Re","Ga","Ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa Re Ga Ma Pa Dha Ni Ṡa","avaroha":"Ṡa Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPURNA-SAMPURNA","time":"Morning","rasa":"","skippedAaroh":"","skippedAvaroh":""},"bhinnashadajkaushikdhwanihindoli":{"name":"Bhinna Shadaj-Kaushik Dhwani-Hindoli","thaat":"Bilawal","order":20,"allowed":["Sa","Ga","Ma","Dha","ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa Ga Ma Dha ni Sȧ","avaroha":"Sȧ ni Dha Ma Ga Sa","pakkad":[{"note":"Sa","oct":1},{"note":"ni","oct":0},{"note":"Dha","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Sa","oct":0}],"jaati":"ODAV-ODAV","time":"Night","rasa":"Shant","skippedAaroh":"R P","skippedAvaroh":"R P"},"bhoopali":{"name":"Bhoopali","thaat":"Bilawal","order":21,"allowed":["Sa","Re","Ga","Pa","Dha"],"vadi":"Ga","samvadi":"Dha","aroha":"Sa Re Ga Pa Dha Ṡa","avaroha":"Ṡa Dha Pa Ga Re Sa","pakkad":[{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Sa","oct":1}],"jaati":"ODAV-ODAV","time":"Evening","rasa":"Shant","skippedAaroh":"M N","skippedAvaroh":"M N"},"bilawalalhaiya":{"name":"Bilawal-Alhaiya","thaat":"Bilawal","order":24,"allowed":["Sa","Re","Ga","Ma","Pa","Dha","ni"],"vadi":"Dha","samvadi":"Ga","aroha":"Sa Ga Re Ga Pa ni Dha ni Ṡa","avaroha":"Ṡa ni Dha ni Dha Pa Ma Ga Ma Re Sa","pakkad":[{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SHADAV-SAMPURNA","time":"Late Morning","rasa":"Shant","skippedAaroh":"","skippedAvaroh":""},"bilawaldevgiri":{"name":"Bilawal-Devgiri","thaat":"Bilawal","order":25,"allowed":["Sa","Re","Ga","Ma","Pa","Dha","ni","Ni"],"vadi":"Sa","samvadi":"Pa","aroha":"Sa Re Ga Pa Dha Ni Ṡa","avaroha":"Ṡa Ni Dha Pa Dha ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Ga","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SHADAV-SAMPURNA","time":"Morning","rasa":"","skippedAaroh":"M","skippedAvaroh":""},"deshkar":{"name":"Deshkar","thaat":"Bilawal","order":26,"allowed":["Sa","Re","Ga","Pa","Dha"],"vadi":"Dha","samvadi":"Ga","aroha":"Sa Re Ga Pa Dha Ṡa","avaroha":"Ṡa Dha Pa Ga Pa Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ga","oct":0},{"note":"Pa","oct":0},{"note":"Sa","oct":1},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"ODAV-ODAV","time":"Morning","rasa":"","skippedAaroh":"M N","skippedAvaroh":"M N"},"bihag":{"name":"Bihag","thaat":"Bilawal","order":27,"allowed":["Sa","Re","Ga","Ma","ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"ˌNi Sa Ga Ma Pa Ni Ṡa","avaroha":"Ṡa Ni Dha Pa ma Ga Re Sa","pakkad":[{"note":"Pa","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0}],"jaati":"ODAV-SAMPURNA","time":"Night","rasa":"","skippedAaroh":"R D","skippedAvaroh":""},"bhinnashadaj":{"name":"Bhinna Shadaj / Kaushik Dhwani","thaat":"Bilawal","order":29,"allowed":["Sa","Ga","Ma","Dha","ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa Ga Ma Dha ni Ṡa","avaroha":"Ṡa ni Dha Ma Ga Sa","pakkad":[{"note":"Sa","oct":1},{"note":"ni","oct":0},{"note":"Dha","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Sa","oct":0}],"jaati":"ODAV-ODAV","time":"Night","rasa":"Shant","skippedAaroh":"R P","skippedAvaroh":"R P"},"khamaj":{"name":"Khamaj","thaat":"Khamaj","order":30,"allowed":["Sa","Re","Ga","Ma","Pa","Dha","ni","Ni"],"vadi":"Ga","samvadi":"ni","aroha":"Sa Re Ga Ma Pa Dha ni Ṡa","avaroha":"Ṡa Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPURNA-SAMPURNA","time":"Evening","rasa":"Shringara","skippedAaroh":"","skippedAvaroh":""},"aabhogi":{"name":"Aabhogi","thaat":"Kafi","order":41,"allowed":["Sa","Re","ga","Ma","Dha","ni"],"vadi":"Sa","samvadi":"Ma","aroha":"Sa Re ga Ma Dha Ṡa","avaroha":"Ṡa ni Dha Ma ga Re Sa","pakkad":[{"note":"Re","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Dha","oct":0},{"note":"ni","oct":0},{"note":"Dha","oct":0},{"note":"Ma","oct":0},{"note":"ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"ODAV-SHADAV","time":"Morning","rasa":"","skippedAaroh":"P N","skippedAvaroh":"P"},"asavari":{"name":"Asavari","thaat":"Asavari","order":50,"allowed":["Sa","Re","ga","Ma","Pa","dha","ni"],"vadi":"dha","samvadi":"ga","aroha":"Sa Re ga Ma Pa dha ni Ṡa","avaroha":"Ṡa ni dha Pa Ma ga Re Sa","pakkad":[{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPURNA-SAMPURNA","time":"Morning","rasa":"Karuna","skippedAaroh":"","skippedAvaroh":""},"bhairavi":{"name":"Bhairavi","thaat":"Bhairavi","order":60,"allowed":["Sa","re","ga","Ma","Pa","dha","ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa re ga Ma Pa dha ni Ṡa","avaroha":"Ṡa ni dha Pa Ma ga re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1}],"jaati":"SAMPURNA-SAMPURNA","time":"Morning","rasa":"Karuna","skippedAaroh":"","skippedAvaroh":""},"malkauns":{"name":"Malkauns","thaat":"Bhairavi","order":61,"allowed":["Sa","ga","Ma","dha","ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa ga Ma dha ni Ṡa","avaroha":"Ṡa ni dha Ma ga Sa","pakkad":[{"note":"Sa","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"ni","oct":0},{"note":"dha","oct":0},{"note":"Ma","oct":0},{"note":"ga","oct":0},{"note":"Sa","oct":0}],"jaati":"ODAV-ODAV","time":"Late Night","rasa":"Veer","skippedAaroh":"R P","skippedAvaroh":"R P"},"todi":{"name":"Todi","thaat":"Todi","order":80,"allowed":["Sa","re","ga","ma","Pa","dha","Ni"],"vadi":"dha","samvadi":"ga","aroha":"Sa re ga ma Pa dha Ni Ṡa","avaroha":"Ṡa Ni dha Pa ma ga re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"ga","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1}],"jaati":"SAMPURNA-SAMPURNA","time":"Late Morning","rasa":"","skippedAaroh":"","skippedAvaroh":""},"hamsadhvani":{"name":"Hamsadhvani","thaat":"Bilawal","order":90,"allowed":["Sa","Re","Ga","Pa","Ni"],"vadi":"Pa","samvadi":"Re","aroha":"Sa Re Ga Pa Ni Ṡa","avaroha":"Ṡa Ni Pa Ga Re Sa","pakkad":[{"note":"Ni","oct":-1},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Pa","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Pa","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Pa","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"ODAV-ODAV","time":"Evening","rasa":"","skippedAaroh":"M D","skippedAvaroh":"M D"},"aabhogikanada":{"name":"Aabhogi-Kanada","order":110,"thaat":"Bhairavi","allowed":["Sa","re","ga","Ma","dha"],"vadi":"ga","samvadi":"dha","aroha":"Sa ga Ma dha Sȧ","avaroha":"Sȧ dha Ma ga re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"dha","oct":0},{"note":"ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0}],"jaati":"ODAV-ODAV","time":"Anytime","rasa":"Shant","skippedAaroh":"Re Pa Ni","skippedAvaroh":"Pa Ni Re"},"aahiri":{"name":"Aahiri","order":120,"thaat":"Bhairavi","allowed":["Sa","Re","ga","Ma","Pa","dha","ni"],"vadi":"ga","samvadi":"dha","aroha":"Sa Re ga Ma Pa dha ni Sȧ","avaroha":"Sȧ ni dha Pa Ma ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Karuna","skippedAaroh":"","skippedAvaroh":""},"aarabhi":{"name":"Aarabhi","order":130,"thaat":"Khamaj","allowed":["Sa","Re","Ga","Ma","Pa","Dha","ni"],"vadi":"Pa","samvadi":"Re","aroha":"Sa Ma Pa ni Sȧ","avaroha":"Sȧ ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0}],"jaati":"ODAV-SAMPUNRA","time":"Afternoon","rasa":"Shringar","skippedAaroh":"Re Ga Dha","skippedAvaroh":""},"adana":{"name":"Adana","order":140,"thaat":"Asawari","allowed":["Sa","Re","ga","Ma","Pa","dha","Ni"],"vadi":"Pa","samvadi":"Re","aroha":"Sa Re Ma Pa Ni Sȧ","avaroha":"Sȧ Ni dha Pa Ma ga Re Sa","pakkad":[{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"ODAV-SAMPUNRA","time":"Night","rasa":"Veera","skippedAaroh":"ga dha","skippedAvaroh":""},"adanamalhar":{"name":"Adana-Malhar","order":150,"thaat":"Asawari","allowed":["Sa","Re","ga","Ma","Pa","dha","Ni"],"vadi":"Pa","samvadi":"Sa","aroha":"Sa Re Ma Pa Ni Sȧ","avaroha":"Sȧ Ni dha Pa Ma ga Re Sa","pakkad":[{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"ODAV-SAMPUNRA","time":"Rainy Season","rasa":"Shringar","skippedAaroh":"ga dha","skippedAvaroh":""},"amritvarshini":{"name":"Amritvarshini","order":160,"thaat":"Kalyan","allowed":["Sa","Ga","ma","Pa","Ni"],"vadi":"Pa","samvadi":"Ga","aroha":"Sa Ga ma Pa Ni Sȧ","avaroha":"Sȧ Ni Pa ma Ga Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Pa","oct":0},{"note":"ma","oct":0},{"note":"Ga","oct":0},{"note":"Sa","oct":0}],"jaati":"ODAV-ODAV","time":"Night","rasa":"Bhakti","skippedAaroh":"Re Dha","skippedAvaroh":"Re Dha"},"asawari":{"name":"Asawari","order":170,"thaat":"Asawari","allowed":["Sa","Re","ga","Ma","Pa","dha","ni"],"vadi":"dha","samvadi":"ga","aroha":"Sa Re Ma Pa dha Sȧ","avaroha":"Sȧ ni dha Pa Ma ga Re Sa","pakkad":[{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"ODAV-SAMPUNRA","time":"Morning","rasa":"Karuna","skippedAaroh":"ga ni","skippedAvaroh":""},"asawarikomalrishabh":{"name":"Asawari-Komal Rishabh","order":180,"thaat":"Bhairavi","allowed":["Sa","re","ga","Ma","Pa","dha","ni"],"vadi":"dha","samvadi":"ga","aroha":"Sa re Ma Pa dha Sȧ","avaroha":"Sȧ ni dha Pa Ma ga re Sa","pakkad":[{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0}],"jaati":"ODAV-SAMPUNRA","time":"Morning","rasa":"Karuna","skippedAaroh":"ga ni","skippedAvaroh":""},"bageshri":{"name":"Bageshri","order":190,"thaat":"Kafi","allowed":["Sa","Re","ga","Ma","Pa","dha","ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa ga Ma dha ni Sȧ","avaroha":"Sȧ ni dha Ma Pa Ma ga Re Sa","pakkad":[{"note":"Ma","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"dha","oct":0},{"note":"Ma","oct":0},{"note":"ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SHADAV-SAMPUNRA","time":"Night","rasa":"Shringar","skippedAaroh":"Re Pa","skippedAvaroh":""},"bahar":{"name":"Bahar","order":200,"thaat":"Kafi","allowed":["Sa","Re","ga","Ma","Pa","Dha","ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa ga Ma Pa Dha ni Sȧ","avaroha":"Sȧ ni Dha Pa Ma ga Re Sa","pakkad":[{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Spring","rasa":"Shringar","skippedAaroh":"","skippedAvaroh":""},"bairagi":{"name":"Bairagi","order":210,"thaat":"Bhairav","allowed":["Sa","re","Ma","Pa","ni"],"vadi":"Sa","samvadi":"Pa","aroha":"Sa re Ma Pa ni Sȧ","avaroha":"Sȧ ni Pa Ma re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0},{"note":"ni","oct":-1},{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0}],"jaati":"ODAV-ODAV","time":"Morning","rasa":"Bhakti","skippedAaroh":"Re Ga Dha","skippedAvaroh":"Re Ga Dha"},"baradi":{"name":"Baradi","order":220,"thaat":"Bhairav","allowed":["Sa","re","Ga","Ma","Pa","Dha","ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa re Ga Ma Pa Dha ni Sȧ","avaroha":"Sȧ ni Dha Pa Ma Ga re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Karuna","skippedAaroh":"","skippedAvaroh":""},"barawa":{"name":"Barawa","order":230,"thaat":"Bhairav","allowed":["Sa","re","Ga","Ma","Pa","Dha","Ni"],"vadi":"Pa","samvadi":"Sa","aroha":"Sa re Ga Ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Shant","skippedAaroh":"","skippedAvaroh":""},"basant":{"name":"Basant","order":240,"thaat":"Poorvi","allowed":["Sa","re","Ga","ma","Pa","dha","Ni"],"vadi":"Sa","samvadi":"Pa","aroha":"Sa re Ga ma Pa Ni Sȧ","avaroha":"Sȧ Ni dha Pa ma Ga re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Ni","oct":-1},{"note":"re","oct":0},{"note":"Sa","oct":0},{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0}],"jaati":"SHADAV-SAMPUNRA","time":"Spring","rasa":"Shringar","skippedAaroh":"dha","skippedAvaroh":""},"basantbahar":{"name":"Basant-Bahar","order":250,"thaat":"Poorvi","allowed":["Sa","Re","re","ga","Ga","Ma","ma","Pa","Dha","dha","Ni","ni"],"vadi":"Pa","samvadi":"Sa","aroha":"Sa re Ga ma Pa Ni Sȧ","avaroha":"Sȧ Ni dha Pa Ma ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Ni","oct":-1},{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1}],"jaati":"SHADAV-SAMPUNRA","time":"Spring","rasa":"Shringar","skippedAaroh":"dha","skippedAvaroh":""},"basantmukhari":{"name":"Basant-Mukhari","order":260,"thaat":"Bhairav","allowed":["Sa","re","Ga","ma","Pa","dha","Ni"],"vadi":"ma","samvadi":"Sa","aroha":"Sa re Ga ma Pa Ni Sȧ","avaroha":"Sȧ Ni dha Pa ma Ga re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"ma","oct":0},{"note":"Ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0}],"jaati":"SHADAV-SAMPUNRA","time":"Morning","rasa":"Bhakti","skippedAaroh":"dha","skippedAvaroh":""},"bhairav":{"name":"Bhairav","order":270,"thaat":"Bhairav","allowed":["Sa","re","Ga","Ma","Pa","dha","Ni"],"vadi":"dha","samvadi":"re","aroha":"Sa re Ga Ma Pa dha Ni Sȧ","avaroha":"Sȧ Ni dha Pa Ma Ga re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0},{"note":"dha","oct":-1},{"note":"Ni","oct":-1},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Bhakti","skippedAaroh":"","skippedAvaroh":""},"bhairavahir":{"name":"Bhairav-Ahir","order":280,"thaat":"Bhairav","allowed":["Sa","re","Ga","Ma","Pa","dha","Ni"],"vadi":"Pa","samvadi":"Sa","aroha":"Sa re Ga Ma Pa dha Ni Sȧ","avaroha":"Sȧ Ni dha Pa Ma Ga re Sa","pakkad":[{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Shant","skippedAaroh":"","skippedAvaroh":""},"bhairavand":{"name":"Bhairav-Anand","order":290,"thaat":"Bhairav","allowed":["Sa","re","Ga","Ma","Pa","dha","Ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa re Ga Ma Pa dha Ni Sȧ","avaroha":"Sȧ Ni dha Pa Ma Ga re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Shant","skippedAaroh":"","skippedAvaroh":""},"bhairavasa":{"name":"Bhairav-Asa","order":300,"thaat":"Bhairav","allowed":["Sa","re","Ga","Ma","Pa","dha","Ni"],"vadi":"dha","samvadi":"re","aroha":"Sa re Ga Ma Pa dha Ni Sȧ","avaroha":"Sȧ Ni dha Pa Ma Ga re Sa","pakkad":[{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Bhakti","skippedAaroh":"","skippedAvaroh":""},"bhairavbahar":{"name":"Bhairav-Bahar","order":310,"thaat":"Bhairav","allowed":["Sa","re","Ga","Ma","Pa","dha","Ni","ni"],"vadi":"dha","samvadi":"re","aroha":"Sa re Ga Ma Pa dha Ni Sȧ","avaroha":"Sȧ ni dha Pa Ma Ga re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Shringar","skippedAaroh":"","skippedAvaroh":""},"bhairavbairagi":{"name":"Bhairav-Bairagi","order":320,"thaat":"Bhairav","allowed":["Sa","re","Ma","Pa","ni"],"vadi":"Sa","samvadi":"Pa","aroha":"Sa re Ma Pa ni Sȧ","avaroha":"Sȧ ni Pa Ma re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0},{"note":"ni","oct":-1},{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1}],"jaati":"ODAV-ODAV","time":"Morning","rasa":"Bhakti","skippedAaroh":"Ga Dha","skippedAvaroh":"Ga Dha"},"bhairavbeehad":{"name":"Bhairav-Beehad","order":330,"thaat":"Bhairav","allowed":["Sa","re","Ga","Ma","ma","Pa","dha","Ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa re Ga Ma Pa dha Ni Sȧ","avaroha":"Sȧ Ni dha Pa ma Ga re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"ma","oct":0},{"note":"Ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Shant","skippedAaroh":"","skippedAvaroh":""},"bhairavbhatiyar":{"name":"Bhairav-Bhatiyar","order":340,"thaat":"Bhairav","allowed":["Sa","re","Ga","Ma","Pa","Dha","dha","Ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa re Ga Ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni dha Pa Ma Ga re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"dha","oct":0},{"note":"Pa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Bhakti","skippedAaroh":"","skippedAvaroh":""},"bhairavbhavmat":{"name":"Bhairav-Bhavmat","order":350,"thaat":"Bhairav","allowed":["Sa","re","Ga","Ma","Pa","dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa re Ga Ma Pa dha Ni Sȧ","avaroha":"Sȧ Ni dha Pa Ma Ga re Sa","pakkad":[{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"Ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Shant","skippedAaroh":"","skippedAvaroh":""},"bhairavdevata":{"name":"Bhairav-Devata","order":360,"thaat":"Bhairav","allowed":["Sa","re","Ga","Ma","Pa","dha","Ni"],"vadi":"Sa","samvadi":"Pa","aroha":"Sa re Ga Ma Pa dha Ni Sȧ","avaroha":"Sȧ Ni dha Pa Ma Ga re Sa","pakkad":[{"note":"Ni","oct":-1},{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0},{"note":"Ni","oct":-1},{"note":"dha","oct":-1},{"note":"Ni","oct":-1},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Bhakti","skippedAaroh":"","skippedAvaroh":""},"bhairavjaun":{"name":"Bhairav-Jaun","order":370,"thaat":"Bhairav","allowed":["Sa","re","Ga","Ma","Pa","dha","Ni"],"vadi":"Pa","samvadi":"re","aroha":"Sa re Ga Ma Pa dha Ni Sȧ","avaroha":"Sȧ Ni dha Pa Ma Ga re Sa","pakkad":[{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Shant","skippedAaroh":"","skippedAvaroh":""},"bhairavkabiri":{"name":"Bhairav-Kabiri","order":380,"thaat":"Bhairav","allowed":["Sa","re","Ga","Ma","Pa","dha","Ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa re Ga Ma Pa dha Ni Sȧ","avaroha":"Sȧ Ni dha Pa Ma Ga re Sa","pakkad":[{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Bhakti","skippedAaroh":"","skippedAvaroh":""},"bhairavkaushi":{"name":"Bhairav-Kaushi","order":390,"thaat":"Bhairav","allowed":["Sa","re","Ga","Ma","Pa","dha","Ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa re Ga Ma Pa dha Ni Sȧ","avaroha":"Sȧ Ni dha Pa Ma Ga re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Ni","oct":0},{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Shant","skippedAaroh":"","skippedAvaroh":""},"bhairavmangal":{"name":"Bhairav-Mangal","order":400,"thaat":"Bhairav","allowed":["Sa","re","Ga","Ma","Pa","dha","Ni"],"vadi":"Sa","samvadi":"Ma","aroha":"Sa re Ga Ma Pa dha Ni Sȧ","avaroha":"Sȧ Ni dha Pa Ma Ga re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0},{"note":"dha","oct":-1},{"note":"Ni","oct":-1},{"note":"Sa","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Bhakti","skippedAaroh":"","skippedAvaroh":""},"bhairavnat":{"name":"Bhairav-Nat","order":410,"thaat":"Bhairav","allowed":["Sa","re","Ga","Ma","Pa","dha","Ni"],"vadi":"Ga","samvadi":"dha","aroha":"Sa Ga re Ga Ma Pa dha Ni Sȧ","avaroha":"Sȧ Ni dha Pa Ma Ga re Ga Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Ga","oct":0},{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Shant","skippedAaroh":"","skippedAvaroh":""},"bhairavprabhat":{"name":"Bhairav-Prabhat","order":420,"thaat":"Bhairav","allowed":["Sa","re","Ga","Ma","Pa","dha","Ni"],"vadi":"dha","samvadi":"Ga","aroha":"Sa re Ga Ma Pa dha Ni Sȧ","avaroha":"Sȧ Ni dha Pa Ma Ga re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"Ni","oct":0},{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Bhakti","skippedAaroh":"","skippedAvaroh":""},"bhairavsaurashtra":{"name":"Bhairav-Saurashtra","order":430,"thaat":"Bhairav","allowed":["Sa","re","Ga","Ma","Pa","dha","Ni"],"vadi":"Pa","samvadi":"Sa","aroha":"Sa re Ga Ma Pa dha Ni Sȧ","avaroha":"Sȧ Ni dha Pa Ma Ga re Sa","pakkad":[{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"dha","oct":0},{"note":"Pa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Shant","skippedAaroh":"","skippedAvaroh":""},"bhairavgauri":{"name":"Bhairav-Gauri","thaat":"Bhairav","order":430,"allowed":["Sa","re","Ga","Ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa re Ga Ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Shant","skippedAaroh":"","skippedAvaroh":""},"bhairavshivmat":{"name":"Bhairav-Shivmat","order":440,"thaat":"Bhairav","allowed":["Sa","re","Ga","Ma","Pa","dha","Ni"],"vadi":"re","samvadi":"dha","aroha":"Sa re Ga Ma Pa dha Ni Sȧ","avaroha":"Sȧ Ni dha Pa Ma Ga re Sa","pakkad":[{"note":"dha","oct":-1},{"note":"Ni","oct":-1},{"note":"re","oct":0},{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Bhakti","skippedAaroh":"","skippedAvaroh":""},"bhairavanand":{"name":"Bhairavi-Anand","thaat":"Bhairavi","order":440,"allowed":["Sa","re","ga","Ma","Pa","dha","ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa ga Ma Pa dha ni Sȧ","avaroha":"Sȧ ni dha Pa Ma ga re Sa","pakkad":[{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"ni","oct":0},{"note":"dha","oct":0},{"note":"Pa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Karuna","skippedAaroh":"re","skippedAvaroh":""},"bhairavvariants":{"name":"Bhairav-Variants","order":450,"thaat":"Bhairav","allowed":["Sa","re","Ga","Ma","Pa","dha","Ni"],"vadi":"dha","samvadi":"re","aroha":"Sa re Ga Ma Pa dha Ni Sȧ","avaroha":"Sȧ Ni dha Pa Ma Ga re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0},{"note":"dha","oct":-1},{"note":"Ni","oct":-1},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Bhakti","skippedAaroh":"","skippedAvaroh":""},"bhairavsindhhu":{"name":"Bhairavi-Sindhu","thaat":"Bhairavi","order":450,"allowed":["Sa","re","ga","Ma","Pa","dha","ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa re ga Ma Pa dha ni Sȧ","avaroha":"Sȧ ni dha Pa Ma ga re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Karuna","skippedAaroh":"","skippedAvaroh":""},"bhairavvikram":{"name":"Bhairav-Vikram","order":460,"thaat":"Bhairav","allowed":["Sa","re","Ga","Ma","Pa","dha","Ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa re Ga Ma Pa dha Ni Sȧ","avaroha":"Sȧ Ni dha Pa Ma Ga re Sa","pakkad":[{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"Ni","oct":0},{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Veera","skippedAaroh":"","skippedAvaroh":""},"bhankar":{"name":"Bhankar","thaat":"Kalyan","order":460,"allowed":["Sa","Re","Ga","ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa Re Ga ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa ma Ga Re Sa","pakkad":[{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Evening","rasa":"Shant","skippedAaroh":"","skippedAvaroh":""},"bhairavvirat":{"name":"Bhairav-Virat","order":470,"thaat":"Bhairav","allowed":["Sa","re","Ga","Ma","Pa","dha","Ni"],"vadi":"Sa","samvadi":"Pa","aroha":"Sa re Ga Ma Pa dha Ni Sȧ","avaroha":"Sȧ Ni dha Pa Ma Ga re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Ni","oct":-1},{"note":"dha","oct":-1},{"note":"Ni","oct":-1},{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Veera","skippedAaroh":"","skippedAvaroh":""},"bhatiyali":{"name":"Bhatiyali","thaat":"Kafi","order":470,"allowed":["Sa","Re","ga","Ma","Pa","Dha","ni"],"vadi":"Pa","samvadi":"Sa","aroha":"Sa Re ga Ma Pa Dha ni Sȧ","avaroha":"Sȧ ni Dha Pa Ma ga Re Sa","pakkad":[{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Any","rasa":"Shringar","skippedAaroh":"","skippedAvaroh":""},"bhatiyar":{"name":"Bhatiyar","thaat":"Marwa","order":480,"allowed":["Sa","re","Ga","ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa re Ga ma Ga Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa ma Ga re Sa","pakkad":[{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Ga","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Early Morning","rasa":"Shant","skippedAaroh":"","skippedAvaroh":""},"bhatiyarkhamaji":{"name":"Bhatiyar-Khamaji","thaat":"Khamaj","order":490,"allowed":["Sa","re","Ga","Ma","ma","Pa","Dha","Ni","ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa re Ga ma Pa Dha Ni Sȧ","avaroha":"Sȧ ni Dha Pa Ma Ga re Sa","pakkad":[{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"ni","oct":1},{"note":"Dha","oct":0},{"note":"Pa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Early Morning","rasa":"Shant","skippedAaroh":"","skippedAvaroh":""},"bhavsakh":{"name":"Bhavsakh","thaat":"Bhairav","order":500,"allowed":["Sa","re","Ga","Ma","Pa","dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa re Ga Ma Pa dha Ni Sȧ","avaroha":"Sȧ Ni dha Pa Ma Ga re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Shant","skippedAaroh":"","skippedAvaroh":""},"bhawani":{"name":"Bhawani","thaat":"Bhairav","order":510,"allowed":["Sa","re","Ga","Ma","Pa","Dha","Ni"],"vadi":"Sa","samvadi":"Pa","aroha":"Sa re Ga Ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"Ga","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Bhakti","skippedAaroh":"","skippedAvaroh":""},"bhimpalas":{"name":"Bhimpalas","thaat":"Kafi","order":520,"allowed":["Sa","Re","ga","Ma","Pa","dha","ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa Re ga Ma Pa ni Sȧ","avaroha":"Sȧ ni dha Pa Ma ga Re Sa","pakkad":[{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"ni","oct":0},{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"ODAV-SAMPUNRA","time":"Afternoon","rasa":"Shringar","skippedAaroh":"dha","skippedAvaroh":""},"bhoopnat":{"name":"Bhoop-Nat","thaat":"Kalyan","order":530,"allowed":["Sa","Re","Ga","ma","Pa","Dha"],"vadi":"Ga","samvadi":"Dha","aroha":"Sa Re Ga Pa Dha Sȧ","avaroha":"Sȧ Dha Pa Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Sa","oct":1},{"note":"Re","oct":1},{"note":"Sa","oct":1},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"ODAV-ODAV","time":"Evening","rasa":"Shant","skippedAaroh":"Ma Ni","skippedAvaroh":"Ma Ni"},"bhoopeshwaripratiksha":{"name":"Bhoopeshwari-Pratiksha","thaat":"Kalyan","order":540,"allowed":["Sa","Re","Ga","ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa Re Ga ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SHADAV","time":"Evening","rasa":"Shant","skippedAaroh":"","skippedAvaroh":"Ma"},"bihaghem":{"name":"Bihag-Hem","thaat":"Bilawal","order":550,"allowed":["Sa","Re","Ga","Ma","ma","Pa","Dha","Ni","ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa Ga Ma Pa Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga Ma Re Sa","pakkad":[{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0}],"jaati":"ODAV-SAMPUNRA","time":"Night","rasa":"Shant","skippedAaroh":"Re Dha","skippedAvaroh":""},"bihagmalati":{"name":"Bihag-Malati","thaat":"Bilawal","order":560,"allowed":["Sa","Re","Ga","Ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa Ga Ma Pa Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"ODAV-SAMPUNRA","time":"Night","rasa":"Shringar","skippedAaroh":"Re Dha","skippedAvaroh":""},"bihagmanjiri":{"name":"Bihag-Manjiri","thaat":"Bilawal","order":570,"allowed":["Sa","Re","Ga","Ma","ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa Re Ga Ma ma Pa Ni Sȧ","avaroha":"Sȧ Ni Dha Pa ma Ma Ga Re Sa","pakkad":[{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"ma","oct":0},{"note":"Ga","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Night","rasa":"Shringar","skippedAaroh":"","skippedAvaroh":""},"bihagmaru":{"name":"Bihag-Maru","thaat":"Bilawal","order":580,"allowed":["Sa","Re","Ga","Ma","ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa Ga ma Pa Ni Sȧ","avaroha":"Sȧ Ni Dha Pa ma Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"ma","oct":0},{"note":"Ga","oct":0}],"jaati":"ODAV-SAMPUNRA","time":"Night","rasa":"Veer","skippedAaroh":"Re Ma Dha","skippedAvaroh":""},"bihagnat":{"name":"Bihag-Nat","thaat":"Bilawal","order":590,"allowed":["Sa","Re","Ga","Ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa Re Ga Ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Night","rasa":"Shringar","skippedAaroh":"","skippedAvaroh":""},"bihagpat":{"name":"Bihag-Pat","thaat":"Bilawal","order":600,"allowed":["Sa","Re","Ga","Ma","ma","Pa","Dha","Ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa Ga Ma ma Pa Ni Sȧ","avaroha":"Sȧ Ni Dha Pa ma Ma Ga Re Sa","pakkad":[{"note":"Ma","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"ma","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0}],"jaati":"ODAV-SAMPUNRA","time":"Night","rasa":"Shant","skippedAaroh":"Re Dha","skippedAvaroh":""},"bihagada":{"name":"Bihagada","thaat":"Bilawal","order":610,"allowed":["Sa","Re","Ga","Ma","ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa Re Ga Ma ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa ma Ma Ga Re Sa","pakkad":[{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"ma","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Night","rasa":"Shringar","skippedAaroh":"","skippedAvaroh":""},"bihari":{"name":"Bihari","thaat":"Kafi","order":620,"allowed":["Sa","Re","ga","Ma","Pa","Dha","ni"],"vadi":"Pa","samvadi":"Sa","aroha":"Sa Re ga Ma Pa Dha ni Sȧ","avaroha":"Sȧ ni Dha Pa Ma ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Afternoon","rasa":"Shringar","skippedAaroh":"","skippedAvaroh":""},"bilawalachampak":{"name":"Bilawal-Champak","thaat":"Bilawal","order":630,"allowed":["Sa","Re","Ga","Ma","Pa","Dha","Ni"],"vadi":"Dha","samvadi":"Ga","aroha":"Sa Re Ga Ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Shant","skippedAaroh":"","skippedAvaroh":""},"bilawalgaud":{"name":"Bilawal-Gaud","thaat":"Bilawal","order":640,"allowed":["Sa","Re","Ga","Ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa Re Ga Ma Pa Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SHADAV-SAMPUNRA","time":"Morning","rasa":"Shant","skippedAaroh":"Dha","skippedAvaroh":""},"bilawalhammer":{"name":"Bilawal-Hameer","thaat":"Bilawal","order":650,"allowed":["Sa","Re","Ga","Ma","ma","Pa","Dha","Ni"],"vadi":"Dha","samvadi":"Ga","aroha":"Sa Re Ga ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma ma Ga Re Sa","pakkad":[{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Late Night","rasa":"Shringar","skippedAaroh":"","skippedAvaroh":""},"bilawaljayant":{"name":"Bilawal-Jayant","thaat":"Bilawal","order":660,"allowed":["Sa","Re","Ga","Ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa Re Ga Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SHADAV-SAMPUNRA","time":"Morning","rasa":"Shant","skippedAaroh":"Ma","skippedAvaroh":""},"bilawalkukubh":{"name":"Bilawal-Kukubh","thaat":"Bilawal","order":670,"allowed":["Sa","Re","Ga","Ma","Pa","Dha","Ni"],"vadi":"Pa","samvadi":"Sa","aroha":"Sa Re Ga Ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Shant","skippedAaroh":"","skippedAvaroh":""},"bilawalnarayani":{"name":"Bilawal-Narayani","thaat":"Bilawal","order":680,"allowed":["Sa","Re","Ga","Ma","Pa","Dha","Ni"],"vadi":"Re","samvadi":"Pa","aroha":"Sa Re Ga Ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Bhakti","skippedAaroh":"","skippedAvaroh":""},"bilawalnat":{"name":"Bilawal-Nat","thaat":"Bilawal","order":690,"allowed":["Sa","Re","Ga","Ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa Re Ga Ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Shant","skippedAaroh":"","skippedAvaroh":""},"bilawalssarparda":{"name":"Bilawal-Sarparda","thaat":"Bilawal","order":700,"allowed":["Sa","Re","Ga","Ma","Pa","Dha","Ni"],"vadi":"Sa","samvadi":"Pa","aroha":"Sa Re Ga Ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Sa","oct":0},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Shant","skippedAaroh":"","skippedAvaroh":""},"bilawalshukla":{"name":"Bilawal-Shukla","thaat":"Bilawal","order":710,"allowed":["Sa","Re","Ga","Ma","Pa","Dha","Ni"],"vadi":"Dha","samvadi":"Ga","aroha":"Sa Re Ga Ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Re","oct":1},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Shant","skippedAaroh":"","skippedAvaroh":""},"bilawalsukhiya":{"name":"Bilawal-Sukhiya","thaat":"Bilawal","order":720,"allowed":["Sa","Re","Ga","Ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa Re Ga Ma Pa Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SHADAV-SAMPUNRA","time":"Morning","rasa":"Shant","skippedAaroh":"Dha","skippedAvaroh":""},"bilawayamani":{"name":"Bilawal-Yamani","thaat":"Bilawal","order":730,"allowed":["Sa","Re","Ga","Ma","ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa Re Ga ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Shant","skippedAaroh":"","skippedAvaroh":""},"champak":{"name":"Champak","thaat":"Bilawal","order":740,"allowed":["Sa","Re","Ga","Ma","Pa","Dha","Ni"],"vadi":"Dha","samvadi":"Ga","aroha":"Sa Re Ga Ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Late Morning","rasa":"Shringar","skippedAaroh":"","skippedAvaroh":""},"chandradhwani":{"name":"Chandradhwani","thaat":"Kafi","order":750,"allowed":["Sa","Re","ga","Ma","Pa","Dha","ni"],"vadi":"Sa","samvadi":"Pa","aroha":"Sa Re ga Ma Pa Dha ni Sȧ","avaroha":"Sȧ ni Dha Pa Ma ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Sa","oct":0},{"note":"ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Night","rasa":"Shant","skippedAaroh":"","skippedAvaroh":""},"chandrakauns":{"name":"Chandrakauns","thaat":"Bhairavi","order":760,"allowed":["Sa","ga","Ma","dha","ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa ga Ma dha ni Sȧ","avaroha":"Sȧ ni dha Ma ga Sa","pakkad":[{"note":"Sa","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"ni","oct":0},{"note":"dha","oct":0},{"note":"Ma","oct":0},{"note":"ga","oct":0},{"note":"Sa","oct":0}],"jaati":"ODAV-ODAV","time":"Late Night","rasa":"Shant","skippedAaroh":"","skippedAvaroh":""},"chandramadhu":{"name":"Chandramadhu","thaat":"Kafi","order":770,"allowed":["Sa","Re","ga","Ma","Pa","Dha","ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa ga Ma Pa Dha ni Sȧ","avaroha":"Sȧ ni Dha Pa Ma ga Re Sa","pakkad":[{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"ga","oct":0},{"note":"Sa","oct":0}],"jaati":"SHADAV-SAMPUNRA","time":"Night","rasa":"Shant","skippedAaroh":"Re","skippedAvaroh":""},"chandraprabha":{"name":"Chandraprabha","thaat":"Bhairav","order":780,"allowed":["Sa","re","Ga","Ma","Pa","Dha","Ni"],"vadi":"Pa","samvadi":"Sa","aroha":"Sa re Ga Ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Shant","skippedAaroh":"","skippedAvaroh":""},"charukeshi":{"name":"Charukeshi","thaat":"Bilawal","order":790,"allowed":["Sa","Re","Ga","Ma","Pa","dha","ni"],"vadi":"Pa","samvadi":"Sa","aroha":"Sa Re Ga Ma Pa dha ni Sȧ","avaroha":"Sȧ ni dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"ni","oct":0},{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Any","rasa":"Karuna","skippedAaroh":"","skippedAvaroh":""},"chhayanat":{"name":"Chhayanat","thaat":"Kalyan","order":800,"allowed":["Sa","Re","Ga","Ma","ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa Re Ga ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Night","rasa":"Shringar","skippedAaroh":"","skippedAvaroh":""},"chiraag":{"name":"Chiraag","thaat":"Bilawal","order":810,"allowed":["Sa","Re","Ga","Ma","Pa","Dha","Ni"],"vadi":"Dha","samvadi":"Ga","aroha":"Sa Re Ga Ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Evening","rasa":"Shringar","skippedAaroh":"","skippedAvaroh":""},"deepak":{"name":"Deepak","order":820,"thaat":"Kalyan","allowed":["Sa","Re","Ga","ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa Re Ga ma Pa Dha Ni Ṡ","avaroha":"Ṡ Ni Dha Pa ma Ga Re Sa","pakkad":[{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1}],"jaati":"SAMPURNA-SAMPURNA","time":"Evening","rasa":"Shringara, Karuna","skippedAaroh":"","skippedAvaroh":""},"deepakkedar":{"name":"Deepak-Kedar","order":830,"thaat":"Kalyan","allowed":["Sa","Re","Ga","Ma","ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa Re Ga ma Ma Pa Dha Ni Ṡ","avaroha":"Ṡ Ni Dha Pa Ma ma Ga Re Sa","pakkad":[{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Ni","oct":0}],"jaati":"SAMPURNA-SAMPURNA","time":"Evening","rasa":"Shringara","skippedAaroh":"","skippedAvaroh":""},"desh":{"name":"Desh","order":840,"thaat":"Khamaj","allowed":["Sa","Re","Ga","Ma","Pa","Dha","ni","Ni"],"vadi":"Re","samvadi":"Pa","aroha":"Sa Re Ma Pa ni Ṡ","avaroha":"Ṡ Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1}],"jaati":"ODAV-SAMPURNA","time":"Night","rasa":"Shringara, Bhakti","skippedAaroh":"Ga Dha","skippedAvaroh":""},"desi":{"name":"Desi","order":850,"thaat":"Kafi","allowed":["Sa","Re","ga","Ma","Pa","Dha","ni"],"vadi":"Pa","samvadi":"Sa","aroha":"Sa Re ga Ma Pa Dha ni Ṡ","avaroha":"Ṡ ni Dha Pa Ma ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0}],"jaati":"SAMPURNA-SAMPURNA","time":"Evening","rasa":"Karuna, Shringara","skippedAaroh":"","skippedAvaroh":""},"deskar":{"name":"Deskar","order":860,"thaat":"Bilawal","allowed":["Sa","Re","Ga","Pa","Dha"],"vadi":"Dha","samvadi":"Ga","aroha":"Sa Re Ga Pa Dha Ṡ","avaroha":"Ṡ Dha Pa Ga Re Sa","pakkad":[{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"ODAV-ODAV","time":"Morning","rasa":"Shanti, Bhakti","skippedAaroh":"Ma Ni","skippedAvaroh":"Ma Ni"},"devgandhar":{"name":"Dev Gandhar","order":870,"thaat":"Bhairav","allowed":["Sa","re","Ga","Ma","Pa","dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa re Ga Ma Pa dha Ni Ṡ","avaroha":"Ṡ Ni dha Pa Ma Ga re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0}],"jaati":"SAMPURNA-SAMPURNA","time":"Morning","rasa":"Karuna, Bhakti","skippedAaroh":"","skippedAvaroh":""},"devgandhari":{"name":"DevGandhari","order":880,"thaat":"Khamaj","allowed":["Sa","Re","Ga","Ma","Pa","Dha","ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa Re Ga Ma Pa Dha ni Ṡ","avaroha":"Ṡ ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0}],"jaati":"SAMPURNA-SAMPURNA","time":"Evening","rasa":"Shringara, Bhakti","skippedAaroh":"","skippedAvaroh":""},"devkauns":{"name":"Devkauns","order":890,"thaat":"Bhairavi","allowed":["Sa","ga","Ma","dha","ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa ga Ma dha ni Ṡ","avaroha":"Ṡ ni dha Ma ga Sa","pakkad":[{"note":"Sa","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"dha","oct":0},{"note":"ni","oct":0}],"jaati":"ODAV-ODAV","time":"Night","rasa":"Karuna, Shanta","skippedAaroh":"Re Pa","skippedAvaroh":"Re Pa"},"devranjani":{"name":"Devranjani","order":900,"thaat":"Kafi","allowed":["Sa","Re","ga","Ma","Pa","Dha"],"vadi":"Re","samvadi":"Pa","aroha":"Sa Re ga Ma Pa Dha Ṡ","avaroha":"Ṡ Dha Pa Ma ga Re Sa","pakkad":[{"note":"Re","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0}],"jaati":"SHADAV-SHADAV","time":"Evening","rasa":"Shringara, Karuna","skippedAaroh":"Ni","skippedAvaroh":"Ni"},"devsakh":{"name":"Devsakh","order":910,"thaat":"Khamaj","allowed":["Sa","Re","Ga","Ma","Pa","Dha","ni"],"vadi":"Pa","samvadi":"Re","aroha":"Sa Re Ga Ma Pa Dha ni Ṡ","avaroha":"Ṡ ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"ni","oct":0}],"jaati":"SAMPURNA-SAMPURNA","time":"Evening","rasa":"Shringara","skippedAaroh":"","skippedAvaroh":""},"dhanashree":{"name":"Dhanashree","order":920,"thaat":"Kafi","allowed":["Sa","Re","ga","Ma","Pa","dha","ni"],"vadi":"Pa","samvadi":"Sa","aroha":"Sa Re ga Ma Pa dha ni Ṡ","avaroha":"Ṡ ni dha Pa Ma ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0}],"jaati":"SAMPURNA-SAMPURNA","time":"Afternoon","rasa":"Karuna, Bhakti","skippedAaroh":"","skippedAvaroh":""},"dhani":{"name":"Dhani","order":930,"thaat":"Kafi","allowed":["Sa","ga","Ma","Pa","ni"],"vadi":"Pa","samvadi":"Sa","aroha":"Sa ga Ma Pa ni Ṡ","avaroha":"Ṡ ni Pa Ma ga Sa","pakkad":[{"note":"Sa","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"ni","oct":0}],"jaati":"ODAV-ODAV","time":"Night","rasa":"Karuna, Shringara","skippedAaroh":"Re Dha","skippedAvaroh":"Re Dha"},"dhavalshree":{"name":"Dhaval Shree","order":940,"thaat":"Poorvi","allowed":["Sa","re","Ga","Ma","ma","Pa","dha","Ni"],"vadi":"re","samvadi":"Pa","aroha":"Sa re Ga ma Pa dha Ni Ṡ","avaroha":"Ṡ Ni dha Pa Ma ma Ga re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0}],"jaati":"SAMPURNA-SAMPURNA","time":"Evening","rasa":"Karuna, Bhakti","skippedAaroh":"","skippedAvaroh":""},"durga":{"name":"Durga","order":950,"thaat":"Bilawal","allowed":["Sa","Re","Ma","Pa","Dha"],"vadi":"Re","samvadi":"Pa","aroha":"Sa Re Ma Pa Dha Ṡ","avaroha":"Ṡ Dha Pa Ma Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Sa","oct":1}],"jaati":"ODAV-ODAV","time":"Evening","rasa":"Shringara, Bhakti","skippedAaroh":"Ga Ni","skippedAvaroh":"Ga Ni"},"gaara":{"name":"Gaara","order":960,"thaat":"Khamaj","allowed":["Sa","Re","Ga","Ma","Pa","Dha","ni","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa Re Ga Ma Pa Dha ni Ṡ","avaroha":"Ṡ Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0}],"jaati":"SAMPURNA-SAMPURNA","time":"Night","rasa":"Shringara, Karuna","skippedAaroh":"","skippedAvaroh":""},"gandhari":{"name":"Gandhari","order":970,"thaat":"Bhairav","allowed":["Sa","re","Ga","Ma","Pa","dha","Ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa re Ga Ma Pa dha Ni Ṡ","avaroha":"Ṡ Ni dha Pa Ma Ga re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0}],"jaati":"SAMPURNA-SAMPURNA","time":"Morning","rasa":"Karuna, Bhakti","skippedAaroh":"","skippedAvaroh":""},"gandharibahar":{"name":"Gandhari-Bahar","order":980,"thaat":"Bhairav","allowed":["Sa","re","Ga","Ma","Pa","dha","Ni","ga"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa re Ga Ma Pa dha Ni Ṡ","avaroha":"Ṡ Ni dha Pa Ma ga re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0}],"jaati":"SAMPURNA-SAMPURNA","time":"Morning","rasa":"Karuna, Shringara","skippedAaroh":"","skippedAvaroh":""},"gangeshwari":{"name":"Gangeshwari","order":990,"thaat":"Bhairav","allowed":["Sa","re","Ga","Ma","Pa","dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa re Ga Ma Pa dha Ni Ṡ","avaroha":"Ṡ Ni dha Pa Ma Ga re Sa","pakkad":[{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0}],"jaati":"SAMPURNA-SAMPURNA","time":"Morning","rasa":"Bhakti, Shanta","skippedAaroh":"","skippedAvaroh":""},"garudhwani":{"name":"Garud Dhwani","order":1000,"thaat":"Bhairav","allowed":["Sa","re","Ma","Pa","Ni"],"vadi":"Pa","samvadi":"Sa","aroha":"Sa re Ma Pa Ni Ṡ","avaroha":"Ṡ Ni Pa Ma re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Ni","oct":0}],"jaati":"ODAV-ODAV","time":"Morning","rasa":"Bhakti, Shanta","skippedAaroh":"Ga Dha","skippedAvaroh":"Ga Dha"},"gauri":{"name":"Gauri","order":1010,"thaat":"Bhairav","allowed":["Sa","re","Ga","Ma","Pa","dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa re Ga Ma Pa dha Ni Ṡ","avaroha":"Ṡ Ni dha Pa Ma Ga re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPURNA-SAMPURNA","time":"Morning","rasa":"Bhakti, Karuna","skippedAaroh":"","skippedAvaroh":""},"gavati":{"name":"Gavati","order":1020,"thaat":"Khamaj","allowed":["Sa","Re","Ga","Ma","Pa","Dha","ni"],"vadi":"Sa","samvadi":"Pa","aroha":"Sa Re Ga Ma Pa Dha ni Ṡ","avaroha":"Ṡ ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"ni","oct":0}],"jaati":"SAMPURNA-SAMPURNA","time":"Evening","rasa":"Shringara","skippedAaroh":"","skippedAvaroh":""},"gunakali":{"name":"Gunakali","order":1030,"thaat":"Bhairav","allowed":["Sa","re","Ma","Pa","dha"],"vadi":"re","samvadi":"Pa","aroha":"Sa re Ma Pa dha Ṡ","avaroha":"Ṡ dha Pa Ma re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0}],"jaati":"ODAV-ODAV","time":"Morning","rasa":"Karuna, Shanta","skippedAaroh":"Ga Ni","skippedAvaroh":"Ga Ni"},"hameer":{"name":"Hameer","order":1040,"thaat":"Kalyan","allowed":["Sa","Re","Ga","Ma","ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa Re Ga Ma Pa Dha Ni Ṡ","avaroha":"Ṡ Ni Dha Pa ma Ga Re Sa","pakkad":[{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"ma","oct":0},{"note":"Ga","oct":0}],"jaati":"SAMPURNA-SAMPURNA","time":"Night","rasa":"Shringara, Veera","skippedAaroh":"ma","skippedAvaroh":"Ma"},"hansadhwani":{"name":"Hansadhwani","order":1050,"thaat":"Bilawal","allowed":["Sa","Re","Ga","Pa","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa Re Ga Pa Ni Ṡ","avaroha":"Ṡ Ni Pa Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Pa","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1}],"jaati":"ODAV-ODAV","time":"Evening","rasa":"Shringara, Ananda","skippedAaroh":"Ma Dha","skippedAvaroh":"Ma Dha"},"hanskinkini":{"name":"Hanskinkini","order":1060,"thaat":"Khamaj","allowed":["Sa","Re","Ga","Ma","Pa","Dha","ni"],"vadi":"Ga","samvadi":"Dha","aroha":"Sa Re Ga Ma Pa Dha ni Ṡ","avaroha":"Ṡ ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0}],"jaati":"SAMPURNA-SAMPURNA","time":"Evening","rasa":"Shringara","skippedAaroh":"","skippedAvaroh":""},"hansnarayani":{"name":"HansNarayani","order":1070,"thaat":"Bilawal","allowed":["Sa","Re","Ga","Ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa Re Ga Pa Dha Ni Ṡ","avaroha":"Ṡ Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Pa","oct":0},{"note":"Ni","oct":0}],"jaati":"SHADAV-SAMPURNA","time":"Evening","rasa":"Shringara, Bhakti","skippedAaroh":"Ma","skippedAvaroh":""},"harikauns":{"name":"Harikauns","order":1080,"thaat":"Bhairavi","allowed":["Sa","ga","Ma","dha","ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa ga Ma dha ni Ṡ","avaroha":"Ṡ ni dha Ma ga Sa","pakkad":[{"note":"Sa","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1}],"jaati":"ODAV-ODAV","time":"Night","rasa":"Karuna, Bhakti","skippedAaroh":"Re Pa","skippedAvaroh":"Re Pa"},"hemant":{"name":"Hemant","order":1090,"thaat":"Khamaj","allowed":["Sa","Re","Ga","Ma","Pa","ni"],"vadi":"Sa","samvadi":"Pa","aroha":"Sa Re Ga Ma Pa ni Ṡ","avaroha":"Ṡ ni Pa Ma Ga Re Sa","pakkad":[{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"ni","oct":0},{"note":"Pa","oct":0}],"jaati":"SHADAV-SHADAV","time":"Evening","rasa":"Shringara, Shanta","skippedAaroh":"Dha","skippedAvaroh":"Dha"},"hemawati":{"name":"Hemawati","order":1100,"thaat":"Poorvi","allowed":["Sa","re","Ga","ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa re Ga ma Pa Dha Ni Ṡ","avaroha":"Ṡ Ni Dha Pa ma Ga re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0}],"jaati":"SAMPURNA-SAMPURNA","time":"Evening","rasa":"Shringara, Bhakti","skippedAaroh":"","skippedAvaroh":""},"hindol":{"name":"Hindol","order":1110,"thaat":"Kalyan","allowed":["Sa","Ga","ma","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa Ga ma Dha Ni Ṡ","avaroha":"Ṡ Ni Dha ma Ga Sa","pakkad":[{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Dha","oct":0}],"jaati":"ODAV-ODAV","time":"Morning","rasa":"Shringara, Ananda","skippedAaroh":"Re Pa","skippedAvaroh":"Re Pa"},"hindolbahar":{"name":"Hindol-Bahar","order":1120,"thaat":"Kalyan","allowed":["Sa","Re","Ga","ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa Ga ma Dha Ni Ṡ","avaroha":"Ṡ Ni Dha Pa ma Ga Re Sa","pakkad":[{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1}],"jaati":"ODAV-SAMPURNA","time":"Morning","rasa":"Shringara","skippedAaroh":"Re Pa","skippedAvaroh":""},"hindolpancham":{"name":"Hindol-Pancham","order":1130,"thaat":"Kalyan","allowed":["Sa","Ga","ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa Ga ma Pa Dha Ni Ṡ","avaroha":"Ṡ Ni Dha Pa ma Ga Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0}],"jaati":"SHADAV-SHADAV","time":"Morning","rasa":"Shringara, Ananda","skippedAaroh":"Re","skippedAvaroh":"Re"},"hindolita":{"name":"Hindolita","order":1140,"thaat":"Kalyan","allowed":["Sa","Re","Ga","ma","Dha","Ni"],"vadi":"Ga","samvadi":"Dha","aroha":"Sa Re Ga ma Dha Ni Ṡ","avaroha":"Ṡ Ni Dha ma Ga Re Sa","pakkad":[{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0}],"jaati":"SHADAV-SHADAV","time":"Morning","rasa":"Shringara","skippedAaroh":"Pa","skippedAvaroh":"Pa"},"jait":{"name":"Jait","order":1150,"thaat":"Marwa","allowed":["Sa","re","Ga","ma","Dha","Ni"],"vadi":"re","samvadi":"Dha","aroha":"Ni re Ga ma Dha Ni Ṡ","avaroha":"Ṡ Ni Dha ma Ga re Sa","pakkad":[{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0}],"jaati":"SHADAV-SHADAV","time":"Evening","rasa":"Karuna, Shringara","skippedAaroh":"Sa Pa","skippedAvaroh":"Pa"},"jaitashri":{"name":"Jaitashri","order":1160,"thaat":"Poorvi","allowed":["Sa","re","Ga","Ma","ma","Pa","dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa re Ga ma Pa dha Ni Ṡ","avaroha":"Ṡ Ni dha Pa Ma ma Ga re Sa","pakkad":[{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"Ni","oct":0}],"jaati":"SAMPURNA-SAMPURNA","time":"Evening","rasa":"Karuna, Bhakti","skippedAaroh":"","skippedAvaroh":""},"janasammohini":{"name":"Janasammohini","order":1170,"thaat":"Bilawal","allowed":["Sa","Re","Ga","Ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa Re Ga Ma Pa Dha Ni Ṡ","avaroha":"Ṡ Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0}],"jaati":"SAMPURNA-SAMPURNA","time":"Evening","rasa":"Shringara, Ananda","skippedAaroh":"","skippedAvaroh":""},"jangala":{"name":"Jangala","order":1180,"thaat":"Bhairav","allowed":["Sa","re","Ga","Ma","Pa","dha"],"vadi":"Pa","samvadi":"Sa","aroha":"Sa re Ga Ma Pa dha Ṡ","avaroha":"Ṡ dha Pa Ma Ga re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0}],"jaati":"SHADAV-SHADAV","time":"Morning","rasa":"Bhakti, Shanta","skippedAaroh":"Ni","skippedAvaroh":"Ni"},"jaunpuri":{"name":"Jaunpuri","order":1190,"thaat":"Asavari","allowed":["Sa","Re","ga","Ma","Pa","dha","ni"],"vadi":"dha","samvadi":"ga","aroha":"Sa Re ga Ma Pa dha ni Ṡ","avaroha":"Ṡ ni dha Pa Ma ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0}],"jaati":"SAMPURNA-SAMPURNA","time":"Morning","rasa":"Karuna, Shanta","skippedAaroh":"","skippedAvaroh":""},"jayjayvanti":{"name":"Jayjayvanti","order":1200,"thaat":"Khamaj","allowed":["Sa","Re","ga","Ga","Ma","Pa","Dha","ni","Ni"],"vadi":"Re","samvadi":"Pa","aroha":"Sa Re ga Ma Pa ni Ṡ","avaroha":"Ṡ Ni Dha Pa Ma Ga Re ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1}],"jaati":"SHADAV-SAMPURNA","time":"Night","rasa":"Shringara, Karuna","skippedAaroh":"Ga Dha Ni","skippedAvaroh":""},"jhinjhoti":{"name":"Jhinjhoti","order":1210,"thaat":"Khamaj","allowed":["Sa","Re","Ga","Ma","Pa","Dha","ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa Re Ga Ma Pa Dha ni Ṡ","avaroha":"Ṡ ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"ni","oct":0}],"jaati":"SAMPURNA-SAMPURNA","time":"Night","rasa":"Shringara, Ananda","skippedAaroh":"","skippedAvaroh":""},"jog":{"name":"Jog","order":1220,"thaat":"Kafi","allowed":["Sa","ga","Ga","Ma","Pa","ni"],"vadi":"Pa","samvadi":"Sa","aroha":"Sa ga Ma Pa ni Ṡ","avaroha":"Ṡ ni Pa Ma Ga Sa","pakkad":[{"note":"Sa","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1}],"jaati":"ODAV-ODAV","time":"Night","rasa":"Karuna, Shringara","skippedAaroh":"Re Dha","skippedAvaroh":"Re Dha"},"kalashree":{"name":"Kalashree","thaat":"Kafi","order":1220,"allowed":["Sa","Re","ga","Ma","Pa","Dha","ni"],"vadi":"Pa","samvadi":"Sa","aroha":"Sa Re ga Ma Pa Dha ni Sȧ","avaroha":"Sȧ ni Dha Pa Ma ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Evening","rasa":"Shringar","skippedAaroh":"","skippedAvaroh":""},"jogiya":{"name":"Jogiya","order":1230,"thaat":"Bhairav","allowed":["Sa","re","Ma","Pa","dha","Ni"],"vadi":"re","samvadi":"Pa","aroha":"Sa re Ma Pa Ni Ṡ","avaroha":"Ṡ Ni dha Pa Ma re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Ni","oct":0}],"jaati":"ODAV-SHADAV","time":"Morning","rasa":"Karuna, Vairagya","skippedAaroh":"Ga dha","skippedAvaroh":"Ga"},"kalavati":{"name":"Kalavati","order":1230,"thaat":"Bilawal","allowed":["Sa","Re","Ma","Pa","Ni"],"vadi":"Pa","samvadi":"Re","aroha":"Sa Re Ma Pa Ni Sȧ","avaroha":"Sȧ Ni Pa Ma Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Ni","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"ODAV-ODAV","time":"Night","rasa":"Shant"},"jogkauns":{"name":"Jogkauns","order":1240,"thaat":"Bhairavi","allowed":["Sa","ga","Ma","Pa","dha","ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa ga Ma Pa dha ni Ṡ","avaroha":"Ṡ ni dha Pa Ma ga Sa","pakkad":[{"note":"Sa","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"ni","oct":0}],"jaati":"SHADAV-SHADAV","time":"Night","rasa":"Karuna, Shanta","skippedAaroh":"Re","skippedAvaroh":"Re"},"kalingada":{"name":"Kalingada","thaat":"Bhairav","order":1240,"allowed":["Sa","re","Ga","Ma","Pa","dha","Ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa re Ga Ma Pa dha Ni Sȧ","avaroha":"Sȧ Ni dha Pa Ma Ga re Sa","pakkad":[{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Early Morning","rasa":"Shanta","skippedAaroh":"","skippedAvaroh":""},"kafi":{"name":"Kafi","order":1250,"thaat":"Kafi","allowed":["Sa","Re","ga","Ma","Pa","Dha","ni"],"vadi":"Pa","samvadi":"Sa","aroha":"Sa Re ga Ma Pa Dha ni Ṡ","avaroha":"Ṡ ni Dha Pa Ma ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0}],"jaati":"SAMPURNA-SAMPURNA","time":"Night","rasa":"Shringara, Karuna","skippedAaroh":"","skippedAvaroh":""},"kalyan":{"name":"Kalyan","order":1250,"thaat":"Kalyan","allowed":["Sa","Re","Ga","ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Nị Re Ga ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa ma Ga Re Sa","pakkad":[{"note":"Ni","oct":-1},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Evening","rasa":"Shringar"},"kafimalhar":{"name":"Kafi-Malhar","order":1260,"thaat":"Kafi","allowed":["Sa","Re","ga","Ma","Pa","Dha","dha","ni"],"vadi":"Pa","samvadi":"Sa","aroha":"Sa Re ga Ma Pa dha ni Ṡ","avaroha":"Ṡ ni Dha Pa Ma ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"ni","oct":0}],"jaati":"SAMPURNA-SAMPURNA","time":"Night","rasa":"Karuna, Shringara","skippedAaroh":"","skippedAvaroh":""},"kalyanchandrakant":{"name":"Kalyan-Chandrakant","order":1260,"thaat":"Kalyan","allowed":["Sa","Re","Ga","Ma","ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa Re Ga ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Evening","rasa":"Shringar"},"kalyandhankoni":{"name":"Kalyan-Dhankoni","order":1270,"thaat":"Kalyan","allowed":["Sa","Re","Ga","ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Dha","aroha":"Sa Re Ga ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Pa","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Evening","rasa":"Shringar"},"kalyangorakh":{"name":"Kalyan-Gorakh","order":1280,"thaat":"Kalyan","allowed":["Sa","Re","Ga","ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa Re Ga ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa ma Ga Re Sa","pakkad":[{"note":"Ni","oct":-1},{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"ma","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Evening","rasa":"Shringar"},"kalyangun":{"name":"Kalyan-Gun","order":1290,"thaat":"Kalyan","allowed":["Sa","Re","Ga","ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa Re Ga ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Evening","rasa":"Shringar"},"kalyanhem":{"name":"Kalyan-Hem","order":1300,"thaat":"Kalyan","allowed":["Sa","Re","Ga","Ma","ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa Re Ga ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Evening","rasa":"Shringar"},"kalyanjait":{"name":"Kalyan-Jait","order":1310,"thaat":"Kalyan","allowed":["Sa","Re","Ga","ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Nị Re Ga ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa ma Ga Re Sa","pakkad":[{"note":"Ni","oct":-1},{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Evening","rasa":"Shringar"},"kalyankhem":{"name":"Kalyan-Khem","order":1320,"thaat":"Kalyan","allowed":["Sa","Re","Ga","Ma","ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa Re Ga ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Evening","rasa":"Shringar"},"kalyanmaluha":{"name":"Kalyan-Maluha","order":1330,"thaat":"Kalyan","allowed":["Sa","Re","Ga","Ma","ma","Pa","Dha","Ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa Re Ga ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Evening","rasa":"Shringar"},"kalyanraj":{"name":"Kalyan-Raj","order":1340,"thaat":"Kalyan","allowed":["Sa","Re","Ga","ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa Re Ga ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Evening","rasa":"Shringar"},"kalyansavani":{"name":"Kalyan-Savani","order":1350,"thaat":"Kalyan","allowed":["Sa","Re","Ga","Ma","ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa Re Ga ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Any","rasa":"Shringar"},"kalyansham":{"name":"Kalyan-Sham","order":1360,"thaat":"Kalyan","allowed":["Sa","Re","Ga","Ma","ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa Re Ga ma Pa Ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Pa","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Evening","rasa":"Karuna"},"kalyanshuddh":{"name":"Kalyan-Shuddh","order":1370,"thaat":"Kalyan","allowed":["Sa","Re","Ga","Ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa Re Ga Ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Evening","rasa":"Shringar"},"kamalashri":{"name":"Kamalashri","thaat":"Bhairav","order":1380,"allowed":["Sa","re","Ga","Ma","Pa","Dha","Ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa re Ga Ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga re Sa","pakkad":[{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Early Morning","rasa":"Shanta","skippedAaroh":"","skippedAvaroh":""},"kamod":{"name":"Kamod","order":1390,"thaat":"Kalyan","allowed":["Sa","Re","Ga","Ma","ma","Pa","Dha","Ni"],"vadi":"Pa","samvadi":"Re","aroha":"Sa Re Ga Pa ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga Re Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Pa","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"ODAV-SAMPUNRA","time":"Evening","rasa":"Shringar"},"kanada":{"name":"Kanada","order":1400,"thaat":"Asavari","allowed":["Sa","Re","ga","Ma","Pa","dha","ni"],"vadi":"Pa","samvadi":"Re","aroha":"Sa Re ga Ma Pa dha ni Sȧ","avaroha":"Sȧ ni dha Pa Ma ga Re ga Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0},{"note":"ni","oct":-1},{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Night","rasa":"Shringar"},"kanadabageshri":{"name":"Kanada-Bageshri","order":1410,"thaat":"Kafi","allowed":["Sa","Re","ga","Ma","Pa","dha","ni"],"vadi":"ga","samvadi":"ni","aroha":"Sa ga Ma dha ni Sȧ","avaroha":"Sȧ ni dha Ma ga Re ga Sa","pakkad":[{"note":"Sa","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"dha","oct":0},{"note":"ni","oct":0},{"note":"dha","oct":0},{"note":"Ma","oct":0},{"note":"ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"ODAV-SAMPUNRA","time":"Night","rasa":"Shringar"},"kanadadarbari":{"name":"Kanada-Darbari","order":1420,"thaat":"Asavari","allowed":["Sa","Re","ga","Ma","Pa","dha","ni"],"vadi":"Re","samvadi":"Pa","aroha":"Nị Sa Re ga Ma Pa dha ni Sȧ","avaroha":"Sȧ ni dha Pa Ma ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0},{"note":"ni","oct":-1},{"note":"dha","oct":-1},{"note":"ni","oct":-1},{"note":"Sa","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"Pa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Night","rasa":"Karuna"},"kanadagunji":{"name":"Kanada-Gunji","order":1430,"thaat":"Asavari","allowed":["Sa","Re","ga","Ma","Pa","dha","ni"],"vadi":"Pa","samvadi":"Re","aroha":"Sa Re ga Ma Pa dha ni Sȧ","avaroha":"Sȧ ni dha Pa Ma ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"ni","oct":0},{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Night","rasa":"Shringar"},"kanadakafi":{"name":"Kanada-Kafi","order":1440,"thaat":"Kafi","allowed":["Sa","Re","ga","Ma","Pa","dha","ni"],"vadi":"ga","samvadi":"ni","aroha":"Sa Re ga Ma Pa dha ni Sȧ","avaroha":"Sȧ ni dha Pa Ma ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0},{"note":"ni","oct":-1},{"note":"Sa","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Night","rasa":"Shringar"},"kanadakaushi":{"name":"Kanada-Kaushi","order":1450,"thaat":"Asavari","allowed":["Sa","ga","Ma","Pa","ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa ga Ma Pa ni Sȧ","avaroha":"Sȧ ni Pa Ma ga Sa","pakkad":[{"note":"Sa","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"ni","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"ga","oct":0},{"note":"Sa","oct":0}],"jaati":"ODAV-ODAV","time":"Night","rasa":"Karuna"},"kanadanayaki":{"name":"Kanada-Nayaki","order":1460,"thaat":"Asavari","allowed":["Sa","Re","ga","Ma","Pa","dha","ni"],"vadi":"ga","samvadi":"dha","aroha":"Sa Re ga Ma Pa dha ni Sȧ","avaroha":"Sȧ ni dha Pa Ma ga Re ga Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"ni","oct":0},{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"ga","oct":0},{"note":"Re","oct":0},{"note":"ga","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Night","rasa":"Karuna"},"kanadaraisa":{"name":"Kanada-Raisa","order":1470,"thaat":"Asavari","allowed":["Sa","Re","ga","Ma","Pa","dha","ni"],"vadi":"Pa","samvadi":"Re","aroha":"Sa Re ga Ma Pa dha ni Sȧ","avaroha":"Sȧ ni dha Pa Ma ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"ni","oct":0},{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Night","rasa":"Shringar"},"kanadashahana":{"name":"Kanada-Shahana","order":1480,"thaat":"Kafi","allowed":["Sa","Re","ga","Ma","Pa","Dha","ni"],"vadi":"ga","samvadi":"Pa","aroha":"Sa Re ga Ma Pa Dha ni Sȧ","avaroha":"Sȧ ni Dha Pa Ma ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Night","rasa":"Shringar"},"kanadasuha":{"name":"Kanada-Suha","order":1490,"thaat":"Asavari","allowed":["Sa","Re","ga","Ma","Pa","dha","ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa Re ga Ma Pa dha ni Sȧ","avaroha":"Sȧ ni dha Pa Ma ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Night","rasa":"Karuna"},"karaharpriya":{"name":"Karaharpriya","order":1500,"thaat":"Kafi","allowed":["Sa","Re","ga","Ma","Pa","Dha","ni"],"vadi":"ga","samvadi":"ni","aroha":"Sa Re ga Ma Pa Dha ni Sȧ","avaroha":"Sȧ ni Dha Pa Ma ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Any","rasa":"Karuna"},"kaushikaunjani":{"name":"Kaushik Ranjani","order":1510,"thaat":"Asavari","allowed":["Sa","ga","Ma","Pa","ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa ga Ma Pa ni Sȧ","avaroha":"Sȧ ni Pa Ma ga Sa","pakkad":[{"note":"Sa","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"ni","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Sa","oct":0}],"jaati":"ODAV-ODAV","time":"Night","rasa":"Karuna"},"kaushiki":{"name":"Kaushiki","order":1520,"thaat":"Asavari","allowed":["Sa","ga","Ma","Pa","dha","ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa ga Ma Pa dha ni Sȧ","avaroha":"Sȧ ni dha Pa Ma ga Sa","pakkad":[{"note":"Sa","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"ni","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"ga","oct":0},{"note":"Sa","oct":0}],"jaati":"SHADAV-SHADAV","time":"Night","rasa":"Karuna"},"kedar":{"name":"Kedar","order":1530,"thaat":"Kalyan","allowed":["Sa","Re","Ga","Ma","ma","Pa","Dha","Ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa Ma Ma Pa Ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Ma","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"ODAV-SAMPUNRA","time":"Evening","rasa":"Shant"},"kedarbasanti":{"name":"Kedar-Basanti","order":1540,"thaat":"Kalyan","allowed":["Sa","Re","Ga","Ma","ma","Pa","Dha","Ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa Ma Pa ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"ODAV-SAMPUNRA","time":"Evening","rasa":"Shringar"},"kedarchandani":{"name":"Kedar-Chandani","order":1550,"thaat":"Kalyan","allowed":["Sa","Re","Ga","Ma","ma","Pa","Dha","Ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa Ma Pa ma Pa Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SHADAV-SAMPUNRA","time":"Evening","rasa":"Shant"},"kedarjaladhar":{"name":"Kedar-Jaladhar","order":1560,"thaat":"Kalyan","allowed":["Sa","Re","Ga","Ma","ma","Pa","Dha","Ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa Ma Pa ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"ODAV-SAMPUNRA","time":"Evening","rasa":"Shant"},"kedarmaluha":{"name":"Kedar-Maluha","order":1570,"thaat":"Kalyan","allowed":["Sa","Re","Ga","Ma","ma","Pa","Dha","Ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa Ma Pa ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Ma","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"ODAV-SAMPUNRA","time":"Evening","rasa":"Shant"},"kedarsuha":{"name":"Kedar-Suha","order":1580,"thaat":"Kalyan","allowed":["Sa","Re","Ga","Ma","ma","Pa","Dha","Ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa Ma Pa ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"ODAV-SAMPUNRA","time":"Evening","rasa":"Shant"},"khambavati":{"name":"Khambavati","order":1590,"thaat":"Khamaj","allowed":["Sa","Re","Ga","Ma","Pa","Dha","ni","Ni"],"vadi":"Ga","samvadi":"ni","aroha":"Sa Re Ga Ma Pa Dha ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Evening","rasa":"Shringar"},"khat":{"name":"Khat","order":1600,"thaat":"Bhairav","allowed":["Sa","Ma","Pa","Ni"],"vadi":"Sa","samvadi":"Pa","aroha":"Sa Ma Pa Ni Sȧ","avaroha":"Sȧ Ni Pa Ma Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Sa","oct":0}],"jaati":"ODAV-ODAV","time":"Morning","rasa":"Shant"},"khokar":{"name":"Khokar","order":1610,"thaat":"Bilawal","allowed":["Sa","Re","Ma","Pa","Dha"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa Re Ma Pa Dha Sȧ","avaroha":"Sȧ Dha Pa Ma Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"ODAV-ODAV","time":"Any","rasa":"Shant"},"kirwani":{"name":"Kirwani","order":1620,"thaat":"Asavari","allowed":["Sa","Re","ga","Ma","Pa","dha","Ni"],"vadi":"ga","samvadi":"dha","aroha":"Sa Re ga Ma Pa dha Ni Sȧ","avaroha":"Sȧ Ni dha Pa Ma ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Night","rasa":"Karuna"},"lacchasakh":{"name":"Lacchasakh","thaat":"Bilawal","order":1630,"allowed":["Sa","Re","Ga","Ma","Pa","Dha","Ni"],"vadi":"Re","samvadi":"Pa","aroha":"Sa Re Ga Ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Shringar","skippedAaroh":"","skippedAvaroh":""},"lagangandhar":{"name":"Lagan-Gandhar","thaat":"Kafi","order":1640,"allowed":["Sa","Re","ga","Ma","Pa","Dha","ni"],"vadi":"ga","samvadi":"Dha","aroha":"Sa Re ga Ma Pa Dha ni Sȧ","avaroha":"Sȧ ni Dha Pa Ma ga Re Sa","pakkad":[{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Evening","rasa":"Shringar","skippedAaroh":"","skippedAvaroh":""},"lalat":{"name":"Lalat","order":1650,"thaat":"Bhairav","allowed":["Sa","re","Ga","Ma","ma","Pa","dha","Ni"],"vadi":"ma","samvadi":"Sa","aroha":"Sa re Ga Ma ma Pa dha Ni Sȧ","avaroha":"Sȧ Ni dha Pa ma Ma Ga re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"ma","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Early Morning","rasa":"Shant"},"lalatbhatiyar":{"name":"Lalat Bhatiyar","order":1660,"thaat":"Bhairav","allowed":["Sa","re","Ga","Ma","ma","Pa","dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa re Ga Ma ma Pa dha Ni Sȧ","avaroha":"Sȧ Ni dha Pa ma Ma Ga re Sa","pakkad":[{"note":"Ni","oct":-1},{"note":"re","oct":0},{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Early Morning","rasa":"Shant"},"lalatpancham":{"name":"Lalat-Pancham","order":1670,"thaat":"Bhairav","allowed":["Sa","re","Ga","Ma","ma","Pa","dha","Ni"],"vadi":"ma","samvadi":"Sa","aroha":"Sa re Ga Ma ma Pa dha Ni Sȧ","avaroha":"Sȧ Ni dha Pa ma Ma Ga re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"Ni","oct":0},{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"ma","oct":0},{"note":"Ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Early Morning","rasa":"Shant"},"lalitadhwani":{"name":"Lalita Dhwani","thaat":"Marwa","order":1680,"allowed":["Sa","re","Ga","ma","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa re Ga ma Dha Ni Sȧ","avaroha":"Sȧ Ni Dha ma Ga re Sa","pakkad":[{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"ma","oct":0},{"note":"Ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0}],"jaati":"SHADAV-SHADAV","time":"Afternoon","rasa":"Shanta","skippedAaroh":"Pa","skippedAvaroh":"Pa"},"lalitagauri":{"name":"Lalita-Gauri","thaat":"Bhairav","order":1690,"allowed":["Sa","re","Ga","Ma","Pa","dha","Ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa re Ga Ma Pa dha Ni Sȧ","avaroha":"Sȧ Ni dha Pa Ma Ga re Sa","pakkad":[{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Early Morning","rasa":"Shanta","skippedAaroh":"","skippedAvaroh":""},"latangi":{"name":"Latangi","thaat":"Khamaj","order":1700,"allowed":["Sa","Re","Ga","Ma","Pa","dha","Ni"],"vadi":"Pa","samvadi":"Re","aroha":"Sa Re Ga Ma Pa dha Ni Sȧ","avaroha":"Sȧ Ni dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Afternoon","rasa":"Shringar","skippedAaroh":"","skippedAvaroh":""},"maand":{"name":"Maand","thaat":"Bilawal","order":1710,"allowed":["Sa","Re","Ga","Ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa Ga Ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SHADAV-SAMPUNRA","time":"Any","rasa":"Shringar","skippedAaroh":"Re","skippedAvaroh":""},"madhukauns":{"name":"Madhukauns","thaat":"Bhairavi","order":1720,"allowed":["Sa","ga","Ma","Pa","ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa ga Ma Pa ni Sȧ","avaroha":"Sȧ ni Pa Ma ga Sa","pakkad":[{"note":"Sa","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"ni","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"ga","oct":0},{"note":"Sa","oct":0}],"jaati":"ODAV-ODAV","time":"Late Night","rasa":"Veer","skippedAaroh":"Re Dha","skippedAvaroh":"Re Dha"},"madhuvanti":{"name":"Madhuvanti","order":1730,"thaat":"Todi","allowed":["Sa","Re","ga","ma","Pa","Dha","Ni"],"vadi":"ma","samvadi":"Sa","aroha":"Sa Re ga ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa ma ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"ga","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"ma","oct":0},{"note":"ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Afternoon","rasa":"Shringar"},"malav":{"name":"Malav","thaat":"Kafi","order":1740,"allowed":["Sa","Re","ga","Ma","Pa","Dha","ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa Re ga Ma Pa Dha ni Sȧ","avaroha":"Sȧ ni Dha Pa Ma ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Evening","rasa":"Shringar","skippedAaroh":"","skippedAvaroh":""},"malavi":{"name":"Malavi","thaat":"Bhairavi","order":1750,"allowed":["Sa","re","ga","Ma","Pa","dha","ni"],"vadi":"ga","samvadi":"dha","aroha":"Sa re ga Ma Pa dha ni Sȧ","avaroha":"Sȧ ni dha Pa Ma ga re Sa","pakkad":[{"note":"re","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"ni","oct":0},{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Karuna","skippedAaroh":"","skippedAvaroh":""},"malgunji":{"name":"Malgunji","thaat":"Kafi","order":1760,"allowed":["Sa","Re","ga","Ma","Pa","Dha","ni"],"vadi":"ga","samvadi":"Pa","aroha":"Sa Re ga Ma Pa Dha ni Sȧ","avaroha":"Sȧ ni Dha Pa Ma ga Re Sa","pakkad":[{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Night","rasa":"Shringar","skippedAaroh":"","skippedAvaroh":""},"malharcharjuki":{"name":"Malhar-Charju Ki","order":1770,"thaat":"Kafi","allowed":["Sa","Re","ga","Ma","Pa","dha","Dha","ni"],"vadi":"Pa","samvadi":"Sa","aroha":"Sa Re ga Ma Pa Dha ni Sȧ","avaroha":"Sȧ ni dha Pa Ma ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"ni","oct":0},{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Any","rasa":"Veer"},"malharchhaya":{"name":"Malhar-Chhaya","order":1780,"thaat":"Kafi","allowed":["Sa","Re","ga","Ma","Pa","dha","Dha","ni"],"vadi":"Pa","samvadi":"Ma","aroha":"Sa ga Ma Pa dha Sȧ","avaroha":"Sȧ ni Dha Pa Ma ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"Sa","oct":1},{"note":"ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SHADAV-SAMPUNRA","time":"Any","rasa":"Karuna"},"malhardhulia":{"name":"Malhar-Dhulia","order":1790,"thaat":"Kafi","allowed":["Sa","Re","ga","Ma","Pa","dha","ni"],"vadi":"Pa","samvadi":"Sa","aroha":"Sa Re ga Ma Pa dha ni Sȧ","avaroha":"Sȧ ni dha Pa Ma ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"ni","oct":0},{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Any","rasa":"Veer"},"malhargaud":{"name":"Malhar-Gaud","order":1800,"thaat":"Kafi","allowed":["Sa","Re","ga","Ma","Pa","Dha","ni"],"vadi":"Pa","samvadi":"Sa","aroha":"Sa Re ga Ma Pa Dha ni Sȧ","avaroha":"Sȧ ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Any","rasa":"Veer"},"malharjayant":{"name":"Malhar-Jayant","order":1810,"thaat":"Kafi","allowed":["Sa","Re","ga","Ga","Ma","Pa","dha","Dha","ni"],"vadi":"Pa","samvadi":"Sa","aroha":"Sa Re ga Ma Pa Dha ni Sȧ","avaroha":"Sȧ ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Any","rasa":"Veer"},"malharmeera":{"name":"Malhar-Meera","order":1820,"thaat":"Kafi","allowed":["Sa","Re","ga","Ma","Pa","dha","ni"],"vadi":"Pa","samvadi":"Sa","aroha":"Sa Re ga Ma Pa dha ni Sȧ","avaroha":"Sȧ ni dha Pa Ma ga Re Sa","pakkad":[{"note":"ni","oct":-1},{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Any","rasa":"Bhakti"},"malharmeegh":{"name":"Malhar-Megh","order":1830,"thaat":"Kafi","allowed":["Sa","Re","ga","Ma","Pa","ni"],"vadi":"Pa","samvadi":"Sa","aroha":"Sa Re Ma Pa ni Sȧ","avaroha":"Sȧ ni Pa Ma ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"ni","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SHADAV-SAMPUNRA","time":"Any","rasa":"Veer"},"malharmiya":{"name":"Malhar-Miya","order":1840,"thaat":"Kafi","allowed":["Sa","Re","ga","Ga","Ma","Pa","dha","ni"],"vadi":"Pa","samvadi":"Sa","aroha":"Sa Re ga Ma Pa dha ni Sȧ","avaroha":"Sȧ ni dha Pa Ma Ga Re ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0},{"note":"ni","oct":-1},{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Any","rasa":"Veer"},"malharramdasi":{"name":"Malhar-Ramdasi","order":1850,"thaat":"Kafi","allowed":["Sa","Re","ga","Ga","Ma","Pa","dha","Dha","ni"],"vadi":"Pa","samvadi":"Sa","aroha":"Sa Re Ga Ma Pa Dha ni Sȧ","avaroha":"Sȧ ni dha Pa Ma ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"ni","oct":0},{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Any","rasa":"Veer"},"malharsur":{"name":"Malhar-Sur","order":1860,"thaat":"Kafi","allowed":["Sa","Re","ga","Ma","Pa","dha","ni"],"vadi":"Pa","samvadi":"Sa","aroha":"Sa Re ga Ma Pa dha ni Sȧ","avaroha":"Sȧ ni dha Pa Ma ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"ni","oct":0},{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Any","rasa":"Shant"},"maligaura":{"name":"MaliGaura","thaat":"Bhairav","order":1870,"allowed":["Sa","re","Ga","Ma","Pa","Dha","Ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa re Ga Ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Early Morning","rasa":"Shanta","skippedAaroh":"","skippedAvaroh":""},"malkauspancham":{"name":"Malkauns-Pancham","order":1880,"thaat":"Bhairavi","allowed":["Sa","ga","Ma","Pa","dha","ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa ga Ma Pa dha ni Sȧ","avaroha":"Sȧ ni dha Pa Ma ga Sa","pakkad":[{"note":"Sa","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"ni","oct":0},{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"ga","oct":0},{"note":"Sa","oct":0}],"jaati":"SHADAV-SHADAV","time":"Night","rasa":"Karuna"},"malkaussampurna":{"name":"Malkauns-Sampurna","order":1890,"thaat":"Bhairavi","allowed":["Sa","re","ga","Ma","Pa","dha","ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa re ga Ma Pa dha ni Sȧ","avaroha":"Sȧ ni dha Pa Ma ga re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"ni","oct":0},{"note":"dha","oct":0},{"note":"Ma","oct":0},{"note":"ga","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Night","rasa":"Karuna"},"megh":{"name":"Megh","thaat":"Kafi","order":1900,"allowed":["Sa","Re","Ma","Pa","ni"],"vadi":"Re","samvadi":"Pa","aroha":"Sa Re Ma Pa ni Sȧ","avaroha":"Sȧ ni Pa Ma Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"ni","oct":0},{"note":"Pa","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"ODAV-ODAV","time":"Any","rasa":"Veer","skippedAaroh":"Ga Dha","skippedAvaroh":"Ga Dha"},"mehekali":{"name":"Mehekali","thaat":"Bhairav","order":1910,"allowed":["Sa","re","Ga","Ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa re Ga Ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga re Sa","pakkad":[{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Early Morning","rasa":"Shanta","skippedAaroh":"","skippedAvaroh":""},"multani":{"name":"Multani","order":1920,"thaat":"Todi","allowed":["Sa","re","ga","ma","Pa","Dha","Ni"],"vadi":"Pa","samvadi":"Sa","aroha":"Sa ga ma Pa Ni Sȧ","avaroha":"Sȧ Ni Dha Pa ma ga re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"ga","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"ma","oct":0},{"note":"ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0}],"jaati":"ODAV-SAMPUNRA","time":"Afternoon","rasa":"Karuna"},"nand":{"name":"Nand","order":1930,"thaat":"Kalyan","allowed":["Sa","Re","Ga","Ma","ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa Re Ga ma Pa Ma Pa Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Evening","rasa":"Shringar"},"nandkauns":{"name":"Nandkauns","order":1940,"thaat":"Kalyan","allowed":["Sa","Re","Ga","Ma","ma","Pa","dha","Dha","ni","Ni"],"vadi":"Ga","samvadi":"ni","aroha":"Sa Re Ga ma Pa Ma Pa Ni Sȧ","avaroha":"Sȧ ni dha Pa Ma ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"ni","oct":0},{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Night","rasa":"Shringar"},"narayani":{"name":"Narayani","order":1950,"thaat":"Bilawal","allowed":["Sa","Re","Ga","Ma","Pa","Dha","Ni"],"vadi":"Pa","samvadi":"Sa","aroha":"Sa Re Ga Ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Bhakti"},"nat":{"name":"Nat","order":1960,"thaat":"Bilawal","allowed":["Sa","Re","Ga","Ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Dha","aroha":"Sa Re Ga Ma Pa Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SHADAV-SAMPUNRA","time":"Any","rasa":"Veer"},"natbilawal":{"name":"Nat-Bilawal","order":1970,"thaat":"Bilawal","allowed":["Sa","Re","Ga","Ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Dha","aroha":"Sa Re Ga Ma Pa Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SHADAV-SAMPUNRA","time":"Morning","rasa":"Veer"},"nathem":{"name":"Nat-Hem","order":1980,"thaat":"Kalyan","allowed":["Sa","Re","Ga","Ma","ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa Re Ga ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Evening","rasa":"Shringar"},"natkamod":{"name":"Nat-Kamod","order":1990,"thaat":"Kalyan","allowed":["Sa","Re","Ga","Ma","ma","Pa","Dha","Ni"],"vadi":"Pa","samvadi":"Re","aroha":"Sa Re Ga Pa ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Pa","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"ODAV-SAMPUNRA","time":"Evening","rasa":"Shringar"},"natmalhar":{"name":"Nat-Malhar","order":2000,"thaat":"Kafi","allowed":["Sa","Re","ga","Ga","Ma","Pa","dha","Dha","ni","Ni"],"vadi":"Pa","samvadi":"Sa","aroha":"Sa Re Ga Ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni dha Pa Ma ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Any","rasa":"Veer"},"natshuddha":{"name":"Nat-Shuddha","order":2010,"thaat":"Bilawal","allowed":["Sa","Re","Ga","Ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Dha","aroha":"Sa Re Ga Ma Pa Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SHADAV-SAMPUNRA","time":"Any","rasa":"Veer"},"neelambari":{"name":"Neelambari","order":2020,"thaat":"Kafi","allowed":["Sa","Re","ga","Ma","Pa","dha","Ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa Re ga Ma Pa dha Ni Sȧ","avaroha":"Sȧ Ni dha Pa Ma ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Night","rasa":"Karuna"},"noorani":{"name":"Noorani","thaat":"Khamaj","order":2030,"allowed":["Sa","Re","Ga","Ma","Pa","Dha","ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa Re Ga Ma Pa Dha ni Sȧ","avaroha":"Sȧ ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Evening","rasa":"Shringar","skippedAaroh":"","skippedAvaroh":""},"pahadi":{"name":"Pahadi","order":2040,"thaat":"Bilawal","allowed":["Sa","Re","Ga","Ma","Pa","Dha","ni","Ni"],"vadi":"Pa","samvadi":"Sa","aroha":"Sa Re Ga Pa Dha Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"ODAV-SAMPUNRA","time":"Any","rasa":"Shringar"},"pancham":{"name":"Pancham","order":2050,"thaat":"Bilawal","allowed":["Sa","Re","Ga","Ma","Pa","Dha","Ni"],"vadi":"Pa","samvadi":"Sa","aroha":"Sa Re Ga Ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Pa","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Any","rasa":"Shant"},"paraj":{"name":"Paraj","order":2060,"thaat":"Poorvi","allowed":["Sa","re","Ga","Ma","ma","Pa","dha","Ni"],"vadi":"dha","samvadi":"re","aroha":"Sa re Ga Ma ma Pa dha Ni Sȧ","avaroha":"Sȧ Ni dha Pa ma Ma Ga re Sa","pakkad":[{"note":"Ni","oct":-1},{"note":"re","oct":0},{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"Ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Night","rasa":"Karuna"},"parameshwari":{"name":"Parameshwari","order":2070,"thaat":"Bhairav","allowed":["Sa","re","Ga","Ma","Pa","dha","Ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa re Ga Ma Pa dha Ni Sȧ","avaroha":"Sȧ Ni dha Pa Ma Ga re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Bhakti"},"patdeep":{"name":"Patdeep","order":2080,"thaat":"Kafi","allowed":["Sa","Re","ga","Ma","Pa","Dha","ni"],"vadi":"Pa","samvadi":"Re","aroha":"Sa Re ga Ma Pa Dha ni Sȧ","avaroha":"Sȧ ni Dha Pa Ma ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Evening","rasa":"Karuna"},"patdeepaki":{"name":"Patdeepaki","order":2090,"thaat":"Kafi","allowed":["Sa","Re","ga","Ma","Pa","Dha","ni"],"vadi":"Pa","samvadi":"Re","aroha":"Sa Re ga Ma Pa Dha ni Sȧ","avaroha":"Sȧ ni Dha Pa Ma ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Evening","rasa":"Karuna"},"patmanjiri":{"name":"Patmanjiri","order":2100,"thaat":"Bhairav","allowed":["Sa","re","Ga","Ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa re Ga Ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga re Sa","pakkad":[{"note":"Ni","oct":-1},{"note":"re","oct":0},{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Bhakti"},"pilu":{"name":"Pilu","order":2110,"thaat":"Kafi","allowed":["Sa","Re","ga","Ga","Ma","Pa","dha","Dha","ni","Ni"],"vadi":"ga","samvadi":"dha","aroha":"Sa Re ga Ma Pa dha ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga Re ga Sa","pakkad":[{"note":"Sa","oct":0},{"note":"ga","oct":0},{"note":"Re","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Any","rasa":"Shringar"},"poorba":{"name":"Poorba","thaat":"Purvi","order":2120,"allowed":["Sa","re","Ga","ma","Pa","dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa re Ga ma Pa dha Ni Sȧ","avaroha":"Sȧ Ni dha Pa ma Ga re Sa","pakkad":[{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"ma","oct":0},{"note":"Ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Evening","rasa":"Shanta","skippedAaroh":"","skippedAvaroh":""},"pooriya":{"name":"Pooriya","order":2130,"thaat":"Marwa","allowed":["Sa","re","Ga","Ma","ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Nị re Ga ma Ga ma Dha Ni Sȧ","avaroha":"Sȧ Ni Dha ma Ga re Sa","pakkad":[{"note":"Ni","oct":-1},{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Ga","oct":0},{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"ma","oct":0},{"note":"Ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0}],"jaati":"SHADAV-SHADAV","time":"Evening","rasa":"Shringar"},"pooriaydhanashri":{"name":"Pooriya Dhanashri","order":2140,"thaat":"Poorvi","allowed":["Sa","re","Ga","Ma","ma","Pa","dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Nị re Ga ma Pa dha Ni Sȧ","avaroha":"Sȧ Ni dha Pa ma Ga re Sa","pakkad":[{"note":"Ni","oct":-1},{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"ma","oct":0},{"note":"Ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Evening","rasa":"Karuna"},"pooriyakalyan":{"name":"Pooriya Kalyan","order":2150,"thaat":"Kalyan","allowed":["Sa","re","Ga","ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Nị re Ga ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa ma Ga re Sa","pakkad":[{"note":"Ni","oct":-1},{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Ga","oct":0},{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"ma","oct":0},{"note":"Ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Evening","rasa":"Shringar"},"poorvi":{"name":"Poorvi","order":2160,"thaat":"Poorvi","allowed":["Sa","re","Ga","Ma","ma","Pa","dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa re Ga Ma ma Pa dha Ni Sȧ","avaroha":"Sȧ Ni dha Pa ma Ma Ga re Sa","pakkad":[{"note":"Ni","oct":-1},{"note":"re","oct":0},{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"ma","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Evening","rasa":"Karuna"},"pratapvarali":{"name":"Pratapvarali","thaat":"Todi","order":2170,"allowed":["Sa","re","ga","ma","Pa","dha","Ni"],"vadi":"ma","samvadi":"Sa","aroha":"Sa re ga ma Pa dha Ni Sȧ","avaroha":"Sȧ Ni dha Pa ma ga re Sa","pakkad":[{"note":"re","oct":0},{"note":"ga","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"ma","oct":0},{"note":"ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Karuna","skippedAaroh":"","skippedAvaroh":""},"priyadarshini":{"name":"Priyadarshini","thaat":"Kalyan","order":2180,"allowed":["Sa","Re","Ga","ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa Re Ga ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa ma Ga Re Sa","pakkad":[{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"ma","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Evening","rasa":"Shringar","skippedAaroh":"","skippedAvaroh":""},"rageshri":{"name":"Rageshri","thaat":"Khamaj","order":2200,"allowed":["Sa","Re","Ga","Ma","Pa","Dha","ni"],"vadi":"Ga","samvadi":"ni","aroha":"Sa Re Ga Ma Dha ni Sȧ","avaroha":"Sȧ ni Dha Ma Ga Re Sa","pakkad":[{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1}],"jaati":"SHADAV-SHADAV","time":"Night","rasa":"Shringara","skippedAaroh":"Pa","skippedAvaroh":"Pa"},"ramgauri":{"name":"RamGauri","thaat":"Bhairav","order":2210,"allowed":["Sa","re","Ga","Ma","Pa","dha","Ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa re Ga Ma Pa dha Ni Sȧ","avaroha":"Sȧ Ni dha Pa Ma Ga re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"Ni","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Shanta","skippedAaroh":"","skippedAvaroh":""},"ramkali":{"name":"Ramkali","thaat":"Bhairav","order":2220,"allowed":["Sa","re","Ga","ma","Pa","dha","Ni"],"vadi":"dha","samvadi":"re","aroha":"Sa re Ga ma Pa dha Ni Sȧ","avaroha":"Sȧ Ni dha Pa ma Ga re Sa","pakkad":[{"note":"dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"re","oct":1},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"dha","oct":0},{"note":"Pa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Bhakti","skippedAaroh":"","skippedAvaroh":""},"roopkali":{"name":"Roopkali","thaat":"Bhairavi","order":2230,"allowed":["Sa","re","ga","Ma","Pa","dha","ni"],"vadi":"ga","samvadi":"dha","aroha":"Sa re ga Ma Pa dha ni Sȧ","avaroha":"Sȧ ni dha Pa Ma ga re Sa","pakkad":[{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"ni","oct":0},{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Karuna","skippedAaroh":"","skippedAvaroh":""},"sadar":{"name":"Sadar","thaat":"Kafi","order":2240,"allowed":["Sa","Re","ga","Ma","Pa","Dha","ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa Re ga Ma Pa Dha ni Sȧ","avaroha":"Sȧ ni Dha Pa Ma ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Evening","rasa":"Shringar","skippedAaroh":"","skippedAvaroh":""},"salagvarali":{"name":"Salagvarali","thaat":"Bhairavi","order":2250,"allowed":["Sa","re","ga","Ma","Pa","dha","ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa re ga Ma Pa dha ni Sȧ","avaroha":"Sȧ ni dha Pa Ma ga re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"ni","oct":0},{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Karuna","skippedAaroh":"","skippedAvaroh":""},"salang":{"name":"Salang","thaat":"Kafi","order":2260,"allowed":["Sa","Re","ga","Ma","Pa","Dha","ni"],"vadi":"Pa","samvadi":"Re","aroha":"Sa Re ga Ma Pa Dha ni Sȧ","avaroha":"Sȧ ni Dha Pa Ma ga Re Sa","pakkad":[{"note":"Re","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Evening","rasa":"Shringar","skippedAaroh":"","skippedAvaroh":""},"sarang":{"name":"Sarang","thaat":"Kafi","order":2270,"allowed":["Sa","Re","Ma","Pa","Dha","ni"],"vadi":"Re","samvadi":"Pa","aroha":"Sa Re Ma Pa ni Sȧ","avaroha":"Sȧ ni Dha Pa Ma Re Sa","pakkad":[{"note":"Re","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Re","oct":1},{"note":"Sa","oct":1}],"jaati":"ODAV-SHADAV","time":"Afternoon","rasa":"Shringara","skippedAaroh":"Ga Dha","skippedAvaroh":"Ga"},"sarangambika":{"name":"Sarang-Ambika","thaat":"Kafi","order":2280,"allowed":["Sa","Re","ga","Ma","Pa","Dha","ni"],"vadi":"Pa","samvadi":"Re","aroha":"Sa Re ga Ma Pa Dha ni Sȧ","avaroha":"Sȧ ni Dha Pa Ma ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"ni","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Afternoon","rasa":"Shringara","skippedAaroh":"","skippedAvaroh":""},"sarangbadhans":{"name":"Sarang-Badhans","thaat":"Kafi","order":2290,"allowed":["Sa","Re","Ma","Pa","Dha","ni"],"vadi":"Pa","samvadi":"Re","aroha":"Sa Re Ma Pa ni Sȧ","avaroha":"Sȧ ni Dha Pa Ma Re Sa","pakkad":[{"note":"Pa","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Re","oct":0}],"jaati":"ODAV-SHADAV","time":"Afternoon","rasa":"Shringara","skippedAaroh":"Ga Dha","skippedAvaroh":"Ga"},"sarangdhulia":{"name":"Sarang-Dhulia","thaat":"Kafi","order":2300,"allowed":["Sa","Re","ga","Ma","Pa","Dha","ni"],"vadi":"Re","samvadi":"Pa","aroha":"Sa Re Ma Pa ni Sȧ","avaroha":"Sȧ ni Dha Pa Ma ga Re Sa","pakkad":[{"note":"Re","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1}],"jaati":"ODAV-SAMPUNRA","time":"Afternoon","rasa":"Shringara","skippedAaroh":"ga Dha","skippedAvaroh":""},"saranggaud":{"name":"Sarang-Gaud","thaat":"Bilawal","order":2310,"allowed":["Sa","Re","Ga","Ma","Pa","Dha","Ni"],"vadi":"Re","samvadi":"Pa","aroha":"Sa Re Ga Ma Pa Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1}],"jaati":"SHADAV-SAMPUNRA","time":"Afternoon","rasa":"Shringara","skippedAaroh":"Dha","skippedAvaroh":""},"saranglankadahani":{"name":"Sarang-Lankadahani","thaat":"Kafi","order":2320,"allowed":["Sa","Re","Ma","Pa","ni"],"vadi":"Pa","samvadi":"Re","aroha":"Sa Re Ma Pa ni Sȧ","avaroha":"Sȧ ni Pa Ma Re Sa","pakkad":[{"note":"Re","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1}],"jaati":"ODAV-ODAV","time":"Afternoon","rasa":"Shringara","skippedAaroh":"Ga Dha","skippedAvaroh":"Ga Dha"},"sarangmadhymad":{"name":"Sarang-Madhymad","thaat":"Kafi","order":2330,"allowed":["Sa","Re","Ma","Pa","Dha","ni"],"vadi":"Re","samvadi":"Pa","aroha":"Sa Re Ma Pa Dha ni Sȧ","avaroha":"Sȧ ni Dha Pa Ma Re Sa","pakkad":[{"note":"Re","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1}],"jaati":"SHADAV-SHADAV","time":"Afternoon","rasa":"Shringara","skippedAaroh":"Ga","skippedAvaroh":"Ga"},"sarangmiyaka":{"name":"Sarang-Miya Ka","thaat":"Kafi","order":2340,"allowed":["Sa","Re","ga","Ma","Pa","Dha","ni"],"vadi":"Re","samvadi":"Pa","aroha":"Sa Re Ma Pa ni Sȧ","avaroha":"Sȧ ni Dha Pa Ma ga Re Sa","pakkad":[{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"ODAV-SAMPUNRA","time":"Afternoon","rasa":"Shringara","skippedAaroh":"ga Dha","skippedAvaroh":""},"sarangsamant":{"name":"Sarang-Samant","thaat":"Kafi","order":2350,"allowed":["Sa","Re","Ga","Ma","Pa","Dha","ni"],"vadi":"Pa","samvadi":"Re","aroha":"Sa Re Ga Ma Pa Dha ni Sȧ","avaroha":"Sȧ ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Afternoon","rasa":"Shringara","skippedAaroh":"","skippedAvaroh":""},"sarangsaraswati":{"name":"Sarang-Saraswati","thaat":"Kafi","order":2360,"allowed":["Sa","Re","Ma","Pa","Dha","ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa Re Ma Pa Dha ni Sȧ","avaroha":"Sȧ ni Dha Pa Ma Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1}],"jaati":"SHADAV-SHADAV","time":"Afternoon","rasa":"Bhakti","skippedAaroh":"Ga","skippedAvaroh":"Ga"},"sarangshuddh":{"name":"Sarang-Shuddh","thaat":"Kafi","order":2370,"allowed":["Sa","Re","Ma","Pa","Dha","ni"],"vadi":"Re","samvadi":"Pa","aroha":"Sa Re Ma Pa Dha ni Sȧ","avaroha":"Sȧ ni Dha Pa Ma Re Sa","pakkad":[{"note":"Re","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SHADAV-SHADAV","time":"Afternoon","rasa":"Shringara","skippedAaroh":"Ga","skippedAvaroh":"Ga"},"sarangvrundavani":{"name":"Sarang-Vrundavani","thaat":"Kafi","order":2380,"allowed":["Sa","Re","Ma","Pa","ni"],"vadi":"Re","samvadi":"Pa","aroha":"Sa Re Ma Pa ni Sȧ","avaroha":"Sȧ ni Pa Ma Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"ni","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Re","oct":0}],"jaati":"ODAV-ODAV","time":"Afternoon","rasa":"Shringara","skippedAaroh":"Ga Dha","skippedAvaroh":"Ga Dha"},"saraswati":{"name":"Saraswati","thaat":"Kafi","order":2390,"allowed":["Sa","Re","ga","Ma","Pa","Dha","ni"],"vadi":"Dha","samvadi":"Re","aroha":"Sa Re ga Ma Pa Dha ni Sȧ","avaroha":"Sȧ ni Dha Pa Ma ga Re Sa","pakkad":[{"note":"Dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"Re","oct":1},{"note":"ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Evening","rasa":"Bhakti","skippedAaroh":"","skippedAvaroh":""},"savninat":{"name":"Savni Nat","thaat":"Bilawal","order":2400,"allowed":["Sa","Re","Ga","Ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Dha","aroha":"Sa Re Ga Ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Re","oct":1},{"note":"Ga","oct":1}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Night","rasa":"Shringara","skippedAaroh":"","skippedAvaroh":""},"sazgiri":{"name":"Sazgiri","thaat":"Khamaj","order":2410,"allowed":["Sa","Re","Ga","Ma","Pa","Dha","ni"],"vadi":"Ga","samvadi":"Dha","aroha":"Sa Re Ga Ma Pa Dha ni Sȧ","avaroha":"Sȧ ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Evening","rasa":"Shringara","skippedAaroh":"","skippedAvaroh":""},"shahana":{"name":"Shahana","thaat":"Khamaj","order":2420,"allowed":["Sa","Re","Ga","Ma","Pa","Dha","ni","Ni"],"vadi":"Ga","samvadi":"ni","aroha":"Sa Re Ga Ma Pa Dha ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Night","rasa":"Shringara","skippedAaroh":"","skippedAvaroh":""},"shankara":{"name":"Shankara","thaat":"Bilawal","order":2430,"allowed":["Sa","Re","Ga","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa Re Ga Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ga Re Sa","pakkad":[{"note":"Ga","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ga","oct":0}],"jaati":"SHADAV-SHADAV","time":"Night","rasa":"Veer","skippedAaroh":"Ma","skippedAvaroh":"Ma"},"shivranjani":{"name":"Shivranjani","thaat":"Kafi","order":2440,"allowed":["Sa","Re","ga","Pa","Dha"],"vadi":"ga","samvadi":"Dha","aroha":"Sa Re ga Pa Dha Sȧ","avaroha":"Sȧ Dha Pa ga Re Sa","pakkad":[{"note":"ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"ga","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0}],"jaati":"ODAV-ODAV","time":"Any","rasa":"Shringara","skippedAaroh":"Ma Ni","skippedAvaroh":"Ma Ni"},"shree":{"name":"Shree","thaat":"Poorvi","order":2450,"allowed":["Sa","re","Ma","ma","Pa","dha","Ni"],"vadi":"re","samvadi":"Pa","aroha":"Sa re Ma ma Pa dha Ni Sȧ","avaroha":"Sȧ Ni dha Pa ma Ma re Sa","pakkad":[{"note":"re","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Evening","rasa":"Bhakti","skippedAaroh":"","skippedAvaroh":""},"sindhura":{"name":"Sindhura","thaat":"Khamaj","order":2460,"allowed":["Sa","Re","Ga","Ma","Pa","Dha","ni"],"vadi":"Pa","samvadi":"Re","aroha":"Sa Re Ga Ma Pa Dha ni Sȧ","avaroha":"Sȧ ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Evening","rasa":"Shringara","skippedAaroh":"","skippedAvaroh":""},"sohoni":{"name":"Sohoni","thaat":"Marwa","order":2470,"allowed":["Sa","re","Ga","ma","Dha","Ni"],"vadi":"Dha","samvadi":"Ga","aroha":"Sa re Ga ma Dha Ni Sȧ","avaroha":"Sȧ Ni Dha ma Ga re Sa","pakkad":[{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ga","oct":1},{"note":"re","oct":1},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Dha","oct":0}],"jaati":"SHADAV-SHADAV","time":"Evening","rasa":"Shringara","skippedAaroh":"Pa","skippedAvaroh":"Pa"},"sooryakauns":{"name":"Sooryakauns","thaat":"Bhairavi","order":2480,"allowed":["Sa","ga","Ma","dha","ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa ga Ma dha ni Sȧ","avaroha":"Sȧ ni dha Ma ga Sa","pakkad":[{"note":"Sa","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1}],"jaati":"ODAV-ODAV","time":"Evening","rasa":"Gambhir","skippedAaroh":"Re Pa","skippedAvaroh":"Re Pa"},"sorath":{"name":"Sorath","thaat":"Khamaj","order":2490,"allowed":["Sa","Re","Ga","Ma","Pa","Dha","ni"],"vadi":"Pa","samvadi":"Re","aroha":"Sa Re Ga Ma Pa Dha ni Sȧ","avaroha":"Sȧ ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"Re","oct":1},{"note":"Sa","oct":1},{"note":"ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Evening","rasa":"Shringara","skippedAaroh":"","skippedAvaroh":""},"sugandh":{"name":"Sugandh","thaat":"Bilawal","order":2500,"allowed":["Sa","Re","Ga","Ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa Re Ga Ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Shringar","skippedAaroh":"","skippedAvaroh":""},"sugharai":{"name":"Sugharai","thaat":"Bhairav","order":2510,"allowed":["Sa","re","Ga","Ma","Pa","dha","Ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa re Ga Ma Pa dha Ni Sȧ","avaroha":"Sȧ Ni dha Pa Ma Ga re Sa","pakkad":[{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Early Morning","rasa":"Shanta","skippedAaroh":"","skippedAvaroh":""},"suha":{"name":"Suha","thaat":"Kafi","order":2520,"allowed":["Sa","Re","ga","Ma","Pa","Dha","ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa Re ga Ma Pa Dha ni Sȧ","avaroha":"Sȧ ni Dha Pa Ma ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Afternoon","rasa":"Shringara","skippedAaroh":"","skippedAvaroh":""},"swanandi":{"name":"Swanandi","thaat":"Kalyan","order":2530,"allowed":["Sa","Re","Ga","ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa Re Ga ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa ma Ga Re Sa","pakkad":[{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"ma","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Evening","rasa":"Shringar","skippedAaroh":"","skippedAvaroh":""},"tilakkamod":{"name":"Tilak Kamod","thaat":"Khamaj","order":2540,"allowed":["Sa","Re","Ga","Ma","Pa","Dha","ni"],"vadi":"Re","samvadi":"Pa","aroha":"Sa Re Ga Pa Dha Sa","avaroha":"Sȧ ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Pa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Re","oct":0}],"jaati":"ODAV-SAMPUNRA","time":"Night","rasa":"Shringara","skippedAaroh":"Ma ni","skippedAvaroh":""},"tilakshyam":{"name":"Tilak Shyam","thaat":"Khamaj","order":2550,"allowed":["Sa","Re","Ga","Ma","Pa","Dha","ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa Re Ga Ma Pa Dha ni Sȧ","avaroha":"Sȧ ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Evening","rasa":"Shringar","skippedAaroh":"","skippedAvaroh":""},"tilang":{"name":"Tilang","thaat":"Khamaj","order":2560,"allowed":["Sa","Ga","Ma","Pa","ni"],"vadi":"Ga","samvadi":"ni","aroha":"Sa Ga Ma Pa ni Sȧ","avaroha":"Sȧ ni Pa Ma Ga Sa","pakkad":[{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"ni","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0}],"jaati":"ODAV-ODAV","time":"Night","rasa":"Shringara","skippedAaroh":"Re Dha","skippedAvaroh":"Re Dha"},"todiahiri":{"name":"Todi-Ahiri","thaat":"Bhairavi","order":2570,"allowed":["Sa","re","ga","Ma","Pa","dha","ni"],"vadi":"ga","samvadi":"ni","aroha":"Sa re ga Ma Pa dha ni Sȧ","avaroha":"Sȧ ni dha Pa Ma ga re Sa","pakkad":[{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Karuna","skippedAaroh":"","skippedAvaroh":""},"todibahaduri":{"name":"Todi-Bahaduri","thaat":"Todi","order":2580,"allowed":["Sa","re","ga","ma","Pa","dha","ni"],"vadi":"ma","samvadi":"Sa","aroha":"Sa re ga ma Pa dha ni Sȧ","avaroha":"Sȧ ni dha Pa ma ga re Sa","pakkad":[{"note":"re","oct":0},{"note":"ga","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"ma","oct":0},{"note":"ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Karuna","skippedAaroh":"","skippedAvaroh":""},"todibhupal":{"name":"Todi-Bhupal","thaat":"Todi","order":2590,"allowed":["Sa","re","ga","Pa","dha"],"vadi":"ga","samvadi":"dha","aroha":"Sa re ga Pa dha Sȧ","avaroha":"Sȧ dha Pa ga re Sa","pakkad":[{"note":"re","oct":0},{"note":"ga","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0}],"jaati":"ODAV-ODAV","time":"Morning","rasa":"Shanta","skippedAaroh":"Ma Ni","skippedAvaroh":"Ma Ni"},"todibilaskhani":{"name":"Todi-Bilaskhani","thaat":"Bhairavi","order":2600,"allowed":["Sa","re","ga","Ma","Pa","dha","ni"],"vadi":"ga","samvadi":"dha","aroha":"Sa re ga Ma Pa dha ni Sȧ","avaroha":"Sȧ ni dha Pa Ma ga re Sa","pakkad":[{"note":"ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0},{"note":"ni","oct":-1},{"note":"re","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Karuna","skippedAaroh":"","skippedAvaroh":""},"todigowardhani":{"name":"Todi-Gowardhani","thaat":"Todi","order":2610,"allowed":["Sa","re","ga","ma","Pa","dha","ni"],"vadi":"ga","samvadi":"ni","aroha":"Sa re ga ma Pa ni Sȧ","avaroha":"Sȧ ni Pa ma ga re Sa","pakkad":[{"note":"ga","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"ni","oct":0},{"note":"Pa","oct":0},{"note":"ma","oct":0},{"note":"ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0}],"jaati":"SHADAV-SHADAV","time":"Morning","rasa":"Karuna","skippedAaroh":"dha","skippedAvaroh":"dha"},"todigujari":{"name":"Todi-Gujari","thaat":"Todi","order":2620,"allowed":["Sa","re","ga","ma","Pa","dha","ni"],"vadi":"ma","samvadi":"Sa","aroha":"Sa re ga ma Pa dha ni Sȧ","avaroha":"Sȧ ni dha Pa ma ga re Sa","pakkad":[{"note":"re","oct":0},{"note":"ga","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"ni","oct":0},{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"ma","oct":0},{"note":"ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Karuna","skippedAaroh":"","skippedAvaroh":""},"todihusaini":{"name":"Todi-Husaini","thaat":"Todi","order":2630,"allowed":["Sa","re","ga","ma","Pa","dha","ni"],"vadi":"ma","samvadi":"Sa","aroha":"Sa re ga ma Pa dha ni Sȧ","avaroha":"Sȧ ni dha Pa ma ga re Sa","pakkad":[{"note":"ga","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"ma","oct":0},{"note":"ga","oct":0},{"note":"re","oct":0},{"note":"ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Shringar","skippedAaroh":"","skippedAvaroh":""},"todikhat":{"name":"Todi-Khat","thaat":"Todi","order":2640,"allowed":["Sa","re","ga","ma","Pa","dha"],"vadi":"ma","samvadi":"Sa","aroha":"Sa re ga ma Pa dha Sȧ","avaroha":"Sȧ dha Pa ma ga re Sa","pakkad":[{"note":"re","oct":0},{"note":"ga","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"Sa","oct":1},{"note":"dha","oct":0},{"note":"Pa","oct":0},{"note":"ma","oct":0},{"note":"ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0}],"jaati":"SHADAV-SHADAV","time":"Morning","rasa":"Karuna","skippedAaroh":"Ni","skippedAvaroh":"Ni"},"todilaxmi":{"name":"Todi-Laxmi","thaat":"Todi","order":2650,"allowed":["Sa","re","ga","ma","Pa","Dha","ni"],"vadi":"ma","samvadi":"Sa","aroha":"Sa re ga ma Pa Dha ni Sȧ","avaroha":"Sȧ ni Dha Pa ma ga re Sa","pakkad":[{"note":"re","oct":0},{"note":"ga","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"ma","oct":0},{"note":"ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Morning","rasa":"Karuna","skippedAaroh":"","skippedAvaroh":""},"triveni":{"name":"Triveni","thaat":"Kafi","order":2660,"allowed":["Sa","Re","ga","Ma","Pa","Dha","ni"],"vadi":"ga","samvadi":"ni","aroha":"Sa Re ga Ma Pa Dha ni Sȧ","avaroha":"Sȧ ni Dha Pa Ma ga Re Sa","pakkad":[{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Evening","rasa":"Shringar","skippedAaroh":"","skippedAvaroh":""},"vaari":{"name":"Vaari","thaat":"Khamaj","order":2670,"allowed":["Sa","Re","Ga","Ma","Pa","Dha","ni"],"vadi":"Pa","samvadi":"Re","aroha":"Sa Re Ga Ma Pa Dha ni Sȧ","avaroha":"Sȧ ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"ni","oct":0},{"note":"Sa","oct":1},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Evening","rasa":"Shringar","skippedAaroh":"","skippedAvaroh":""},"vibhas":{"name":"Vibhas","thaat":"Bhairav","order":2680,"allowed":["Sa","re","Ga","Pa","dha"],"vadi":"re","samvadi":"dha","aroha":"Sa re Ga Pa dha Sȧ","avaroha":"Sȧ dha Pa Ga re Sa","pakkad":[{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"re","oct":0},{"note":"Sa","oct":0},{"note":"dha","oct":-1},{"note":"Sa","oct":0},{"note":"re","oct":0},{"note":"Ga","oct":0},{"note":"Pa","oct":0},{"note":"dha","oct":0},{"note":"Sa","oct":1}],"jaati":"ODAV-ODAV","time":"Early Morning","rasa":"Shanta","skippedAaroh":"Ma Ni","skippedAvaroh":"Ma Ni"},"vihang":{"name":"Vihang","thaat":"Kalyan","order":2690,"allowed":["Sa","Re","Ga","ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa Re Ga ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa ma Ga Re Sa","pakkad":[{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1},{"note":"Ni","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"ma","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Evening","rasa":"Shringar","skippedAaroh":"","skippedAvaroh":""},"yamankalyan":{"name":"Yaman-Kalyan","thaat":"Kalyan","order":2700,"allowed":["Sa","Re","Ga","ma","Pa","Dha","Ni"],"vadi":"Ga","samvadi":"Ni","aroha":"Sa Re Ga ma Pa Dha Ni Sȧ","avaroha":"Sȧ Ni Dha Pa Ma Ga Re Sa","pakkad":[{"note":"Ni","oct":-1},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0},{"note":"Ni","oct":-1},{"note":"Re","oct":0},{"note":"Ga","oct":0},{"note":"ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Ni","oct":0},{"note":"Sa","oct":1}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Evening","rasa":"Shringar","skippedAaroh":"","skippedAvaroh":""},"zilaf":{"name":"Zilaf","thaat":"Kafi","order":2710,"allowed":["Sa","Re","ga","Ma","Pa","Dha","ni"],"vadi":"Ma","samvadi":"Sa","aroha":"Sa Re ga Ma Pa Dha ni Sȧ","avaroha":"Sȧ ni Dha Pa Ma ga Re Sa","pakkad":[{"note":"Re","oct":0},{"note":"ga","oct":0},{"note":"Ma","oct":0},{"note":"Pa","oct":0},{"note":"Dha","oct":0},{"note":"Pa","oct":0},{"note":"Ma","oct":0},{"note":"ga","oct":0},{"note":"Re","oct":0},{"note":"Sa","oct":0}],"jaati":"SAMPUNRA-SAMPUNRA","time":"Night","rasa":"Shringar","skippedAaroh":"","skippedAvaroh":""}};
const BUILTIN_RAGA_DATA = SEED_RAGAS;  // legacy alias used by Seed button

// ─── Audio Initialisation ────────────────────────────────────────────────────
function initAudio() {
    if (audioCtx) {
        if (audioCtx.state === 'suspended') audioCtx.resume();
        return;
    }
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();

    // Dynamics compressor prevents clipping from stacked harmonics
    compressor = audioCtx.createDynamicsCompressor();
    compressor.threshold.value = -18;
    compressor.knee.value      = 8;
    compressor.ratio.value     = 4;
    compressor.attack.value    = 0.003;
    compressor.release.value   = 0.25;

    masterBus = audioCtx.createGain();
    masterBus.gain.value = 1.0;

    // Dry/wet reverb split
    dryGain = audioCtx.createGain();
    wetGain = audioCtx.createGain();
    reverbNode = buildReverb(2.0, 2.2);

    const wetVal = parseFloat(document.getElementById('reverb-wet').value);
    dryGain.gain.value = 1.0 - wetVal * 0.4;
    wetGain.gain.value = wetVal;

    // Post-compressor volume — allows boosting up to 10× without the compressor
    // fighting the gain (compressor acts on the pre-boost signal, then we amplify).
    masterVolNode = audioCtx.createGain();
    masterVolNode.gain.value = parseFloat(document.getElementById('master-vol').value) || 1.0;

    // Chain: instruments → masterBus → compressor → masterVolNode → dry+reverb → destination
    masterBus.connect(compressor);
    compressor.connect(masterVolNode);
    masterVolNode.connect(dryGain);
    masterVolNode.connect(reverbNode);
    reverbNode.connect(wetGain);
    dryGain.connect(audioCtx.destination);
    wetGain.connect(audioCtx.destination);

    // Pre-generate noise buffers
    noiseBuffer  = buildNoiseBuffer(0.18);  // short burst for tabla head-slap
    breathBuffer = buildNoiseBuffer(2.0);   // long loopable buffer for flute breath

    // Sample bus: a single gain that all sampled notes route through.
    // Setting it to 0 on Stop immediately silences all ringing samples.
    sampleBus = audioCtx.createGain();
    sampleBus.gain.value = 1.0;
    sampleBus.connect(masterBus);

    // Start loading all instrument samples in the background (non-blocking).
    // triggerNote() uses oscillator fallback until samples are available.
    const el = document.getElementById('sample-status');
    if (el) { el.className = 'loading'; el.textContent = '♪ Loading…'; }
    Promise.allSettled(Object.keys(SAMPLE_DEFS).map(ensureSamplesLoaded))
        .then(async () => {
            updateSampleStatusPill();
        });
}

function buildReverb(durationSec, decay) {
    const rate   = audioCtx.sampleRate;
    const length = Math.floor(rate * durationSec);
    const buf    = audioCtx.createBuffer(2, length, rate);
    for (let ch = 0; ch < 2; ch++) {
        const d = buf.getChannelData(ch);
        for (let i = 0; i < length; i++) {
            d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / length, decay);
        }
    }
    const conv = audioCtx.createConvolver();
    conv.buffer = buf;
    return conv;
}

function buildNoiseBuffer(sec) {
    const len = Math.floor(audioCtx.sampleRate * sec);
    const buf = audioCtx.createBuffer(1, len, audioCtx.sampleRate);
    const d   = buf.getChannelData(0);
    for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;
    return buf;
}

// ─── Sample-based Audio Engine ───────────────────────────────────────────────
// Samples are fetched from free CDNs once and cached as AudioBuffers.
// triggerNote() prefers samples; falls back to oscillator synthesis when
// a sample for that instrument is not yet loaded.

const SAMPLE_DEFS = {
    // Locally-hosted individual note files served from Firebase Hosting.
    harmonium: {
        base: '/samples/Harmonium/',
        fallback_base: null,
        notes: ['C2','Cs2','D2','Ds2','E2','F2','Fs2','G2','Gs2','A2','As2','B2','C3','Cs3','D3','Ds3','E3','F3','Fs3','G3','Gs3','A3','As3','B3','C4','Cs4','D4','Ds4','E4','F4','Fs4','G4','Gs4','A4','As4','B4','C5','Cs5','D5','Ds5','E5','F5','Fs5']
    },
    flute: {
        base: '/samples/Flute/',
        fallback_base: null,
        notes: ['C2','Cs2','D2','Ds2','E2','F2','Fs2','G2','Gs2','A2','As2','B2','C3','Cs3','D3','Ds3','E3','F3','Fs3','G3','Gs3','A3','As3','B3','C4','Cs4','D4','Ds4','E4','F4','Fs4','G4','Gs4','A4','As4','B4','C5','Cs5','D5','Ds5','E5','F5','Fs5']
    },
    sitar: {
        base: '/samples/Sitar/',
        fallback_base: null,
        notes: ['C2','Cs2','D2','Ds2','E2','F2','Fs2','G2','Gs2','A2','As2','B2','C3','Cs3','D3','Ds3','E3','F3','Fs3','G3','Gs3','A3','As3','B3','C4','Cs4','D4','Ds4','E4','F4','Fs4','G4','Gs4','A4','As4','B4','C5','Cs5','D5','Ds5','E5','F5','Fs5']
    },
    violin: {
        base: '/samples/Violin/',
        fallback_base: null,
        notes: ['C2','Cs2','D2','Ds2','E2','F2','Fs2','G2','Gs2','A2','As2','B2','C3','Cs3','D3','Ds3','E3','F3','Fs3','G3','Gs3','A3','As3','B3','C4','Cs4','D4','Ds4','E4','F4','Fs4','G4','Gs4','A4','As4','B4','C5','Cs5','D5','Ds5','E5','F5','Fs5']
    },
    piano: {
        base: '/samples/Piano/',
        fallback_base: null,
        notes: ['C2','Cs2','D2','Ds2','E2','F2','Fs2','G2','Gs2','A2','As2','B2','C3','Cs3','D3','Ds3','E3','F3','Fs3','G3','Gs3','A3','As3','B3','C4','Cs4','D4','Ds4','E4','F4','Fs4','G4','Gs4','A4','As4','B4','C5','Cs5','D5','Ds5','E5','F5','Fs5']
    }
};

// Cache: 'harmonium|A3' → AudioBuffer (with .baseHz property attached)
const SampleCache = {};
const SampleLoadState = {};  // instrument → 'pending'|'loading'|'done'
const CACHE_NAME = 'swar-laya-samples-v4';  // bump version to force re-download

// Fetch a sample MP3 — checks the browser's persistent Cache API first,
// so after the first visit the files load from disk with zero network traffic.
async function fetchSampleArrayBuffer(url) {
    if (typeof caches !== 'undefined') {
        try {
            const cache = await caches.open(CACHE_NAME);
            let cached  = await cache.match(url);
            if (!cached) {
                const fresh = await fetch(url);
                if (!fresh.ok) throw new Error(fresh.status);
                await cache.put(url, fresh.clone());   // store for next visit
                cached = fresh;
            }
            return cached.arrayBuffer();
        } catch (e) { /* fall through to plain fetch */ }
    }
    // Cache API unavailable (e.g. file:// protocol) — plain network fetch
    const r = await fetch(url);
    if (!r.ok) throw new Error(r.status);
    return r.arrayBuffer();
}

// Convert note name (e.g. "Ds4", "A3", "Cs5") to Hz
function noteNameToHz(name) {
    const map = {C:0,Cs:1,D:2,Ds:3,E:4,F:5,Fs:6,G:7,Gs:8,A:9,As:10,B:11};
    const m   = name.match(/^([A-G]s?)(\d)$/);
    if (!m || !(m[1] in map)) return null;
    return 440 * Math.pow(2, (map[m[1]] - 9 + (parseInt(m[2]) - 4) * 12) / 12);
}

// Load all samples for one instrument; silently skips failed URLs
async function ensureSamplesLoaded(instrument) {
    if (SampleLoadState[instrument] === 'loading' ||
        SampleLoadState[instrument] === 'done') return;
    const def = SAMPLE_DEFS[instrument];
    if (!def || !audioCtx) return;

    SampleLoadState[instrument] = 'loading';
    const results = await Promise.allSettled(def.notes.map(async noteName => {
        const key = `${instrument}|${noteName}`;
        if (SampleCache[key]) return;
        let arr = null;
        try {
            arr = await fetchSampleArrayBuffer(def.base + noteName + '.mp3');
        } catch(e) {
            if (def.fallback_base)
                arr = await fetchSampleArrayBuffer(def.fallback_base + noteName + '.mp3');
            else throw e;
        }
        const buf  = await audioCtx.decodeAudioData(arr);
        buf.baseHz = noteNameToHz(noteName);
        if (buf.baseHz) SampleCache[key] = buf;
    }));
    SampleLoadState[instrument] = 'done';

    const loaded = results.filter(r => r.status === 'fulfilled').length;
    console.log(`[Samples] ${instrument}: ${loaded}/${def.notes.length} loaded`);
    updateSampleStatusPill();
}

// Return closest cached AudioBuffer for a target frequency
function findBestSample(instrument, targetHz) {
    const def = SAMPLE_DEFS[instrument];
    if (!def) return null;
    let bestBuf = null, bestDiff = Infinity;
    for (const noteName of def.notes) {
        const buf = SampleCache[`${instrument}|${noteName}`];
        if (!buf) continue;
        const diff = Math.abs(Math.log2(targetHz / buf.baseHz));
        if (diff < bestDiff) { bestDiff = diff; bestBuf = buf; }
    }
    return bestBuf;
}

// Play one note from a sample; returns true if a sample was found and played
// toFreq > 0  → SOURCE meend: note's own sample, natural attack, pitch ramps OUT toward toFreq.
// prevFreq > 0 → DEST meend (legacy / ~-meend): sample starts at prevFreq pitch, arrives at freq.
function triggerSampledNote(freq, instrument, glideTime, noteDuration, vs, prevFreq, legato = false, toFreq = 0) {
    const buf = findBestSample(instrument, freq);
    if (!buf || !sampleBus) return false;
    const now = audioCtx.currentTime;

    const src = audioCtx.createBufferSource();
    src.buffer = buf;

    const naturalRate = freq / buf.baseHz;
    if (toFreq > 0) {
        // SOURCE meend: start at natural pitch, ramp toward destination
        src.playbackRate.setValueAtTime(naturalRate, now);
        src.playbackRate.exponentialRampToValueAtTime(toFreq / buf.baseHz, now + glideTime);
    } else {
        // DEST meend (or no glide): optionally start at prevFreq pitch and arrive at naturalRate
        const startRate = (glideTime > 0.02 && prevFreq) ? (prevFreq / buf.baseHz) : naturalRate;
        src.playbackRate.setValueAtTime(startRate, now);
        if (startRate !== naturalRate) {
            src.playbackRate.exponentialRampToValueAtTime(naturalRate, now + glideTime);
        }
    }

    const env = audioCtx.createGain();

    // Legato handoff: fade out the previous note quickly so it doesn't ring
    // under the new one.  Each sampled note has its own independent GainNode,
    // so without this step both notes play simultaneously.
    if (legato && legatoEnv) {
        legatoEnv.gain.cancelScheduledValues(now);
        legatoEnv.gain.setTargetAtTime(0, now, 0.006);   // ~6 ms exponential fade
    }
    if (legato) legatoEnv = env;

    // Amplitude envelope — matched to each instrument's character
    if (instrument === 'piano' || instrument === 'sitar') {
        const peak = 0.55 * vs;
        if (legato && instrument === 'sitar') {
            // Legato sitar: pluck attack → quick decay to a resonance sustain level
            // → short release at note end (or cut by next note's handoff above).
            const resonance = peak * 0.28;
            const decayTime = Math.min(0.14, noteDuration * 0.35);
            const rel       = 0.010;
            const sus       = Math.max(0.001, noteDuration - decayTime - rel);
            env.gain.setValueAtTime(0,          now);
            env.gain.linearRampToValueAtTime(peak,      now + 0.008);
            env.gain.exponentialRampToValueAtTime(resonance, now + decayTime);
            env.gain.setValueAtTime(resonance,  now + decayTime + sus);
            env.gain.linearRampToValueAtTime(0.001, now + noteDuration);
        } else {
            // Non-legato: natural percussive decay to silence
            env.gain.setValueAtTime(0,    now);
            env.gain.linearRampToValueAtTime(peak,  now + 0.008);
            env.gain.exponentialRampToValueAtTime(0.001, now + Math.min(noteDuration * 1.5, 2.2));
        }
    } else if (instrument === 'flute') {
        const att = 0.09, rel = 0.07;
        const sus = Math.max(0.01, noteDuration - att - rel);
        const peak = 0.50 * vs;
        env.gain.setValueAtTime(0,    now);
        env.gain.linearRampToValueAtTime(peak * 0.7, now + att * 0.5);
        env.gain.linearRampToValueAtTime(peak,       now + att);
        env.gain.setValueAtTime(peak,                now + att + sus);
        env.gain.linearRampToValueAtTime(0.001,      now + att + sus + rel);
    } else {
        // harmonium, violin — attack and release scale with note length so that
        // fast sub-notes (e.g. 4 per beat) sound just as full as slow ones.
        // In legato mode use an 8 ms snap-on / 8 ms snap-off so notes connect
        // without audible gaps while still landing precisely on the beat.
        const att = legato ? 0.008 : Math.min(0.040, noteDuration * 0.18);
        const rel = legato ? 0.008 : Math.min(0.040, noteDuration * 0.12);
        const sus = Math.max(0.001, noteDuration - att - rel);
        const peak = 0.50 * vs;
        env.gain.setValueAtTime(0,    now);
        env.gain.linearRampToValueAtTime(peak, now + att);
        env.gain.setValueAtTime(peak,          now + att + sus);
        env.gain.linearRampToValueAtTime(0.001, now + att + sus + rel);
    }

    src.connect(env);
    env.connect(sampleBus);   // sampleBus → masterBus → compressor → reverb/dry
    src.start(now);
    src.stop(now + noteDuration + 0.25);
    return true;
}

function updateSampleStatusPill() {
    const el = document.getElementById('sample-status');
    if (!el) return;
    const instruments = Object.keys(SAMPLE_DEFS);
    const done = instruments.filter(i => SampleLoadState[i] === 'done').length;
    const loaded = instruments.reduce((n, i) => n + Object.keys(SampleCache)
        .filter(k => k.startsWith(i + '|')).length, 0);
    if (done === 0) { el.className = 'osc'; el.textContent = '♪ Synth'; return; }
    if (done < instruments.length) {
        el.className = 'loading';
        el.textContent = `♪ Loading… (${done}/${instruments.length})`;
    } else {
        el.className = 'ready';
        el.textContent = `♪ Samples ✓ (${loaded})`;
    }
}

function updateReverbMix() {
    if (!dryGain || !wetGain) return;
    const wet = parseFloat(document.getElementById('reverb-wet').value);
    dryGain.gain.value = 1.0 - wet * 0.4;
    wetGain.gain.value = wet;
}

function updateMasterVol() {
    const val = parseFloat(document.getElementById('master-vol').value);
    const display = document.getElementById('master-vol-display');
    if (display) display.textContent = val.toFixed(1) + '×';
    if (masterVolNode) masterVolNode.gain.value = val;
    saveSettings();
}

// ─── Tabla Synthesis ──────────────────────────────────────────────────────────
// Each bol modelled with:
//   • Bayan  – two sine sweeps (fundamental + 2nd harmonic) + low-pass noise click
//   • Dayan  – pitch glide from attack frequency down to tuned pitch (characteristic
//              tabla sound) + presence partial + two bandpass noise bursts
// All output routes through masterBus → compressor → reverb for proper mix level.
function playTablaSound(bol, isSam) {
    if (!audioCtx) return;
    const now = audioCtx.currentTime;
    const vol = parseFloat(document.getElementById('tabla-vol').value);
    const out = masterBus;              // route through master bus (compressor + reverb)
    const sw  = isSam ? 1.30 : 1.0;   // accent on sam (beat 1)

    // Tune the dayan to the composition's Sa.
    // Calibration: hardcoded values were written for Kali 4 (G#4 ≈ 415.3 Hz).
    // When the user selects a different scale, scaleRatio shifts the dayan pitch
    // proportionally so the tabla sounds in tune with the melody.
    const tuningSa   = parseFloat(document.getElementById('scale-select').value);
    const scaleRatio = tuningSa / 415.3;

    // ── Helper: sine oscillator with frequency glide then exponential decay ──
    function sineSwoop(f0, f1, sweepSec, gain, decaySec) {
        const o = audioCtx.createOscillator();
        const g = audioCtx.createGain();
        o.type = 'sine';
        o.frequency.setValueAtTime(f0, now);
        o.frequency.exponentialRampToValueAtTime(f1, now + sweepSec);
        g.gain.setValueAtTime(gain * sw * vol, now);
        g.gain.exponentialRampToValueAtTime(0.0001, now + decaySec);
        o.connect(g); g.connect(out);
        o.start(now); o.stop(now + decaySec + 0.02);
    }

    // ── Helper: bandpass / lowpass noise transient ──
    function noiseBurst(freq, q, gain, decaySec, btype) {
        if (!noiseBuffer) return;
        btype = btype || 'bandpass';
        const ns = audioCtx.createBufferSource();
        const nf = audioCtx.createBiquadFilter();
        const ng = audioCtx.createGain();
        ns.buffer = noiseBuffer;
        nf.type = btype; nf.frequency.value = freq; nf.Q.value = q;
        ng.gain.setValueAtTime(gain * sw * vol, now);
        ng.gain.exponentialRampToValueAtTime(0.0001, now + decaySec);
        ns.connect(nf); nf.connect(ng); ng.connect(out);
        ns.start(now);
    }

    // ── Tirkit (4-stroke rapid roll) ──
    if (bol.toLowerCase().indexOf('tirkit') !== -1) {
        [0, 0.055, 0.11, 0.165].forEach(function(t, i) {
            const o = audioCtx.createOscillator();
            const g = audioCtx.createGain();
            o.type = 'sine';
            o.frequency.setValueAtTime(i % 2 === 0 ? 1100 : 950, now + t);
            o.frequency.exponentialRampToValueAtTime(i % 2 === 0 ? 490 : 420, now + t + 0.012);
            g.gain.setValueAtTime(0.45 * vol, now + t);
            g.gain.exponentialRampToValueAtTime(0.0001, now + t + 0.14);
            o.connect(g); g.connect(out);
            o.start(now + t); o.stop(now + t + 0.16);
            if (noiseBuffer) {
                const ns = audioCtx.createBufferSource();
                const nf = audioCtx.createBiquadFilter();
                const ng = audioCtx.createGain();
                ns.buffer = noiseBuffer;
                nf.type = 'bandpass'; nf.frequency.value = 4000; nf.Q.value = 1.5;
                ng.gain.setValueAtTime(0.22 * vol, now + t);
                ng.gain.exponentialRampToValueAtTime(0.0001, now + t + 0.022);
                ns.connect(nf); nf.connect(ng); ng.connect(out);
                ns.start(now + t);
            }
        });
        return;
    }

    // ── Bayan (bass / left drum): Dha, Dhin, Dhi, Dhinna, Ge, Gi ──
    // Ge/Gi = open bayan tone only (no dayan follows)
    if (/Dha|Dhin|Dhi|Dhinna|Ge|Gi/.test(bol)) {
        const isGe  = /Ge|Gi/.test(bol);
        const bF0   = isGe ? 95  : 85;
        const bF1   = isGe ? 58  : 43;
        const bDec  = isGe ? 0.65 : 0.52;
        sineSwoop(bF0,       bF1,       0.080, 1.30, bDec);         // fundamental
        sineSwoop(bF0 * 2.0, bF1 * 2.0, 0.060, 0.50, bDec * 0.70); // 2nd harmonic
        noiseBurst(220, 1.2, 0.55, 0.030, 'lowpass');               // skin thump
    }

    // ── Dayan (treble / right drum) ──
    // Parameters: [attackHz, tunedHz, sweepSec, bodyGain, decaySec, noiseHz, noiseGain]
    // attackHz  = frequency at moment of strike (high, creates the characteristic glide)
    // tunedHz   = pitch the membrane settles at (the "note" of the tabla)
    const D = {
        'Tin'   : [1150, 560, 0.013, 0.92, 0.52, 4800, 0.38],
        'Tina'  : [1100, 545, 0.013, 0.88, 0.48, 4600, 0.36],
        'Dhin'  : [1100, 540, 0.012, 0.88, 0.46, 4500, 0.35],
        'Dhinna': [1050, 520, 0.012, 0.85, 0.42, 4400, 0.34],
        'Dhi'   : [1050, 510, 0.011, 0.84, 0.38, 4400, 0.33],
        'Na'    : [1050, 488, 0.011, 0.85, 0.36, 4200, 0.32],
        'Dha'   : [1000, 470, 0.010, 0.82, 0.30, 4000, 0.30],
        'Tu'    : [1000, 505, 0.012, 0.80, 0.55, 3800, 0.25],
        'Tun'   : [1000, 505, 0.013, 0.80, 0.60, 3800, 0.24],
        'Ti'    : [ 950, 470, 0.009, 0.75, 0.22, 4000, 0.30],
        'Ka'    : [ 820, 660, 0.005, 0.70, 0.07, 5200, 0.42],
        'Kat'   : [ 820, 660, 0.005, 0.68, 0.07, 5200, 0.40],
        'Ta'    : [ 780, 620, 0.006, 0.65, 0.09, 4800, 0.35]
    };
    const p = D[bol];
    if (!p) return;   // bayan-only bols (Ge, Gi) already handled above

    // Apply scaleRatio to attack + tuned pitch; noise bands stay in their fixed
    // spectral range since they represent the membrane's transient snap, not pitch.
    const f0    = p[0] * scaleRatio;   // attack Hz (scaled)
    const f1    = p[1] * scaleRatio;   // tuned Hz  (scaled — this is what you "hear" as pitch)
    const swMs  = p[2], dGain = p[3], dDec = p[4], nHz = p[5], nGain = p[6];

    sineSwoop(f0,        f1,        swMs,        dGain,        dDec);          // fundamental
    sineSwoop(f0 * 1.52, f1 * 1.52, swMs * 0.80, dGain * 0.28, dDec * 0.55); // presence partial
    noiseBurst(nHz,        1.8, nGain,        0.022);  // high-freq skin snap
    noiseBurst(nHz * 0.45, 2.5, nGain * 0.45, 0.012); // mid-range definition crack
}

// ─── Dholak Synthesis ─────────────────────────────────────────────────────────
// Dholak is a two-headed folk drum:
//   Left head (bayi)  — deeper, rounder bass; no syahi ring
//   Right head (dagga)— smaller, woodier, drier attack; no metallic tabla ring
// Frequencies are NOT scaled to Sa (dholak is not a pitched instrument like tabla).
function playDholakSound(bol, isSam) {
    if (!audioCtx) return;
    const now = audioCtx.currentTime;
    const vol = parseFloat(document.getElementById('tabla-vol').value);
    const out = masterBus;
    const sw  = isSam ? 1.25 : 1.0;

    // ── Helper: sine with frequency glide + exponential decay ──
    function sineSwoop(f0, f1, sweepSec, gain, decaySec) {
        const o = audioCtx.createOscillator();
        const g = audioCtx.createGain();
        o.type = 'sine';
        o.frequency.setValueAtTime(f0, now);
        o.frequency.exponentialRampToValueAtTime(f1, now + sweepSec);
        g.gain.setValueAtTime(gain * sw * vol, now);
        g.gain.exponentialRampToValueAtTime(0.0001, now + decaySec);
        o.connect(g); g.connect(out);
        o.start(now); o.stop(now + decaySec + 0.02);
    }

    // ── Helper: bandpass noise burst ──
    function noiseBurst(freq, q, gain, decaySec) {
        if (!noiseBuffer) return;
        const ns = audioCtx.createBufferSource();
        const nf = audioCtx.createBiquadFilter();
        const ng = audioCtx.createGain();
        ns.buffer = noiseBuffer;
        nf.type = 'bandpass'; nf.frequency.value = freq; nf.Q.value = q;
        ng.gain.setValueAtTime(gain * sw * vol, now);
        ng.gain.exponentialRampToValueAtTime(0.0001, now + decaySec);
        ns.connect(nf); nf.connect(ng); ng.connect(out);
        ns.start(now);
    }

    // ── Left head (bayi) — Dha, Dhin, Ge ──
    const hasLeft = /Dha|Dhin|Ge/.test(bol);
    if (hasLeft) {
        const isGe = /Ge/.test(bol);
        // Rounder/deeper than tabla bayan; longer resonance for open Ge
        sineSwoop(isGe ? 110 : 95,  isGe ? 70 : 52,  0.10, 1.15, isGe ? 0.72 : 0.55);   // fundamental
        sineSwoop(isGe ? 220 : 190, isGe ? 140 : 104, 0.07, 0.38, isGe ? 0.48 : 0.36);   // 2nd harmonic
        noiseBurst(190, 1.4, 0.45, 0.032);    // skin thump (lower Q = woodier)
    }

    // ── Right head (dagga) — Dha, Dhin, Na, Ti, Ka, Te ──
    // Shorter, drier, more wooden than tabla dayan.  No sustained ring.
    // [attackHz, settleHz, sweepSec, gain, decaySec, noiseHz, noiseGain]
    const R = {
        'Dha' : [520, 380, 0.013, 0.80, 0.20, 1800, 0.24],
        'Dhin': [500, 360, 0.012, 0.75, 0.18, 1700, 0.22],
        'Na'  : [480, 340, 0.011, 0.72, 0.14, 2100, 0.28],
        'Ti'  : [460, 325, 0.009, 0.68, 0.12, 2300, 0.30],
        'Ka'  : [400, 330, 0.006, 0.62, 0.06, 2600, 0.36],
        'Te'  : [440, 315, 0.009, 0.65, 0.11, 2000, 0.27],
        'Ge'  : null,   // left-only bol — no right head
    };
    const rp = R[bol] !== undefined ? R[bol] : R['Na'];  // default to Na timbre
    if (rp) {
        const [rF0, rF1, swMs, rGain, rDec, nHz, nGain] = rp;
        sineSwoop(rF0,        rF1,        swMs,        rGain,        rDec);
        sineSwoop(rF0 * 1.42, rF1 * 1.42, swMs * 0.75, rGain * 0.20, rDec * 0.48); // 2nd partial (weaker than tabla)
        noiseBurst(nHz,        2.2, nGain,        0.016);
        noiseBurst(nHz * 0.50, 3.0, nGain * 0.38, 0.010);
    }
}

// ─── Melodic Synth ────────────────────────────────────────────────────────────
// All instruments use pure-sine additive synthesis (each oscillator produces
// only its fundamental — no extra partials from sawtooth/triangle).
// Harmonic gains are balanced to match real instrument spectra.

const InstrumentHarmonics = {
    // Reed organ: strong fundamental, rich harmonics, chorus layers for warmth
    harmonium: [
        {ratio: 1.0,  gainFactor: 0.28, detuneCents:  0},
        {ratio: 2.0,  gainFactor: 0.22, detuneCents:  0},
        {ratio: 3.0,  gainFactor: 0.16, detuneCents:  0},
        {ratio: 4.0,  gainFactor: 0.10, detuneCents:  0},
        {ratio: 5.0,  gainFactor: 0.06, detuneCents:  0},
        {ratio: 6.0,  gainFactor: 0.03, detuneCents:  0},
        {ratio: 1.0,  gainFactor: 0.07, detuneCents: +5},  // chorus (reed spread)
        {ratio: 1.0,  gainFactor: 0.05, detuneCents: -5},  // chorus
        {ratio: 2.0,  gainFactor: 0.05, detuneCents: +6}   // upper chorus
    ],
    // Bansuri: mostly fundamental + octave; breath noise added separately
    flute: [
        {ratio: 1.0,  gainFactor: 0.55, detuneCents:  0},
        {ratio: 2.0,  gainFactor: 0.28, detuneCents:  0},
        {ratio: 3.0,  gainFactor: 0.08, detuneCents:  0},
        {ratio: 4.0,  gainFactor: 0.02, detuneCents:  0}
    ],
    // Sitar: bright, many harmonics; slight inharmonicity for jawari buzz
    sitar: [
        {ratio: 1.0,   gainFactor: 0.22, detuneCents:  0},
        {ratio: 2.0,   gainFactor: 0.18, detuneCents:  0},
        {ratio: 3.0,   gainFactor: 0.14, detuneCents:  0},
        {ratio: 4.0,   gainFactor: 0.11, detuneCents:  0},
        {ratio: 5.0,   gainFactor: 0.08, detuneCents:  0},
        {ratio: 6.0,   gainFactor: 0.06, detuneCents:  0},
        {ratio: 7.0,   gainFactor: 0.04, detuneCents:  0},
        {ratio: 2.01,  gainFactor: 0.04, detuneCents:  0}  // jawari shimmer
    ],
    // Violin: 2nd partial often louder than fundamental; strong odd harmonics
    violin: [
        {ratio: 1.0,  gainFactor: 0.18, detuneCents:  0},
        {ratio: 2.0,  gainFactor: 0.26, detuneCents:  0},  // 2nd strongest
        {ratio: 3.0,  gainFactor: 0.20, detuneCents:  0},
        {ratio: 4.0,  gainFactor: 0.14, detuneCents:  0},
        {ratio: 5.0,  gainFactor: 0.09, detuneCents:  0},
        {ratio: 6.0,  gainFactor: 0.05, detuneCents:  0},
        {ratio: 7.0,  gainFactor: 0.03, detuneCents:  0}
    ],
    // Piano: percussive, slight inharmonicity in upper partials
    piano: [
        {ratio: 1.000, gainFactor: 0.36, detuneCents:  0},
        {ratio: 2.001, gainFactor: 0.18, detuneCents:  0},
        {ratio: 3.006, gainFactor: 0.10, detuneCents:  0},
        {ratio: 4.015, gainFactor: 0.06, detuneCents:  0},
        {ratio: 5.030, gainFactor: 0.03, detuneCents:  0}
    ],
    // Bellpad: inharmonic bell partials (carillon-like) + shimmer layers for pad warmth
    bellpad: [
        {ratio: 1.000, gainFactor: 0.32, detuneCents:  0},   // fundamental
        {ratio: 2.000, gainFactor: 0.20, detuneCents:  0},   // octave
        {ratio: 3.000, gainFactor: 0.14, detuneCents:  0},   // 5th of 2nd octave
        {ratio: 4.170, gainFactor: 0.09, detuneCents:  0},   // bell minor 3rd partial
        {ratio: 5.400, gainFactor: 0.05, detuneCents:  0},   // upper bell partial
        {ratio: 1.000, gainFactor: 0.06, detuneCents: +7},   // shimmer layer
        {ratio: 2.000, gainFactor: 0.04, detuneCents: -6}    // shimmer octave
    ]
};

const InstrumentFilter = {
    harmonium: {type: 'lowpass',  freq:  900, Q: 1.0},  // warm reed tone
    flute:     {type: 'lowpass',  freq: 3500, Q: 0.6},  // airy, open
    sitar:     {type: 'highpass', freq:  180, Q: 0.7},  // remove low mud
    violin:    {type: 'lowpass',  freq: 3200, Q: 1.2},  // slight nasal peak
    piano:     {type: 'lowpass',  freq: 4500, Q: 0.5},  // bright, clear
    bellpad:   {type: 'lowpass',  freq: 5000, Q: 0.4}   // open, crystalline
};

// All instruments: pure sine per oscillator — correct additive synthesis.
// (sawtooth/triangle waveforms add uncontrolled extra partials that clash.)
const InstrumentOscType = {
    harmonium: 'sine',
    flute:     'sine',
    sitar:     'sine',
    violin:    'sine',
    piano:     'sine',
    bellpad:   'sine'
};

function setupSynth(instrument) {
    stopSynth();
    lastPlayedFreq = null;   // reset meend glide state for new instrument

    mainGain   = audioCtx.createGain();
    mainGain.gain.setValueAtTime(0, audioCtx.currentTime);
    mainFilter = audioCtx.createBiquadFilter();

    const fSpec = InstrumentFilter[instrument] || InstrumentFilter.harmonium;
    mainFilter.type            = fSpec.type;
    mainFilter.frequency.value = fSpec.freq;
    mainFilter.Q.value         = fSpec.Q;

    mainFilter.connect(mainGain);
    mainGain.connect(masterBus);

    const oscType = InstrumentOscType[instrument] || 'sine';
    const hSpecs  = InstrumentHarmonics[instrument] || InstrumentHarmonics.harmonium;

    harmonics = [];
    for (const spec of hSpecs) {
        const o = audioCtx.createOscillator();
        const g = audioCtx.createGain();
        o.type         = oscType;
        o.detune.value = spec.detuneCents;
        g.gain.setValueAtTime(spec.gainFactor, audioCtx.currentTime);
        o.connect(g);
        g.connect(mainFilter);
        o.start();
        harmonics.push({osc: o, gain: g, ratio: spec.ratio});
    }

    // Keep a reference to the first oscillator as "mainOsc" for frequency control
    mainOsc = harmonics.length > 0 ? harmonics[0].osc : null;

    // ── Instrument-specific effects ──────────────────────────────────────────
    if (instrument === 'flute' || instrument === 'violin') {
        // Pitch vibrato (LFO → detune AudioParam of every oscillator).
        // Depth starts at 0 and is faded in per-note inside triggerNote()
        // so the vibrato swells in naturally after the attack.
        vibratoLFO  = audioCtx.createOscillator();
        vibratoGain = audioCtx.createGain();
        vibratoLFO.type             = 'sine';
        vibratoLFO.frequency.value  = instrument === 'violin' ? 5.8 : 5.2;
        vibratoGain.gain.value      = 0;   // triggerNote ramps this up
        vibratoLFO.connect(vibratoGain);
        for (const h of harmonics) {
            vibratoGain.connect(h.osc.detune);   // ±N cents of wobble
        }
        vibratoLFO.start();
    }

    if (instrument === 'flute') {
        // Bansuri breath noise: looping white noise → bandpass filter → mixed
        // into the signal before mainGain so the ADSR silences it with the note.
        fluteNoiseNode = audioCtx.createBufferSource();
        fluteNoiseNode.buffer = breathBuffer;
        fluteNoiseNode.loop   = true;

        const breathFilter            = audioCtx.createBiquadFilter();
        breathFilter.type             = 'bandpass';
        breathFilter.frequency.value  = 800;   // centred around 800 Hz
        breathFilter.Q.value          = 1.5;

        fluteNoiseGain              = audioCtx.createGain();
        fluteNoiseGain.gain.value   = 0;       // triggerNote sets the level

        fluteNoiseNode.connect(breathFilter);
        breathFilter.connect(fluteNoiseGain);
        fluteNoiseGain.connect(mainFilter);    // noise passes through same filter+gain chain
        fluteNoiseNode.start();
    }
}

function stopSynth() {
    // Stop and disconnect instrument-specific effects first
    if (vibratoLFO)     { try { vibratoLFO.stop(); }     catch(e){} vibratoLFO  = null; }
    if (fluteNoiseNode) { try { fluteNoiseNode.stop(); }  catch(e){} fluteNoiseNode = null; }
    if (vibratoGain)    { try { vibratoGain.disconnect(); }    catch(e){} vibratoGain    = null; }
    if (fluteNoiseGain) { try { fluteNoiseGain.disconnect(); } catch(e){} fluteNoiseGain = null; }

    // Stop all harmonic oscillators
    for (const h of harmonics) { try { h.osc.stop(); } catch(e){} }
    harmonics = [];
    mainOsc   = null;
    if (mainGain) { mainGain.disconnect(); mainGain = null; }
}

// toFreq > 0 → SOURCE meend: this note starts at its own pitch and glides TOWARD toFreq.
function triggerNote(freq, instrument, glideTime, noteDuration, volScale, legato = false, toFreq = 0) {
    const vs = (volScale !== undefined) ? volScale : 1.0;

    // ── Sample-based playback (preferred when samples are loaded) ─────────────
    // For SOURCE meend (toFreq > 0) don't use prevFreq — the glide goes outward.
    const prevFreq = (toFreq === 0 && glideTime > 0.02) ? lastPlayedFreq : null;
    if (triggerSampledNote(freq, instrument, glideTime, noteDuration, vs, prevFreq, legato, toFreq)) {
        // Update lastPlayedFreq to the pitch the note ENDS on (destination for meend)
        lastPlayedFreq = toFreq > 0 ? toFreq : freq;
        return;
    }
    lastPlayedFreq = toFreq > 0 ? toFreq : freq;

    // ── Oscillator fallback (used until samples finish loading) ───────────────
    if (!mainOsc || !mainGain) return;
    const now = audioCtx.currentTime;

    // ── Frequency (with glide or near-instant snap) ────────────────────────
    for (const h of harmonics) {
        const thisFreq = Math.max(freq * h.ratio, 1.0);
        h.osc.frequency.cancelScheduledValues(now);
        if (toFreq > 0) {
            // SOURCE meend: start at this note's pitch, ramp to destination
            h.osc.frequency.setValueAtTime(thisFreq, now);
            h.osc.frequency.exponentialRampToValueAtTime(
                Math.max(toFreq * h.ratio, 1.0), now + glideTime);
        } else {
            // Normal or DEST meend: arrive at this note's pitch
            h.osc.frequency.setValueAtTime(h.osc.frequency.value || thisFreq, now);
            h.osc.frequency.exponentialRampToValueAtTime(thisFreq,
                now + Math.max(glideTime, 0.005));
        }
    }

    // ── Amplitude envelope (Attack → Sustain → Release) ───────────────────
    // Always schedule a release so the note ends cleanly before the next beat.
    mainGain.gain.cancelScheduledValues(now);
    mainGain.gain.setValueAtTime(mainGain.gain.value, now);
    const peakGain = 0.18 * vs;

    if (instrument === 'piano' || instrument === 'sitar') {
        if (legato && instrument === 'sitar') {
            // Legato sitar (synth fallback): pluck → resonance sustain → release
            const resonance = peakGain * 0.28;
            const decayTime = Math.min(0.14, noteDuration * 0.35);
            const rel       = 0.010;
            const sus       = Math.max(0.001, noteDuration - decayTime - rel);
            mainGain.gain.linearRampToValueAtTime(peakGain,   now + 0.010);
            mainGain.gain.exponentialRampToValueAtTime(resonance, now + decayTime);
            mainGain.gain.setValueAtTime(resonance,           now + decayTime + sus);
            mainGain.gain.linearRampToValueAtTime(0.0001,     now + noteDuration);
        } else {
            // Non-legato: sharp attack, natural exponential decay
            const decayEnd = Math.min(noteDuration * 1.4, 1.6);
            mainGain.gain.linearRampToValueAtTime(peakGain,  now + 0.010);
            mainGain.gain.exponentialRampToValueAtTime(0.001, now + decayEnd);
        }

    } else if (instrument === 'bellpad') {
        // Bell strike + pad sustain: fast attack, bell-like initial decay to a
        // warm pad level, then clean release at note end.
        const bellDecay = Math.min(noteDuration * 0.45, 1.2);
        const padLevel  = peakGain * 0.28;
        const rel       = Math.min(0.10, noteDuration * 0.12);
        const sus       = Math.max(0.01, noteDuration - bellDecay - rel);
        mainGain.gain.linearRampToValueAtTime(peakGain,  now + 0.012);
        mainGain.gain.exponentialRampToValueAtTime(padLevel, now + bellDecay);
        mainGain.gain.setValueAtTime(padLevel,           now + bellDecay + sus);
        mainGain.gain.linearRampToValueAtTime(0.0001,    now + bellDecay + sus + rel);

    } else if (instrument === 'flute') {
        // Bansuri: slow breath attack, sustain, gentle release
        const att = 0.090;
        const rel = 0.070;
        const sus = Math.max(0.01, noteDuration - att - rel);
        mainGain.gain.linearRampToValueAtTime(peakGain * 0.75, now + att * 0.55);
        mainGain.gain.linearRampToValueAtTime(peakGain,        now + att);
        mainGain.gain.setValueAtTime(peakGain,                 now + att + sus);
        mainGain.gain.linearRampToValueAtTime(0.0001,          now + att + sus + rel);

        // Breath noise: same envelope profile as the tonal component
        if (fluteNoiseGain) {
            fluteNoiseGain.gain.cancelScheduledValues(now);
            fluteNoiseGain.gain.setValueAtTime(0,      now);
            fluteNoiseGain.gain.linearRampToValueAtTime(0.040,  now + att);
            fluteNoiseGain.gain.setValueAtTime(0.040,           now + att + sus);
            fluteNoiseGain.gain.linearRampToValueAtTime(0,      now + att + sus + rel);
        }
        // Vibrato: fades in after the attack settles
        if (vibratoGain) {
            vibratoGain.gain.cancelScheduledValues(now);
            vibratoGain.gain.setValueAtTime(0,   now);
            vibratoGain.gain.linearRampToValueAtTime(0,   now + att);
            vibratoGain.gain.linearRampToValueAtTime(20,  now + att + 0.12); // ±20 cents
        }

    } else {
        // Harmonium / violin: attack and release scale with note length so that
        // fast sub-notes sound just as full as slow single-beat notes.
        // Legato overrides to 8 ms snap-on / snap-off for seamless note flow.
        const att = legato ? 0.008 : Math.min(0.045, noteDuration * 0.18);
        const rel = legato ? 0.008 : Math.min(0.045, noteDuration * 0.12);
        const sus = Math.max(0.001, noteDuration - att - rel);
        mainGain.gain.linearRampToValueAtTime(peakGain, now + att);
        mainGain.gain.setValueAtTime(peakGain,          now + att + sus);
        mainGain.gain.linearRampToValueAtTime(0.0001,   now + att + sus + rel);

        // Violin vibrato: swell in after attack
        if (instrument === 'violin' && vibratoGain) {
            vibratoGain.gain.cancelScheduledValues(now);
            vibratoGain.gain.setValueAtTime(0,   now);
            vibratoGain.gain.linearRampToValueAtTime(0,   now + att);
            vibratoGain.gain.linearRampToValueAtTime(14,  now + att + 0.10); // ±14 cents
        }
    }
}

function silenceNote() {
    if (!mainGain) return;
    mainGain.gain.setTargetAtTime(0, audioCtx.currentTime, 0.04);
}

// ─── Playback Engine ──────────────────────────────────────────────────────────
// singleLineEl: if provided, only that line plays (loops); null = full song
// singleLineEl: null = whole sheet; a .line-container = that line;
//               an array of .line-container = just those lines (e.g. transcription drafts)
function startPlayback(singleLineEl = null) {
    initAudio();
    stopPlayback();
    // stopPlayback() silences sampleBus for 300 ms (to let ringing notes die).
    // Cancel that timer and restore immediately so the very first note is audible.
    if (sampleBus) {
        sampleBus.gain.cancelScheduledValues(audioCtx.currentTime);
        sampleBus.gain.setValueAtTime(1, audioCtx.currentTime);
    }

    const allLines = Array.from(document.querySelectorAll('.line-container'));
    if (!allLines.length) return;
    const lines = Array.isArray(singleLineEl) ? singleLineEl : (singleLineEl ? [singleLineEl] : allLines);
    if (!lines.length) return;

    const baseSa    = parseFloat(document.getElementById('scale-select').value);
    const masterBpm = parseInt(document.getElementById('bpm').value) || 90;
    // Single-line mode always starts from beat 0 of that line
    const startBeat = singleLineEl ? 0 : (parseInt(document.getElementById('start-beat').value) || 1) - 1;
    // Pre-roll is meaningful only for full-song playback; skip it for single-line play.
    const preRoll   = singleLineEl ? 0 : (parseInt(document.getElementById('pre-roll-select').value) || 0);

    // Total beats across the active lines
    let totalBeats = 0;
    lines.forEach(l => { totalBeats += parseInt(l.getAttribute('data-beats')); });

    let preRollLeft    = preRoll;
    let globalBeat     = startBeat % totalBeats;
    let lastInstrument = '';

    // Pre-roll uses master BPM and first active line's taal
    const masterBeatMs = (60 / masterBpm) * 1000;
    const refTaal      = lines[0].getAttribute('data-taal');
    const refBols      = TaalBols[refTaal] || [];

    // ── Single tick function, rescheduled each beat with the current line's tempo ──
    // Replaces the old fixed setInterval so Adlib/Alap lines can run at their
    // own slower BPM while the main composition lines use the master BPM.
    function tick() {
        // ── Pre-roll (tabla only, always at master BPM) ──
        if (preRollLeft > 0) {
            const bolIdx = (preRoll - preRollLeft) % refBols.length;
            playTablaSound(refBols[bolIdx] || 'Dha', bolIdx === 0);
            preRollLeft--;
            playbackTimer = setTimeout(tick, masterBeatMs);
            return;
        }

        if (globalBeat >= totalBeats) {
            if (document.getElementById('loop-toggle')?.checked) {
                globalBeat = 0;   // loop: wrap around to start
            } else {
                stopPlayback();   // play-once: stop cleanly
                return;
            }
        }

        // ── Find which line owns this beat ──
        let accumulated = 0, targetLine = null, localIdx = 0;
        for (const line of lines) {
            const b = parseInt(line.getAttribute('data-beats'));
            if (globalBeat < accumulated + b) {
                targetLine = line;
                localIdx   = globalBeat - accumulated;
                break;
            }
            accumulated += b;
        }

        // ── Effective BPM for this beat ───────────────────────────────────────
        // Any line can set its own tempo via the Line BPM input.
        // Adlib/Alap lines typically use a slow per-line BPM (e.g. 30–50).
        // Leave Line BPM blank to fall back to the master BPM.
        let beatMs = masterBeatMs;
        if (targetLine) {
            const lineBpmEl = targetLine.querySelector('.line-bpm');
            const lineBpm   = lineBpmEl ? parseFloat(lineBpmEl.value) : 0;
            if (lineBpm > 0) beatMs = (60 / lineBpm) * 1000;
        }

        // ── Clear previous highlights (ruler + note cells) ──
        document.querySelectorAll('#taal-ruler .beat-num').forEach(el => el.classList.remove('playing'));
        document.querySelectorAll('.grid-row[data-row="notes"] .cell.playing').forEach(el => el.classList.remove('playing'));

        if (targetLine) {
            // Auto-scroll + ruler update when playback enters a new line
            if (targetLine !== lastPlayingLine) {
                lastPlayingLine = targetLine;
                const lineTaal = targetLine.getAttribute('data-taal');
                const linePerc = targetLine.querySelector('.line-perc-sel')?.value || 'tabla';
                renderTaalRuler(lineTaal, linePerc);
                targetLine.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }

            // Highlight current beat in the ruler and in the note cell
            const beatEl = document.querySelector(`#taal-ruler .beat-num[data-beat-idx="${localIdx}"]`);
            if (beatEl) beatEl.classList.add('playing');
            const noteCell = targetLine.querySelector(`.grid-row[data-row="notes"] .cell[data-beat-idx="${localIdx}"]`);
            if (noteCell) noteCell.classList.add('playing');

            const inst    = targetLine.querySelector('.line-inst-select').value;
            // Adlib lines always silence percussion regardless of the dropdown
            const isAdlib  = targetLine.querySelector('.line-adlib') &&
                             targetLine.querySelector('.line-adlib').checked;
            const isLegato = !!(targetLine.querySelector('.line-legato')?.checked);
            const percSel  = targetLine.querySelector('.line-perc-sel');
            const percType = isAdlib ? 'mute' : (percSel ? percSel.value : 'tabla');
            const lineVol  = parseFloat(targetLine.querySelector('.line-vol').value);
            const taalKey  = targetLine.getAttribute('data-taal');

            // Re-setup synth only when instrument changes between lines
            if (inst !== lastInstrument) {
                setupSynth(inst);
                lastInstrument = inst;
            }

            // Percussion — tabla, dholak, or mute
            if (percType === 'dholak') {
                const bols = DholakBols[taalKey] || [];
                playDholakSound(bols[localIdx] || 'Dha', localIdx === 0);
            } else if (percType === 'tabla') {
                const bols = TaalBols[taalKey] || [];
                playTablaSound(bols[localIdx] || 'Dha', localIdx === 0);
            }
            // percType === 'mute' → do nothing

            // Notes — support variable sub-beats per beat
            const notesRow = targetLine.querySelector('.grid-row[data-row="notes"]');
            if (notesRow) {
                const cells = notesRow.querySelectorAll('.cell');
                if (cells[localIdx]) {
                    const units = cells[localIdx].querySelectorAll('.note-unit');
                    const subMs = beatMs / (units.length || 1);

                    units.forEach((unit, si) => {
                        setTimeout(() => {
                            const swara = unit.querySelector('.note-select').value;

                            // "~" = hold/sustain: previous note is still ringing.
                            // If meend is active on this ~, and it is the FIRST in a
                            // consecutive ~(m) chain, launch a long glide from the last
                            // played pitch to the next sounding note right now.
                            // Subsequent ~(m) beats in the same chain are skipped normally.
                            if (swara === '~') {
                                const mb = unit.querySelector('.meend-btn');
                                if (mb && mb.classList.contains('active') && lastPlayedFreq) {
                                    // Is the PREVIOUS beat also a ~(meend)?  If so the
                                    // slide was already started there — skip.
                                    const prevCell = localIdx > 0 ? cells[localIdx - 1] : null;
                                    const prevIsMeendTilde =
                                        prevCell &&
                                        prevCell.querySelector('.note-select')?.value === '~' &&
                                        prevCell.querySelector('.meend-btn')?.classList.contains('active');

                                    if (!prevIsMeendTilde) {
                                        // Count consecutive ~(meend) beats from here
                                        let slideBeats = 1;
                                        for (let ci = localIdx + 1; ci < cells.length; ci++) {
                                            const ns = cells[ci].querySelector('.note-select')?.value;
                                            const nm = cells[ci].querySelector('.meend-btn')?.classList.contains('active');
                                            if (ns === '~' && nm) slideBeats++;
                                            else break;
                                        }
                                        // Target note = first non-~ cell after the chain
                                        const tCell = cells[localIdx + slideBeats];
                                        const tSel  = tCell?.querySelector('.note-select');
                                        const tRatio = tSel ? SwaraMap[tSel.value] : null;
                                        if (tRatio) {
                                            let tf = baseSa * tRatio;
                                            const tOct = tCell.querySelector('.octave-btn.active');
                                            if (tOct) tf *= Math.pow(2, parseInt(tOct.dataset.oct));
                                            // Glide covers 92% of the sustained period so the
                                            // pitch arrives naturally just before the target beat.
                                            const slideDur = slideBeats * beatMs / 1000;
                                            triggerNote(tf, inst, slideDur * 0.92, slideDur, lineVol, isLegato);
                                        }
                                    }
                                }
                                return;
                            }

                            const ratio   = SwaraMap[swara];
                            const isMeend = unit.querySelector('.meend-btn') &&
                                            unit.querySelector('.meend-btn').classList.contains('active');

                            // Duration = cover trailing '-'/'~' sub-beats in THIS cell
                            // (subDur) PLUS any full hold-beats that follow (holdBeats).
                            let subDur = 1;
                            for (let k = si + 1; k < units.length; k++) {
                                const sv = units[k].querySelector('.note-select').value;
                                if (sv === '-' || sv === '~') subDur++;
                                else break;
                            }
                            const reachesEnd = (si + subDur === units.length);
                            const holdBeats  = reachesEnd ? countSustains(targetLine, localIdx) : 0;
                            const noteDuration = (subDur * subMs + holdBeats * beatMs) / 1000;

                            // SOURCE meend convention: this note glides TOWARD the next real note.
                            // Look ahead in remaining sub-beats, then in subsequent cells.
                            let glide  = 0.008;
                            let toFreq = 0;
                            if (isMeend) {
                                let found = false;
                                for (let k = si + subDur; k < units.length && !found; k++) {
                                    const sv = units[k].querySelector('.note-select').value;
                                    if (sv === '-' || sv === '~') continue;
                                    const r = SwaraMap[sv];
                                    if (r) {
                                        toFreq = baseSa * r;
                                        const ob = units[k].querySelector('.octave-btn.active');
                                        if (ob) toFreq *= Math.pow(2, parseInt(ob.dataset.oct));
                                        found = true;
                                    }
                                }
                                for (let ci = localIdx + 1; ci < cells.length && !found; ci++) {
                                    const nUnits = cells[ci].querySelectorAll('.note-unit');
                                    for (const nu of nUnits) {
                                        if (found) break;
                                        const sv = nu.querySelector('.note-select')?.value;
                                        if (!sv || sv === '-' || sv === '~') continue;
                                        const r = SwaraMap[sv];
                                        if (r) {
                                            toFreq = baseSa * r;
                                            const ob = nu.querySelector('.octave-btn.active');
                                            if (ob) toFreq *= Math.pow(2, parseInt(ob.dataset.oct));
                                            found = true;
                                        }
                                    }
                                }
                                if (toFreq > 0) glide = noteDuration * 0.92;
                            }

                            if (ratio) {
                                let freq = baseSa * ratio;
                                const octBtn = unit.querySelector('.octave-btn.active');
                                if (octBtn) freq *= Math.pow(2, parseInt(octBtn.dataset.oct));
                                triggerNote(freq, inst, glide, noteDuration, lineVol, isLegato, toFreq);
                            } else {
                                silenceNote();
                            }
                        }, si * subMs);
                    });
                }
            }
        }

        globalBeat++;
        playbackTimer = setTimeout(tick, beatMs);   // reschedule at this line's tempo
    }

    playbackTimer = setTimeout(tick, 0);   // start immediately
}

function stopPlayback() {
    if (playbackTimer) { clearTimeout(playbackTimer); playbackTimer = null; }
    stopSynth();
    // Instantly silence any ringing sampled notes, then restore the bus after
    // they've all been garbage-collected by the browser
    if (sampleBus) {
        sampleBus.gain.cancelScheduledValues(audioCtx.currentTime);
        sampleBus.gain.setValueAtTime(0, audioCtx.currentTime);
        setTimeout(() => { if (sampleBus) sampleBus.gain.setValueAtTime(1, audioCtx.currentTime); }, 300);
    }
    lastPlayedFreq  = null;
    lastPlayingLine = null;
    legatoEnv       = null;
    document.querySelectorAll('#taal-ruler .beat-num').forEach(el => el.classList.remove('playing'));
    document.querySelectorAll('.grid-row[data-row="notes"] .cell.playing').forEach(el => el.classList.remove('playing'));
    // Clear the orange "active" state on all per-line play buttons
    document.querySelectorAll('.btn-play-line.active').forEach(b => b.classList.remove('active'));
    renderTaalRuler(document.getElementById('taal-select').value, 'tabla');
}

// ─── Tanpura (Cycling String Drone) ──────────────────────────────────────────
function toggleTanpura() {
    initAudio();
    isTanpuraOn = !isTanpuraOn;
    const btn = document.getElementById('tanpuraBtn');
    if (isTanpuraOn) {
        btn.textContent = '🪕 Tanpura: ON';
        startTanpura();
    } else {
        btn.textContent = '🪕 Tanpura: OFF';
        stopTanpura();
    }
}

function startTanpura() {
    stopTanpura();
    const sa = parseFloat(document.getElementById('scale-select').value);
    // Classic tanpura string order: Pa (lower), lower Sa, Sa, Sa
    const strings = [sa * 0.75, sa * 0.5, sa, sa];
    let idx = 0;

    function pluck() {
        if (!isTanpuraOn) return;
        const freq = strings[idx % strings.length];
        idx++;

        const osc  = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        const lfo  = audioCtx.createOscillator();
        const lfog = audioCtx.createGain();

        osc.type          = 'sawtooth';
        osc.frequency.value = freq;

        // Subtle shimmer (sympathetic resonance simulation)
        lfo.frequency.value = 3.5 + Math.random() * 1.5;
        lfog.gain.value     = freq * 0.0025;
        lfo.connect(lfog);
        lfog.connect(osc.frequency);

        gain.gain.setValueAtTime(0, audioCtx.currentTime);
        gain.gain.linearRampToValueAtTime(0.038, audioCtx.currentTime + 0.1);
        gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 3.0);

        osc.connect(gain);
        gain.connect(masterBus || audioCtx.destination);
        osc.start(); lfo.start();
        osc.stop(audioCtx.currentTime + 3.1);
        lfo.stop(audioCtx.currentTime + 3.1);
        tanpuraNodes.push(osc, lfo);
    }

    pluck();
    tanpuraTimer = setInterval(pluck, 650);
}

function stopTanpura() {
    if (tanpuraTimer) { clearInterval(tanpuraTimer); tanpuraTimer = null; }
    tanpuraNodes.forEach(n => { try { n.stop(); } catch(e){} });
    tanpuraNodes = [];
}

function updateTanpura() { if (isTanpuraOn) startTanpura(); }

