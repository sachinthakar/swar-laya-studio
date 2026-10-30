// Swar-Laya Studio v5 — export: "Export Audio" (record the composition to a WAV file)
//
// Deliberately NOT a second, offline scheduler re-implementing note timing,
// instruments, tabla/dholak, tanpura and reverb mixing from scratch -- that
// would risk silently drifting out of sync with what Play actually sounds
// like. Instead this taps the real engine's final mixed output (dryGain +
// wetGain -- see engine.js's initAudio(), the same two nodes that already
// feed audioCtx.destination) into a MediaStreamAudioDestinationNode, records
// that with the browser's own MediaRecorder while a single, non-looping pass
// of startPlayback() runs, then decodes the recording back into an
// AudioBuffer and re-encodes it as a plain WAV file for download. What you
// hear on Play is exactly what gets exported, because it's the same audio.
let _exportActive   = false;
let _exportRecorder = null;
let _exportChunks   = [];
let _exportTapDest  = null;

function exportAudio() {
    if (_exportActive) return;
    if (!document.querySelectorAll('.line-container').length) {
        alert("Add some notes first — there's nothing to export yet.");
        return;
    }
    if (typeof MediaRecorder === 'undefined' || !window.MediaStreamAudioDestinationNode) {
        alert('Audio export is not supported in this browser. Try a recent version of Chrome, Edge or Firefox.');
        return;
    }

    initAudio();   // same call startPlayback() makes -- ensures audioCtx/dryGain/wetGain exist
    if (!dryGain || !wetGain) {
        alert('Audio engine not ready yet — click ▶ Play once first, then try Export Audio again.');
        return;
    }

    const loopEl      = document.getElementById('loop-toggle');
    const wasLooping  = loopEl.checked;
    loopEl.checked    = false;   // export always renders exactly one pass through

    _exportTapDest = audioCtx.createMediaStreamDestination();
    dryGain.connect(_exportTapDest);
    wetGain.connect(_exportTapDest);

    _exportChunks = [];
    const mimeType = MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm'
                   : MediaRecorder.isTypeSupported('audio/mp4')  ? 'audio/mp4'
                   : '';
    _exportRecorder = new MediaRecorder(_exportTapDest.stream, mimeType ? { mimeType } : undefined);
    _exportRecorder.ondataavailable = e => { if (e.data && e.data.size) _exportChunks.push(e.data); };
    _exportRecorder.onstop = () => _finishExport(wasLooping);

    _exportActive = true;
    showToast('Recording… playing through once');
    _exportRecorder.start();
    startPlayback();

    // startPlayback() clears playbackTimer itself (via stopPlayback()) once a
    // non-looping pass reaches the end -- poll for that rather than
    // pre-computing a duration, so this works correctly for Adlib/Alap lines,
    // per-line tempo overrides and any future timing feature without needing
    // a parallel duration calculator kept in sync by hand.
    const checkEnd = setInterval(() => {
        if (!_exportActive) { clearInterval(checkEnd); return; }
        if (!playbackTimer) {
            clearInterval(checkEnd);
            // Let reverb tails and the tanpura's last pluck ring out before
            // the recording is cut, instead of chopping the ending off.
            setTimeout(() => {
                if (_exportRecorder && _exportRecorder.state === 'recording') _exportRecorder.stop();
            }, 2500);
        }
    }, 150);
}

async function _finishExport(wasLooping) {
    try {
        if (dryGain && _exportTapDest) dryGain.disconnect(_exportTapDest);
        if (wetGain && _exportTapDest) wetGain.disconnect(_exportTapDest);
    } catch (e) { /* already disconnected -- harmless */ }
    document.getElementById('loop-toggle').checked = wasLooping;
    _exportActive  = false;
    _exportTapDest = null;

    showToast('Rendering audio file…');
    try {
        const blob      = new Blob(_exportChunks, { type: (_exportRecorder && _exportRecorder.mimeType) || 'audio/webm' });
        const arrayBuf  = await blob.arrayBuffer();
        const decoded   = await audioCtx.decodeAudioData(arrayBuf);
        const wavBlob   = _audioBufferToWav(decoded);

        const rawTitle = (document.getElementById('song-title').value || 'composition').trim();
        const fileName = (rawTitle.replace(/[^\w\- ]+/g, '').trim() || 'composition') + '.wav';

        const url = URL.createObjectURL(wavBlob);
        const a   = document.createElement('a');
        a.href = url; a.download = fileName;
        document.body.appendChild(a); a.click(); a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 4000);

        showToast('Audio exported ✓');
    } catch (e) {
        alert('Export failed: ' + e.message);
    }
}

// Plain 16-bit PCM WAV encoder for an AudioBuffer -- no external library
// needed for a feature this small and self-contained.
function _audioBufferToWav(buffer) {
    const numCh   = buffer.numberOfChannels;
    const sr      = buffer.sampleRate;
    const samples = buffer.length;
    const bytesPerSample = 2;
    const blockAlign     = numCh * bytesPerSample;
    const dataSize       = samples * blockAlign;
    const bufOut  = new ArrayBuffer(44 + dataSize);
    const view    = new DataView(bufOut);

    function writeStr(offset, str) {
        for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
    }

    writeStr(0, 'RIFF');
    view.setUint32(4, 36 + dataSize, true);
    writeStr(8, 'WAVE');
    writeStr(12, 'fmt ');
    view.setUint32(16, 16, true);          // PCM fmt chunk size
    view.setUint16(20, 1, true);           // audio format = PCM
    view.setUint16(22, numCh, true);
    view.setUint32(24, sr, true);
    view.setUint32(28, sr * blockAlign, true);  // byte rate
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, 16, true);          // bits per sample
    writeStr(36, 'data');
    view.setUint32(40, dataSize, true);

    const channelData = [];
    for (let ch = 0; ch < numCh; ch++) channelData.push(buffer.getChannelData(ch));

    let offset = 44;
    for (let i = 0; i < samples; i++) {
        for (let ch = 0; ch < numCh; ch++) {
            let s = Math.max(-1, Math.min(1, channelData[ch][i]));
            s = s < 0 ? s * 0x8000 : s * 0x7fff;
            view.setInt16(offset, s, true);
            offset += 2;
        }
    }
    return new Blob([bufOut], { type: 'audio/wav' });
}
