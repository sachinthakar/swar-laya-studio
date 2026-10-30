// Swar-Laya Studio v5 — share: read-only "?share=<id>" viewer mode
//
// Visiting the normal app URL with a "share" query parameter (as produced by
// cloud.js's shareComposition()) loads that one composition from the public
// "shared" Firestore collection (no sign-in needed to READ it -- see
// firestore.rules) straight into this same page, then switches the page into
// a read-only "listening" layout: editing controls are hidden or disabled,
// and a banner explains what's being viewed. Everything else (this is still
// the full Studio page) is unchanged, so Play/Stop/Loop, the instrument
// volumes, Tanpura and Reverb all work normally for whoever opens the link.
(function () {
    const shareId = new URLSearchParams(location.search).get('share');
    if (!shareId) return;   // normal editor page -- nothing to do

    function enterReadOnlyMode(data) {
        window._histSuspend = true;
        applyCompositionData(data);
        window._histSuspend = false;

        document.body.classList.add('shared-view');

        const banner = document.createElement('div');
        banner.className = 'shared-view-banner';
        const openUrl = location.origin + location.pathname;
        banner.innerHTML =
            `&#128065; Viewing a shared composition, read-only &mdash; ` +
            `<a href="${openUrl}">open Swar-Laya Studio</a> to write your own.`;
        document.body.insertBefore(banner, document.body.firstChild);
    }

    function showError(msg) {
        const banner = document.createElement('div');
        banner.className = 'shared-view-banner shared-view-error';
        banner.textContent = msg;
        document.body.insertBefore(banner, document.body.firstChild);
    }

    // cloud.js assigns `db` once Firebase has finished initialising; this
    // page's scripts can run before that, so wait for it rather than race it.
    function waitForDbThenLoad() {
        if (typeof db === 'undefined' || !db) { setTimeout(waitForDbThenLoad, 150); return; }
        db.collection('shared').doc(shareId).get()
            .then(doc => {
                if (!doc.exists) { showError('This shared link is no longer available.'); return; }
                enterReadOnlyMode(doc.data());
            })
            .catch(e => showError('Could not load the shared composition: ' + e.message));
    }
    waitForDbThenLoad();
})();
