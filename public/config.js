// Swar-Laya Studio — runtime configuration for the Render deployment.
// apiBase: URL of the Raga API web service (Render dashboard -> swar-laya-api -> URL).
// firebase: web app config for project "swar-laya-studio" (used for sign-in and saved songs).
//   Firebase web config values are public identifiers, not secrets; access is
//   controlled by Firestore security rules and Firebase Auth authorized domains.
window.SWAR_CONFIG = {
  apiBase: "https://swar-laya-api.onrender.com",
  firebase: {
    apiKey:            "AIzaSyCsVf2UuPEgexC7WoQSK0ONxuiOUxo0a88",
    authDomain:        "swar-laya-studio.firebaseapp.com",
    projectId:         "swar-laya-studio",
    storageBucket:     "swar-laya-studio.firebasestorage.app",
    messagingSenderId: "508121720546",
    appId:             "1:508121720546:web:61c8535e222ac0916ae0bb"
  }
};
if (window.firebase && !firebase.apps.length) {
  firebase.initializeApp(window.SWAR_CONFIG.firebase);
}
