// config.js
const IS_PRODUCTION = true;
const BACKEND_URL = 'https://drlaw-backend.onrender.com';

// Make it globally available
window.APP_CONFIG = {
  backendUrl: BACKEND_URL,
  isProduction: IS_PRODUCTION
};

