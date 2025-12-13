// API Configuration
// Automatically detects environment and uses appropriate backend URL
const API_CONFIG = {
    // Auto-detect: use relative URLs in production (same origin), localhost for local dev
    BACKEND_URL: (() => {
        // Local file access - use localhost
        if (window.location.protocol === 'file:' || window.location.hostname === '') {
            return 'http://localhost:5000';
        }
        // Local development
        if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
            return 'http://localhost:5000';
        }
        // Production: use same origin (relative URLs work since frontend is served by Flask)
        return '';
    })(),
    
    // API Endpoints
    ENDPOINTS: {
        PREDICT: '/predict',
        STATS: '/stats',
        PATIENTS: '/patients',
        CLUSTER_DATA: '/cluster-data'
    }
};

// Helper function to build full API URL
function getApiUrl(endpoint) {
    return `${API_CONFIG.BACKEND_URL}${API_CONFIG.ENDPOINTS[endpoint] || endpoint}`;
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { API_CONFIG, getApiUrl };
}
