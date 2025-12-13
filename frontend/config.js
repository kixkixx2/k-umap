// API Configuration
// Update BACKEND_URL with your Render backend URL after deployment
const API_CONFIG = {
    // Development: Use localhost
    // Production: Use your Render URL (e.g., https://patient-clustering.onrender.com)
    // If the page is opened via file:// (local file), default to localhost backend
    BACKEND_URL: (window.location.protocol === 'file:') || window.location.hostname === ''
        ? 'http://localhost:5000'
        : (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
            ? 'http://localhost:5000'
            : 'https://patient-clustering-api.onrender.com', // Updated with actual Render URL
    
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
