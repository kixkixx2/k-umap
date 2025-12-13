// API Configuration
// API Configuration - Use config from config.js
const API_BASE_URL = API_CONFIG.BACKEND_URL;

// Global variables
let allPatientsData = [];

// Color scheme for clusters
const CLUSTER_COLORS = [
    '#10B981',
    '#F59E0B',
    '#EF4444',
    '#8B5CF6',
    '#EC4899',
    '#14B8A6',
    '#F97316',
    '#6366F1',
    '#A855F7',
    '#0EA5E9'
];

const DEFAULT_CLUSTER_LABELS = {
    0: 'Metabolic Risk',
    1: 'Preventive Focus',
    2: 'Acute Respiratory'
};

const FEATURE_COLOR_SETTINGS = {
    bmi: {
        cmin: 15,
        cmax: 50,
        colorscale: 'Viridis',
        colorbarTitle: 'BMI (kg/m²)'
    },
    age_years: {
        cmin: 16,
        cmax: 25,
        colorscale: 'Blues',
        colorbarTitle: 'Age (years)'
    },
    year_level: {
        cmin: 1,
        cmax: 4,
        colorscale: 'Plasma',
        colorbarTitle: 'Year Level'
    }
};

let clusterProfiles = {};

// ===== INTERACTIVE ENHANCEMENTS =====

// Initialize interactive features on page load
function initInteractiveFeatures() {
    // Add intersection observer for scroll animations
    initScrollAnimations();
    
    // Add number counting animation for stats
    initNumberAnimations();
    
    // Add drag-over effect for file uploads
    initDragDropEffects();
    
    // Add cursor trail effect (subtle)
    // initCursorTrail();
    
    // Add parallax effect to hero section
    initParallaxEffect();
    
    // Add typing effect to headers
    initTypingEffect();
}

// Scroll-triggered animations
function initScrollAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);
    
    // Observe all animatable elements
    document.querySelectorAll('.card, .stat-card, .result-box, .chart-container').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(30px)';
        el.style.transition = 'all 0.6s ease-out';
        observer.observe(el);
    });
}

// Animated number counting
function initNumberAnimations() {
    const animateNumber = (element, target, duration = 1000) => {
        const start = 0;
        const startTime = performance.now();
        
        const updateNumber = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            // Ease out cubic
            const easeOut = 1 - Math.pow(1 - progress, 3);
            const current = Math.floor(start + (target - start) * easeOut);
            
            element.textContent = current.toLocaleString();
            
            if (progress < 1) {
                requestAnimationFrame(updateNumber);
            } else {
                element.textContent = target.toLocaleString();
            }
        };
        
        requestAnimationFrame(updateNumber);
    };
    
    // Store the function globally for use when stats are loaded
    window.animateNumber = animateNumber;
}

// Drag and drop visual effects
function initDragDropEffects() {
    document.querySelectorAll('.file-upload-area').forEach(area => {
        area.addEventListener('dragenter', (e) => {
            e.preventDefault();
            area.classList.add('drag-over');
        });
        
        area.addEventListener('dragleave', (e) => {
            e.preventDefault();
            area.classList.remove('drag-over');
        });
        
        area.addEventListener('dragover', (e) => {
            e.preventDefault();
        });
        
        area.addEventListener('drop', (e) => {
            area.classList.remove('drag-over');
        });
    });
}

// Parallax effect for hero section
function initParallaxEffect() {
    const hero = document.querySelector('.dashboard-hero');
    if (!hero) return;
    
    window.addEventListener('scroll', () => {
        const scrolled = window.pageYOffset;
        const rate = scrolled * 0.3;
        hero.style.transform = `translateY(${rate}px)`;
    }, { passive: true });
}

// Typing effect for headers
function initTypingEffect() {
    const heroTitle = document.querySelector('.hero-title, h1');
    if (heroTitle && !heroTitle.dataset.typed) {
        heroTitle.dataset.typed = 'true';
        heroTitle.classList.add('gradient-text');
    }
}

// Add ripple effect to buttons
function addRippleEffect(event) {
    const button = event.currentTarget;
    const ripple = document.createElement('span');
    const rect = button.getBoundingClientRect();
    const size = Math.max(rect.width, rect.height);
    const x = event.clientX - rect.left - size / 2;
    const y = event.clientY - rect.top - size / 2;
    
    ripple.style.cssText = `
        position: absolute;
        width: ${size}px;
        height: ${size}px;
        left: ${x}px;
        top: ${y}px;
        background: rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        transform: scale(0);
        animation: ripple 0.6s ease-out;
        pointer-events: none;
    `;
    
    button.appendChild(ripple);
    setTimeout(() => ripple.remove(), 600);
}

// Initialize ripple effect on all buttons
document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.btn').forEach(btn => {
        btn.style.position = 'relative';
        btn.style.overflow = 'hidden';
        btn.addEventListener('click', addRippleEffect);
    });
});

// Smooth page transitions
function smoothPageTransition(url) {
    document.body.style.opacity = '0';
    document.body.style.transition = 'opacity 0.3s ease';
    setTimeout(() => {
        window.location.href = url;
    }, 300);
}

// Add loading state to buttons
function setButtonLoading(button, isLoading) {
    if (isLoading) {
        button.dataset.originalText = button.innerHTML;
        button.innerHTML = `
            <svg class="animate-spin" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10" stroke-opacity="0.25"></circle>
                <path d="M12 2a10 10 0 0 1 10 10" stroke-linecap="round"></path>
            </svg>
            Loading...
        `;
        button.disabled = true;
        button.style.opacity = '0.7';
    } else {
        button.innerHTML = button.dataset.originalText || button.innerHTML;
        button.disabled = false;
        button.style.opacity = '1';
    }
}

// Toast notification with animation
function showToast(message, type = 'info', duration = 3000) {
    const toast = document.createElement('div');
    toast.className = `notification notification-${type}`;
    toast.innerHTML = `
        <span class="notification-icon">${type === 'success' ? '✓' : type === 'error' ? '✕' : 'ℹ'}</span>
        <span class="notification-message">${message}</span>
        <button class="notification-close" onclick="this.parentElement.remove()">×</button>
    `;
    
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease-out forwards';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', initInteractiveFeatures);

const COORDINATE_CANDIDATES = {
    x: ['x', 'umap_x', 'umapX', 'umap1'],
    y: ['y', 'umap_y', 'umapY', 'umap2']
};

function getFeatureColorSettings(featureKey) {
    if (!featureKey) {
        return null;
    }
    if (FEATURE_COLOR_SETTINGS[featureKey]) {
        return FEATURE_COLOR_SETTINGS[featureKey];
    }
    const normalized = featureKey.toLowerCase();
    return FEATURE_COLOR_SETTINGS[normalized] || null;
}

function getClusterProfile(clusterId) {
    return clusterProfiles[clusterId] || null;
}

function getClusterLabel(clusterId) {
    const profile = getClusterProfile(clusterId);
    if (profile && profile.label) {
        return profile.label;
    }
    return DEFAULT_CLUSTER_LABELS[clusterId] || `Cluster ${clusterId}`;
}

async function loadClusterProfiles() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/cluster_summary`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        const data = await response.json();
        if (data.success && data.profiles) {
            clusterProfiles = Object.keys(data.profiles).reduce((acc, key) => {
                const clusterId = Number(key);
                acc[clusterId] = data.profiles[key];
                return acc;
            }, {});
            updateClusterChipLabels();
        }
    } catch (error) {
        console.warn('⚠️ Cluster profile metadata unavailable. Falling back to defaults.', error);
    }
}

function updateClusterChipLabels() {
    const chipMappings = [
        { elementId: 'cluster0Label', clusterId: 0 },
        { elementId: 'cluster1Label', clusterId: 1 },
        { elementId: 'cluster2Label', clusterId: 2 }
    ];
    chipMappings.forEach(({ elementId, clusterId }) => {
        const el = document.getElementById(elementId);
        if (el) {
            el.textContent = `Cluster ${clusterId}`;
        }
    });
    const legendMappings = [
        { elementId: 'legendCluster0Label', clusterId: 0 },
        { elementId: 'legendCluster1Label', clusterId: 1 },
        { elementId: 'legendCluster2Label', clusterId: 2 }
    ];
    legendMappings.forEach(({ elementId, clusterId }) => {
        const el = document.getElementById(elementId);
        if (el) {
            el.textContent = `Cluster ${clusterId}`;
        }
    });
}

// Helper to build a cluster display string without duplicating the cluster id
function formatClusterDisplay(clusterId, label) {
    const base = `Cluster ${clusterId}`;
    if (!label) return base;
    const normalized = String(label).trim();
    // If the label already contains the cluster id or starts with 'Cluster', don't duplicate it
    if (normalized === base || /^(Cluster\s*\d+)/i.test(normalized) || normalized.includes(base)) {
        return base;
    }
    return `${base} · ${normalized}`;
}

function getCoordinateValue(patient, axis = 'x') {
    const candidates = COORDINATE_CANDIDATES[axis] || [];
    for (const key of candidates) {
        const value = patient[key];
        if (value !== undefined && value !== null && value !== '') {
            const numeric = Number(value);
            if (!Number.isNaN(numeric)) {
                return numeric;
            }
        }
    }
    return 0;
}

function getFeatureValue(patient, featureKey) {
    if (!featureKey || featureKey === 'cluster') {
        return null;
    }
    if (patient.features && patient.features[featureKey] !== undefined) {
        return patient.features[featureKey];
    }
    const directValue = patient[featureKey];
    if (directValue === undefined || directValue === null || directValue === '') {
        return null;
    }
    const numeric = Number(directValue);
    return Number.isNaN(numeric) ? null : numeric;
}

function hasDisplayableValue(value) {
    return value !== null && value !== undefined && value !== '' && value !== 'N/A';
}

function coalescePatientValue(patient, keys = []) {
    if (!patient || !Array.isArray(keys)) {
        return null;
    }
    const sources = [patient, patient.features || {}];
    for (const key of keys) {
        for (const source of sources) {
            if (source && hasDisplayableValue(source[key])) {
                return source[key];
            }
        }
    }
    return null;
}

function formatNumericValue(value, decimals = 0) {
    const numeric = Number(value);
    if (Number.isNaN(numeric)) {
        return null;
    }
    return decimals > 0 ? numeric.toFixed(decimals) : numeric.toString();
}

function formatPercentageValue(value, decimals = 0) {
    const numeric = Number(value);
    if (Number.isNaN(numeric)) {
        return null;
    }
    return `${(numeric * 100).toFixed(decimals)}%`;
}

function formatYesNoValue(value) {
    if (!hasDisplayableValue(value)) {
        return null;
    }
    if (typeof value === 'string') {
        const normalized = value.toLowerCase();
        if (['yes', '1', 'true'].includes(normalized)) return 'Yes';
        if (['no', '0', 'false'].includes(normalized)) return 'No';
    }
    const numeric = Number(value);
    if (!Number.isNaN(numeric)) {
        if (numeric === 1) return 'Yes';
        if (numeric === 0) return 'No';
    }
    return value;
}

function deriveGenderValue(patient) {
    const gender = coalescePatientValue(patient, ['gender']);
    if (hasDisplayableValue(gender)) {
        return gender;
    }
    const femaleFlag = coalescePatientValue(patient, ['is_female']);
    if (!hasDisplayableValue(femaleFlag)) {
        return null;
    }
    const numeric = Number(femaleFlag);
    if (!Number.isNaN(numeric)) {
        return numeric === 1 ? 'Female' : numeric === 0 ? 'Male' : null;
    }
    if (typeof femaleFlag === 'string') {
        const normalized = femaleFlag.toLowerCase();
        if (normalized.includes('female')) return 'Female';
        if (normalized.includes('male')) return 'Male';
    }
    return null;
}

function formatBloodPressureValue(systolic, diastolic) {
    const systolicFormatted = formatNumericValue(systolic, 0);
    const diastolicFormatted = formatNumericValue(diastolic, 0);
    if (!systolicFormatted || !diastolicFormatted) {
        return null;
    }
    return `${systolicFormatted}/${diastolicFormatted} mmHg`;
}

function formatPatientName(patient) {
    if (!patient) {
        return null;
    }
    const first = coalescePatientValue(patient, ['First_Name', 'first_name', 'first']);
    const last = coalescePatientValue(patient, ['Last_Name', 'last_name', 'last']);
    const display = coalescePatientValue(patient, ['display_name']);

    const firstStr = hasDisplayableValue(first) ? String(first).trim() : '';
    const lastStr = hasDisplayableValue(last) ? String(last).trim() : '';

    if (firstStr || lastStr) {
        if (firstStr && lastStr) {
            return `${firstStr} - ${lastStr}`;
        }
        return firstStr || lastStr;
    }

    if (hasDisplayableValue(display)) {
        return String(display).trim();
    }

    return null;
}

function normalizeTextValue(value) {
    if (value === null || value === undefined) {
        return '';
    }
    return String(value).trim().toLowerCase();
}

function getPatientNameCandidates(patient) {
    if (!patient) {
        return [];
    }
    const candidates = [];
    const displayName = normalizeTextValue(coalescePatientValue(patient, ['display_name', 'displayName']));
    if (displayName) {
        candidates.push(displayName);
    }

    const first = normalizeTextValue(coalescePatientValue(patient, ['First_Name', 'first_name', 'first']));
    const last = normalizeTextValue(coalescePatientValue(patient, ['Last_Name', 'last_name', 'last']));

    if (first || last) {
        if (first && last) {
            candidates.push(`${first} ${last}`);
            candidates.push(`${last} ${first}`);
            candidates.push(`${last}, ${first}`);
        }
        if (first) {
            candidates.push(first);
        }
        if (last) {
            candidates.push(last);
        }
    }

    return candidates;
}

// Initialize application
document.addEventListener('DOMContentLoaded', async () => {
    console.log('🚀 ClusterMed Home Page initialized');
    
    // Check API health
    await checkAPIHealth();

    // Fetch cluster profile metadata before rendering any UI text
    await loadClusterProfiles();
    
    // Load all patients data for visualization
    await loadAllPatientsData();
    
    // Setup patient search autocomplete
    if (typeof patientSearch !== 'undefined' && allPatientsData.length > 0) {
        patientSearch.setPatientData(allPatientsData);
        patientSearch.setupAutocomplete('patientSearchInput', 'autocompleteResults');
        console.log('✅ Patient search autocomplete initialized');
    }
    
    // Setup event listeners
    setupEventListeners();
    
    // Update stats
    updateStats();
    
    // Render initial scatter plot
    renderScatterPlot();
});

// Check if API is running
async function checkAPIHealth() {
    const statusIndicator = document.getElementById('apiStatus');
    const statusText = document.getElementById('apiStatusText');
    
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        
        if (data.status === 'healthy') {
            statusIndicator.className = 'status-indicator online';
            statusText.textContent = 'API Online';
        } else {
            statusIndicator.className = 'status-indicator offline';
            statusText.textContent = 'API Offline';
        }
    } catch (error) {
        statusIndicator.className = 'status-indicator offline';
        statusText.textContent = 'API Offline';
        console.error('❌ API health check failed:', error);
        showNotification('API server is not running. Please start the Flask server.', 'error');
    }
}

// Load all patients data from backend
async function loadAllPatientsData() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/get_all_patients`);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        allPatientsData = await response.json();
        console.log(`✅ Loaded ${allPatientsData.length} patients for visualization`);
        
    } catch (error) {
        console.error('❌ Failed to load patients data:', error);
        showNotification('Failed to load visualization data. Make sure train.py has been run.', 'warning');
        allPatientsData = [];
    }
}

// Update statistics
function updateStats() {
    // Count unique clusters
    const uniqueClusters = new Set(allPatientsData.map(p => p.cluster));
    document.getElementById('totalClusters').textContent = uniqueClusters.size || '-';
    
    // Total patients
    document.getElementById('totalPatients').textContent = 
        allPatientsData.length > 0 ? allPatientsData.length.toLocaleString() : '-';
    
    // Model status
    document.getElementById('modelStatus').textContent = 
        allPatientsData.length > 0 ? 'Ready' : 'Training Required';
    
    // Update cluster summary chips dynamically for known clusters
    const clusterCounts = allPatientsData.reduce((acc, patient) => {
        const key = typeof patient.cluster === 'number' ? patient.cluster : 'unknown';
        acc[key] = (acc[key] || 0) + 1;
        return acc;
    }, {});

    [
        { elementId: 'cluster0Count', clusterId: 0 },
        { elementId: 'cluster1Count', clusterId: 1 },
        { elementId: 'cluster2Count', clusterId: 2 }
    ].forEach(({ elementId, clusterId }) => {
        const el = document.getElementById(elementId);
        if (el) {
            const count = clusterCounts[clusterId] || 0;
            el.textContent = `${count.toLocaleString()} patients`;
        }
    });
}

// Setup event listeners
function setupEventListeners() {
    // Search button
    document.getElementById('searchBtn').addEventListener('click', handleSearch);
    
    // Refresh button
    document.getElementById('refreshBtn').addEventListener('click', handleRefresh);
    
    // Enter key on search input
    document.getElementById('patientSearchInput').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            handleSearch();
        }
    });
    
    // Feature selector for visualization
    const featureSelector = document.getElementById('featureSelector');
    if (featureSelector) {
        featureSelector.addEventListener('change', () => {
            renderScatterPlot();
        });
    }
    
    // Cluster explanation toggle for dashboard
    const dashboardExplanationToggle = document.getElementById('dashboardClusterExplanationToggle');
    const dashboardExplanationContent = document.getElementById('dashboardClusterExplanationContent');
    
    if (dashboardExplanationToggle) {
        dashboardExplanationToggle.addEventListener('click', () => {
            dashboardExplanationToggle.classList.toggle('active');
            dashboardExplanationContent.classList.toggle('show');
        });
    }
}

// Handle refresh data
async function handleRefresh() {
    const refreshBtn = document.getElementById('refreshBtn');
    const originalText = refreshBtn.innerHTML;
    
    try {
        // Disable button and show loading state
        refreshBtn.disabled = true;
        refreshBtn.innerHTML = '<span class="btn-icon">⏳</span> Refreshing...';
        
        // Reload patient data
        await loadAllPatientsData();
        
        // Update stats
        updateStats();
        
        // Re-render scatter plot
        renderScatterPlot();
        
        showNotification(`Data refreshed! Total patients: ${allPatientsData.length}`, 'success');
        
    } catch (error) {
        console.error('Refresh error:', error);
        showNotification('Failed to refresh data', 'error');
    } finally {
        // Re-enable button
        refreshBtn.disabled = false;
        refreshBtn.innerHTML = originalText;
    }
}

// Handle patient search
function handleSearch() {
    const searchInput = document.getElementById('patientSearchInput');
    const searchValue = searchInput.value.trim();
    
    if (!searchValue) {
        showNotification('Please enter a Patient ID', 'warning');
        return;
    }
    
    // Normalize search for flexible ID formats (e.g., 22-0641)
    const normalizedInput = searchValue.toLowerCase();
    
    // Search for patient in loaded data, treating IDs as strings
    let patient = allPatientsData.find(p => {
        const patientId = (p.patient_id ?? '').toString().trim().toLowerCase();
        return patientId === normalizedInput;
    });

    if (!patient) {
        patient = allPatientsData.find(p => {
            return getPatientNameCandidates(p).some(candidate => candidate.includes(normalizedInput));
        });
    }
    
    if (patient) {
        displaySearchResults(patient);
    } else {
        showNotification(`Patient ID "${searchValue}" not found in the dataset`, 'error');
        document.getElementById('searchResults').style.display = 'none';
        document.getElementById('patientDetailsCard').style.display = 'none';
    }
}

// Display search results
async function displaySearchResults(patient) {
    const resultsDiv = document.getElementById('searchResults');
    resultsDiv.style.display = 'block';
    
    // Helper function to safely display values
    const safeValue = (value) => (value !== null && value !== undefined && value !== 'N/A') ? value : 'N/A';
    
    // Update result values - Basic info in results grid
    const patientIdLabel = patient.auto_generated_id ? `${patient.patient_id || 'N/A'} (Auto ID)` : (patient.patient_id || 'N/A');
    document.getElementById('searchPatientId').textContent = patientIdLabel;
    const clusterLabelText = getClusterLabel(patient.cluster);
    const clusterDisplayName = formatClusterDisplay(patient.cluster, clusterLabelText);
    document.getElementById('searchCluster').textContent = clusterDisplayName;
    document.getElementById('searchCluster').className = `result-value cluster-badge cluster-${patient.cluster}`;
    
    // Fetch full patient details to populate all fields
    try {
        const response = await fetch(`${API_BASE_URL}/api/get_patient/${patient.patient_id}`);
        if (response.ok) {
            const result = await response.json();
            if (result.success && result.patient) {
                const fullPatient = result.patient;

                const yearLevelRaw = coalescePatientValue(fullPatient, ['year_level']);
                const yearLevelValue = formatNumericValue(yearLevelRaw, 0) || (hasDisplayableValue(yearLevelRaw) ? yearLevelRaw : null);
                document.getElementById('searchYearLevel').textContent = yearLevelValue || 'N/A';

                const ageValue = formatNumericValue(coalescePatientValue(fullPatient, ['age', 'age_years']), 0);
                document.getElementById('searchAge').textContent = ageValue || 'N/A';

                const genderValue = deriveGenderValue(fullPatient);
                document.getElementById('searchGender').textContent = genderValue || 'N/A';

                const bmiValue = formatNumericValue(coalescePatientValue(fullPatient, ['BMI', 'bmi']), 1);
                document.getElementById('searchBMI').textContent = bmiValue || 'N/A';

                const respiratory = formatYesNoValue(coalescePatientValue(fullPatient, ['has_respiratory_issue']));
                document.getElementById('searchRespiratory').textContent = respiratory || 'N/A';

                const pain = formatYesNoValue(coalescePatientValue(fullPatient, ['has_pain']));
                document.getElementById('searchPain').textContent = pain || 'N/A';

                const fever = formatYesNoValue(coalescePatientValue(fullPatient, ['has_fever']));
                document.getElementById('searchFever').textContent = fever || 'N/A';

                const allergy = formatYesNoValue(coalescePatientValue(fullPatient, ['has_allergy']));
                document.getElementById('searchAllergy').textContent = allergy || 'N/A';

                const uti = formatYesNoValue(coalescePatientValue(fullPatient, ['is_uti']));
                document.getElementById('searchUTI').textContent = uti || 'N/A';

                const heartRate = formatNumericValue(coalescePatientValue(fullPatient, ['heart_rate']), 0);
                const heartRateEl = document.getElementById('searchHeartRate');
                if (heartRateEl) {
                    heartRateEl.textContent = heartRate ? `${heartRate} bpm` : 'N/A';
                }

                const cholesterol = formatNumericValue(coalescePatientValue(fullPatient, ['cholesterol_total']), 0);
                const cholesterolEl = document.getElementById('searchCholesterol');
                if (cholesterolEl) {
                    cholesterolEl.textContent = cholesterol ? `${cholesterol} mg/dL` : 'N/A';
                }

                const glucose = formatNumericValue(coalescePatientValue(fullPatient, ['blood_glucose']), 0);
                const glucoseEl = document.getElementById('searchGlucose');
                if (glucoseEl) {
                    glucoseEl.textContent = glucose ? `${glucose} mg/dL` : 'N/A';
                }

                const medications = coalescePatientValue(fullPatient, ['num_medications']);
                const medsEl = document.getElementById('searchMedications');
                if (medsEl) {
                    medsEl.textContent = hasDisplayableValue(medications) ? medications : 'N/A';
                }

                const visits = coalescePatientValue(fullPatient, ['doctor_visits_per_year']);
                const visitsEl = document.getElementById('searchVisits');
                if (visitsEl) {
                    visitsEl.textContent = hasDisplayableValue(visits) ? visits : 'N/A';
                }

                const successRate = formatPercentageValue(coalescePatientValue(fullPatient, ['treatment_success_rate']));
                const successEl = document.getElementById('searchSuccess');
                if (successEl) {
                    successEl.textContent = successRate || 'N/A';
                }
                
                // Update cluster recommendation
                const recDiv = document.getElementById('clusterRec');
                const recText = document.getElementById('recText');
                if (recDiv && recText) {
                    recDiv.style.display = 'flex';
                    const profile = fullPatient.cluster_profile || getClusterProfile(fullPatient.cluster);
                    if (profile) {
                        const riskHeadline = profile.risk_summary || profile.summary || clusterDisplayName;
                        const careFocus = profile.care_focus || profile.label;
                        const actionPlan = Array.isArray(profile.recommendations) && profile.recommendations.length > 0
                            ? profile.recommendations[0]
                            : null;
                        const badgeLabel = profile.risk_level || 'Clinical Insight';
                        const cohortLabel = profile.label || clusterDisplayName;
                        const summaryCopy = riskHeadline || cohortLabel;
                        const focusCopy = careFocus && careFocus !== summaryCopy ? careFocus : '';
                        const actionCopy = actionPlan && actionPlan !== focusCopy ? actionPlan : '';
                        recText.innerHTML = `
                            <div class="rec-risk-row">
                                <span class="rec-risk-badge">${badgeLabel}</span>
                                <span class="rec-label-text">${cohortLabel}</span>
                            </div>
                            <p class="rec-summary">${summaryCopy}</p>
                            ${focusCopy ? `<p class="rec-focus">${focusCopy}</p>` : ''}
                            ${actionCopy ? `<p class="rec-action">${actionCopy}</p>` : ''}
                        `;
                    } else {
                        recText.textContent = 'Cluster guidance unavailable. Review comprehensive vitals before making care decisions.';
                    }
                }
                
                // Populate detailed cluster explanation
                const activeProfile = fullPatient.cluster_profile || getClusterProfile(fullPatient.cluster);
                populateDashboardClusterExplanation(fullPatient.cluster, activeProfile);

                if (typeof patientSearch !== 'undefined' && patientSearch.addToHistory) {
                    patientSearch.addToHistory(fullPatient.patient_id, fullPatient);
                }
            }
        }
    } catch (error) {
        console.error('Error loading full patient details:', error);
    }
    
    // Highlight patient on scatter plot
    renderScatterPlot(patient);
    
    // Scroll to results
    resultsDiv.scrollIntoView({ behavior: 'smooth' });
    
    showNotification('Patient found successfully!', 'success');
}

// Load full patient details
async function loadPatientDetails(patientId) {
    try {
        const response = await fetch(`${API_BASE_URL}/api/get_patient/${patientId}`);
        
        if (!response.ok) {
            throw new Error(`Failed to load patient details`);
        }
        
        const result = await response.json();
        
        if (result.success && result.patient) {
            displayPatientDetails(result.patient);
        }
    } catch (error) {
        console.error('Error loading patient details:', error);
        showNotification('Could not load full patient details', 'warning');
    }
}

// Display full patient details
function displayPatientDetails(patient) {
    const detailsCard = document.getElementById('patientDetailsCard');
    detailsCard.style.display = 'block';
    
    // Helper function to safely display values (handles null, undefined, 'N/A')
    const safeValue = (value) => (value !== null && value !== undefined && value !== 'N/A') ? value : 'N/A';
    
    // Update patient ID
    const detailIdLabel = patient.auto_generated_id ? `${safeValue(patient.patient_id)} (Auto ID)` : safeValue(patient.patient_id);
    document.getElementById('detailsPatientId').textContent = detailIdLabel;
    
    // Demographics
    const detailAge = formatNumericValue(coalescePatientValue(patient, ['age', 'age_years']), 0);
    document.getElementById('detailAge').textContent = detailAge || 'N/A';
    const detailGender = deriveGenderValue(patient);
    document.getElementById('detailGender').textContent = detailGender || 'N/A';
    document.getElementById('detailEthnicity').textContent = safeValue(patient.ethnicity);
    document.getElementById('detailInsurance').textContent = safeValue(patient.insurance_type);
    
    // Vital Signs
    const detailBMI = formatNumericValue(coalescePatientValue(patient, ['BMI', 'bmi']), 1);
    document.getElementById('detailBMI').textContent = detailBMI || 'N/A';
    
    const detailSystolic = coalescePatientValue(patient, ['systolic_bp', 'systolic']);
    const detailDiastolic = coalescePatientValue(patient, ['diastolic_bp', 'diastolic']);
    const detailBP = formatBloodPressureValue(detailSystolic, detailDiastolic);
    document.getElementById('detailBloodPressure').textContent = detailBP || 'N/A';
    
    const detailGlucose = formatNumericValue(coalescePatientValue(patient, ['blood_glucose']), 0);
    document.getElementById('detailGlucose').textContent = detailGlucose ? `${detailGlucose} mg/dL` : 'N/A';
    
    const detailCholesterol = formatNumericValue(coalescePatientValue(patient, ['cholesterol_total']), 0);
    document.getElementById('detailCholesterol').textContent = detailCholesterol ? `${detailCholesterol} mg/dL` : 'N/A';
    
    // Health Conditions
    const diabetes = formatYesNoValue(coalescePatientValue(patient, ['diabetes']));
    document.getElementById('detailDiabetes').textContent = diabetes || 'N/A';
    
    const hypertension = formatYesNoValue(coalescePatientValue(patient, ['hypertension']));
    document.getElementById('detailHypertension').textContent = hypertension || 'N/A';
    
    const heartDisease = formatYesNoValue(coalescePatientValue(patient, ['heart_disease']));
    document.getElementById('detailHeartDisease').textContent = heartDisease || 'N/A';
    
    document.getElementById('detailSmoking').textContent = safeValue(patient.smoking_status);
    document.getElementById('detailAlcohol').textContent = safeValue(patient.alcohol_consumption);
    
    // Treatment
    const visits = coalescePatientValue(patient, ['doctor_visits_per_year']);
    document.getElementById('detailVisits').textContent = hasDisplayableValue(visits) ? visits : 'N/A';
    const meds = coalescePatientValue(patient, ['num_medications']);
    document.getElementById('detailMedications').textContent = hasDisplayableValue(meds) ? meds : 'N/A';
    
    const adherence = formatPercentageValue(coalescePatientValue(patient, ['medication_adherence']));
    document.getElementById('detailAdherence').textContent = adherence || 'N/A';
    
    const success = formatPercentageValue(coalescePatientValue(patient, ['treatment_success_rate']));
    document.getElementById('detailSuccess').textContent = success || 'N/A';
    
    // Show note if it's a new patient
    if (patient.note) {
        const noteElement = document.createElement('div');
        noteElement.className = 'patient-note';
        noteElement.style.cssText = 'background: rgba(79, 70, 229, 0.1); border-left: 4px solid #4F46E5; padding: 1rem; margin-top: 1rem; border-radius: 8px; color: #E5E7EB;';
        noteElement.innerHTML = `<strong>ℹ️ Note:</strong> ${patient.note}`;
        
        const detailsCard = document.getElementById('patientDetailsCard');
        const existingNote = detailsCard.querySelector('.patient-note');
        if (existingNote) existingNote.remove();
        detailsCard.appendChild(noteElement);
    }
    
    // Scroll to details
    detailsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Create scatter plot with Plotly
function renderScatterPlot(highlightPatient = null) {
    if (allPatientsData.length === 0) {
        document.getElementById('clusterPlot').innerHTML = 
            '<div class="plot-error">⚠️ No data available. Please run train.py first.</div>';
        return;
    }
    
    // Get selected feature
    const featureSelector = document.getElementById('featureSelector');
    const selectedFeature = featureSelector ? featureSelector.value : 'cluster';
    
    // Create traces based on selected feature
    const traces = [];
    
    if (selectedFeature === 'cluster') {
        // Group patients by cluster
        const clusterGroups = {};
        allPatientsData.forEach(patient => {
            const cluster = patient.cluster;
            if (!clusterGroups[cluster]) {
                clusterGroups[cluster] = [];
            }
            clusterGroups[cluster].push(patient);
        });
        
        // Create traces for each cluster
        Object.keys(clusterGroups).sort((a, b) => parseInt(a) - parseInt(b)).forEach(cluster => {
            const patients = clusterGroups[cluster];
            const clusterNum = parseInt(cluster);
            const label = getClusterLabel(clusterNum) || 'Mixed Risk';
            const displayName = formatClusterDisplay(clusterNum, label);
            
            traces.push({
            x: patients.map(p => getCoordinateValue(p, 'x')),
            y: patients.map(p => getCoordinateValue(p, 'y')),
                mode: 'markers',
                type: 'scatter',
                name: displayName,
                text: patients.map(p => `Patient: ${p.patient_id || 'N/A'}<br>${displayName}`),
                marker: {
                    size: 8,
                    color: CLUSTER_COLORS[clusterNum % CLUSTER_COLORS.length],
                    opacity: 0.7
                },
                showlegend: true
            });
        });
    } else {
        // Color by selected feature
        const x = allPatientsData.map(p => getCoordinateValue(p, 'x'));
        const y = allPatientsData.map(p => getCoordinateValue(p, 'y'));
        const featureValues = allPatientsData.map(p => {
            const value = getFeatureValue(p, selectedFeature);
            return value !== null ? value : 0;
        });
        const colorSettings = getFeatureColorSettings(selectedFeature) || {};
        const featureDisplayName = selectedFeature.replace(/_/g, ' ');
        const hoverText = allPatientsData.map(p => {
            const value = getFeatureValue(p, selectedFeature);
            const featureDisplay = value !== null ? value : 'N/A';
            const clusterLabel = getClusterLabel(p.cluster);
            const labelSuffix = (clusterLabel && !(/^(Cluster\s*\d+)/i.test(clusterLabel))) ? ` · ${clusterLabel}` : '';
            return `Patient: ${p.patient_id || 'N/A'}<br>Cluster: ${p.cluster}${labelSuffix}<br>${selectedFeature}: ${featureDisplay}`;
        });
        const markerSettings = {
            size: 8,
            color: featureValues,
            colorscale: colorSettings.colorscale || 'Viridis',
            showscale: true,
            opacity: 0.7,
            colorbar: {
                title: colorSettings.colorbarTitle || featureDisplayName,
                titleside: 'right',
                titlefont: { color: '#E5E7EB' },
                tickfont: { color: '#E5E7EB' },
                bgcolor: '#1F2937',
                bordercolor: '#374151',
                borderwidth: 1
            },
            line: {
                color: 'white',
                width: 0.5
            }
        };
        if (typeof colorSettings.cmin === 'number') {
            markerSettings.cmin = colorSettings.cmin;
            markerSettings.cauto = false;
        }
        if (typeof colorSettings.cmax === 'number') {
            markerSettings.cmax = colorSettings.cmax;
            markerSettings.cauto = false;
        }
        
        traces.push({
            x: x,
            y: y,
            mode: 'markers',
            type: 'scatter',
            name: 'All Patients',
            text: hoverText,
            hovertemplate: '%{text}<extra></extra>',
            marker: markerSettings,
            showlegend: false
        });
    }
    
    // Add highlighted patient if provided
    if (highlightPatient) {
        const highlightedLabel = getClusterLabel(highlightPatient.cluster) || 'Cluster Member';
        traces.push({
            x: [getCoordinateValue(highlightPatient, 'x')],
            y: [getCoordinateValue(highlightPatient, 'y')],
            mode: 'markers',
            type: 'scatter',
            name: 'Selected Patient',
            text: [`Patient: ${highlightPatient.patient_id || 'N/A'}<br>${formatClusterDisplay(highlightPatient.cluster, highlightedLabel)}`],
            marker: {
                size: 20,
                color: '#EF4444',
                symbol: 'star',
                line: {
                    color: 'white',
                    width: 2
                }
            }
        });
    }
    
    // Layout configuration
    const featureName = selectedFeature === 'cluster' ? 'Cluster' : selectedFeature.replace(/_/g, ' ');
    const layout = {
        title: {
            text: `Patient Distribution (UMAP) - Colored by ${featureName}`,
            font: { size: 18, color: '#E5E7EB' }
        },
        xaxis: {
            title: 'UMAP Dimension 1',
            gridcolor: '#374151',
            zerolinecolor: '#4B5563',
            color: '#9CA3AF'
        },
        yaxis: {
            title: 'UMAP Dimension 2',
            gridcolor: '#374151',
            zerolinecolor: '#4B5563',
            color: '#9CA3AF'
        },
        plot_bgcolor: '#1F2937',
        paper_bgcolor: '#111827',
        font: {
            color: '#E5E7EB'
        },
        hovermode: 'closest',
        showlegend: true,
        legend: {
            x: 1.02,
            y: 1,
            bgcolor: '#1F2937',
            bordercolor: '#374151',
            borderwidth: 1
        },
        margin: {
            l: 60,
            r: 60,
            t: 60,
            b: 60,
            pad: 5
        },
        autosize: true
    };
    
    // Plot configuration
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['lasso2d', 'select2d']
    };
    
    // Create the plot - use Plotly.react for better updates
    const plotDiv = document.getElementById('clusterPlot');
    Plotly.newPlot(plotDiv, traces, layout, config).then(function() {
        // Ensure plot fits container after creation
        if (plotDiv) {
            Plotly.Plots.resize(plotDiv);
        }
        
        // Add visualization enhancements (stats overlay only)
        if (typeof vizEnhancements !== 'undefined') {
            vizEnhancements.addClusterStatsOverlay('clusterPlotContainer', allPatientsData);
        }
    });
}

// Show notification
function showNotification(message, type = 'info') {
    // Remove existing notifications
    const existingNotif = document.querySelector('.notification');
    if (existingNotif) {
        existingNotif.remove();
    }
    
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    
    const icon = {
        'success': '✅',
        'error': '❌',
        'warning': '⚠️',
        'info': 'ℹ️'
    }[type] || 'ℹ️';
    
    notification.innerHTML = `
        <span class="notification-icon">${icon}</span>
        <span class="notification-message">${message}</span>
        <button class="notification-close" onclick="this.parentElement.remove()">×</button>
    `;
    
    document.body.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.style.animation = 'slideOut 0.3s ease-out';
            setTimeout(() => notification.remove(), 300);
        }
    }, 5000);
}

function renderListOrText(element, listValues, fallbackText) {
    if (!element) {
        return;
    }
    if (Array.isArray(listValues)) {
        const cleaned = listValues
            .filter(Boolean)
            .map((value) => String(value).trim())
            .filter(Boolean);
        if (cleaned.length) {
            const listItems = cleaned.map(item => `<li>${item}</li>`).join('');
            element.innerHTML = `<ul class="cluster-detail-list">${listItems}</ul>`;
            return;
        }
    }
    element.textContent = fallbackText || 'No details available yet.';
}

// Populate detailed cluster explanation for dashboard
function populateDashboardClusterExplanation(cluster, profileOverride = null) {
    const container = document.getElementById('dashboardClusterExplanationContent');
    if (!container) {
        return;
    }

    // Default explanations matching actual cluster data
    const defaultExplanations = {
        0: {
            characteristics: [
                'Largest group (92% of patients) with mixed symptom presentation',
                'Higher BMI range (mean ~36) indicating metabolic considerations',
                'Mixed respiratory (49%), fever (63%), and pain (40%) complaints',
                'Nearly balanced gender distribution (46% female)'
            ],
            riskFactors: 'Metabolic-inflammatory profile with intermittent symptoms. High BMI combined with recurring fever and respiratory issues requires integrated management approach.',
            recommendations: [
                'Weight management counseling alongside symptom monitoring',
                'Screen for metabolic indicators at follow-up visits',
                'Provide action plans for managing febrile or respiratory flares'
            ]
        },
        1: {
            characteristics: [
                'Small group (4%) presenting for wellness or clearance visits',
                'No reported symptoms (respiratory, fever, pain, allergy, or UTI)',
                'All male with normal BMI range (~24)',
                'First-year students seeking routine health services'
            ],
            riskFactors: 'Low-risk preventive cohort with no acute concerns. These students are seen for wellness checks, clearances, or vitals tracking.',
            recommendations: [
                'Continue routine annual screenings and preventive care',
                'Reinforce healthy lifestyle habits (nutrition, sleep, exercise)',
                'Educate on early symptom reporting for future visits'
            ]
        },
        2: {
            characteristics: [
                '100% report respiratory difficulty at triage',
                '100% present with fever requiring management',
                'All female with lean BMI (~21)',
                'Small but distinct acute care group (4% of patients)'
            ],
            riskFactors: 'Acute respiratory-febrile presentation suggesting infectious respiratory cluster. Requires prompt assessment and monitoring for deterioration.',
            recommendations: [
                'Rapid respiratory assessment (respiratory rate, pulse oximetry)',
                'Initiate fever management and review contagion controls',
                'Escalate if symptoms persist beyond 48 hours or red flags emerge'
            ]
        }
    };

    const defaultExplanation = {
        characteristics: [
            `Patients in Cluster ${cluster} represent a distinct subgroup`,
            'Exhibits unique patterns in health indicators',
            'Requires cluster-specific monitoring strategies'
        ],
        riskFactors: `Risk profile determined by analyzing clinical parameters and patient demographics for this cluster.`,
        recommendations: [
            'Clinical management tailored to cluster characteristics',
            'Regular monitoring and appropriate follow-up',
            'Lifestyle interventions based on individual needs'
        ]
    };

    // Use profile override or default explanations
    const profile = profileOverride || getClusterProfile(cluster);
    let explanation;
    
    if (profile && profile.key_characteristics) {
        explanation = {
            characteristics: profile.key_characteristics,
            riskFactors: profile.risk_summary || profile.risk_level || defaultExplanation.riskFactors,
            recommendations: profile.recommendations || defaultExplanation.recommendations
        };
    } else {
        explanation = defaultExplanations[cluster] || defaultExplanation;
    }

    // Build HTML matching the Clinical Summary style
    const characteristicsHtml = Array.isArray(explanation.characteristics) 
        ? explanation.characteristics.map(item => `<li>${item}</li>`).join('')
        : `<li>${explanation.characteristics}</li>`;
    const recommendationsHtml = Array.isArray(explanation.recommendations)
        ? explanation.recommendations.map(item => `<li>${item}</li>`).join('')
        : `<li>${explanation.recommendations}</li>`;
    
    container.innerHTML = `
        <div class="clinical-content">
            <div class="clinical-summary-box">
                <p>${explanation.riskFactors}</p>
            </div>
            <div class="clinical-recommendations">
                <h4>Characteristics</h4>
                <ul>${characteristicsHtml}</ul>
            </div>
            <div class="clinical-recommendations">
                <h4>Recommendations</h4>
                <ul>${recommendationsHtml}</ul>
            </div>
        </div>
    `;
}

// Resize plot on window resize for better responsiveness
window.addEventListener('resize', function() {
    const plotElement = document.getElementById('clusterPlot');
    if (plotElement && plotElement.data) {
        Plotly.Plots.resize('clusterPlot');
    }
});

// ==============================================
// Chatbot functionality is now in chatbot.js (shared component)
// ==============================================

// Log application info
console.log('%c🔬 ClusterMed - Home Dashboard', 'color: #4F46E5; font-size: 20px; font-weight: bold;');
console.log('%cAPI Endpoint: ' + API_BASE_URL, 'color: #10B981;');
console.log('%cFeatures: Patient Search, Cluster Visualization & AI Chat', 'color: #F59E0B;');

