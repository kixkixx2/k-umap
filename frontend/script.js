// API Configuration
const API_BASE_URL = API_CONFIG.BACKEND_URL;

// Global variables
let allPatientsData = [];
let currentPrediction = null;
let lastSubmittedPatient = null;

// Color scheme for clusters
const CLUSTER_COLORS = [
    '#4F46E5', // Purple
    '#10B981', // Green
    '#F59E0B', // Orange
    '#EF4444', // Red
    '#8B5CF6', // Violet
    '#EC4899', // Pink
    '#14B8A6', // Teal
    '#F97316', // Orange-red
    '#6366F1', // Indigo
    '#A855F7'  // Purple-pink
];

const COORDINATE_CANDIDATES = {
    x: ['x', 'umap_x', 'umapX', 'umap1'],
    y: ['y', 'umap_y', 'umapY', 'umap2']
};

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

// Initialize application
document.addEventListener('DOMContentLoaded', async () => {
    console.log('🚀 ClusterMed initialized');
    
    // Check API health
    await checkAPIHealth();
    
    // Load validation rules
    await formValidator.loadValidationRules();
    
    // Add required indicators and units
    formValidator.addRequiredIndicators();
    formValidator.addUnitIndicators();
    
    // Setup real-time validation for all form fields
    setupFormValidation();
    
    // Load all patients data for visualization
    await loadAllPatientsData();
    
    // Setup event listeners
    setupEventListeners();

    // Setup batch upload center interactions
    initializeBatchUploadCenter();
    
    // Setup BMI calculator if height/weight fields exist
    setupBMICalculator();

    // Initialize improved info tooltips (move to body to avoid clipping)
    initializeInfoTooltips();
    
    // Update stats
    updateStats();
});

// Check if API is running with retry for cold starts
async function checkAPIHealth(retries = 3, delay = 2000) {
    const statusIndicator = document.getElementById('apiStatus');
    const statusText = document.getElementById('apiStatusText');
    
    for (let attempt = 1; attempt <= retries; attempt++) {
        try {
            if (attempt > 1) {
                statusText.textContent = `Connecting... (attempt ${attempt}/${retries})`;
            }
            
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 30000); // 30s timeout for cold starts
            
            const response = await fetch(`${API_BASE_URL}/health`, {
                signal: controller.signal
            });
            clearTimeout(timeoutId);
            
            const data = await response.json();
            
            if (data.status === 'healthy') {
                statusIndicator.className = 'status-indicator online';
                statusText.textContent = 'API Online';
                console.log('✅ API is healthy');
                return true;
            } else {
                throw new Error('API not healthy');
            }
        } catch (error) {
            console.warn(`API health check attempt ${attempt} failed:`, error.message);
            if (attempt < retries) {
                await new Promise(resolve => setTimeout(resolve, delay));
            }
        }
    }
    
    statusIndicator.className = 'status-indicator offline';
    statusText.textContent = 'API Offline';
    console.error('❌ API health check failed after all retries');
    showNotification('API server is not responding. It may be starting up (cold start). Please refresh in 30 seconds.', 'warning');
    return false;
}

// Load all patients data from backend
async function loadAllPatientsData() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/get_all_patients`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        allPatientsData = await response.json();
        console.log(`✅ Loaded ${allPatientsData.length} patients for visualization`);
        
        // Update patient search module if available
        if (typeof patientSearch !== 'undefined' && patientSearch.setPatientData) {
            patientSearch.setPatientData(allPatientsData);
            console.log('✅ Patient search data updated');
        }
        
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
    
    // Update elements only if they exist (not all pages have these elements)
    const totalClustersElem = document.getElementById('totalClusters');
    const totalPatientsElem = document.getElementById('totalPatients');
    const modelStatusElem = document.getElementById('modelStatus');
    
    if (totalClustersElem) {
        totalClustersElem.textContent = uniqueClusters.size || '-';
    }
    
    if (totalPatientsElem) {
        totalPatientsElem.textContent = allPatientsData.length > 0 ? allPatientsData.length.toLocaleString() : '-';
    }
    
    if (modelStatusElem) {
        modelStatusElem.textContent = allPatientsData.length > 0 ? 'Ready' : 'Training Required';
    }
}

// Setup event listeners
function setupEventListeners() {
    // Form submission
    document.getElementById('patientForm').addEventListener('submit', handleFormSubmit);
    
    // Fill sample data button
    document.getElementById('fillSampleBtn').addEventListener('click', fillSampleData);
    
    // Cluster explanation toggle
    const explanationToggle = document.getElementById('clusterExplanationToggle');
    const explanationContent = document.getElementById('clusterExplanationContent');
    
    if (explanationToggle) {
        explanationToggle.addEventListener('click', () => {
            explanationToggle.classList.toggle('active');
            explanationContent.classList.toggle('show');
        });
    }

    const collapseBtn = document.getElementById('resultsCollapseBtn');
    if (collapseBtn) {
        collapseBtn.addEventListener('click', () => {
            const panels = document.querySelectorAll('.results-panel');
            const currentlyExpanded = collapseBtn.getAttribute('aria-expanded') === 'true';
            panels.forEach(panel => {
                panel.open = !currentlyExpanded;
            });
            collapseBtn.setAttribute('aria-expanded', String(!currentlyExpanded));
            const label = collapseBtn.querySelector('.btn-label');
            if (label) {
                label.textContent = currentlyExpanded ? 'Expand All' : 'Collapse All';
            }
        });
    }
}

function initializeBatchUploadCenter() {
    const uploadArea = document.getElementById('fileUploadArea');
    const fileInput = document.getElementById('csvFileInput');
    const removeBtn = document.getElementById('removeFile');
    const startBtn = document.getElementById('startBatchPredict');

    if (!uploadArea || !fileInput || !startBtn) {
        return;
    }

    uploadArea.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    if (removeBtn) {
        removeBtn.addEventListener('click', removeFile);
    }
    startBtn.addEventListener('click', startBatchPrediction);

    // Drag & drop support
    ['dragenter', 'dragover'].forEach(evt => {
        uploadArea.addEventListener(evt, (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragging');
        });
    });
    ['dragleave', 'drop'].forEach(evt => {
        uploadArea.addEventListener(evt, (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragging');
        });
    });
    uploadArea.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0 && files[0].name.endsWith('.csv')) {
            handleFileSelect({ target: { files } });
        } else {
            showNotification('Please upload a CSV file', 'error');
        }
    });

    resetBatchUpload();
}

// Setup real-time form validation
function setupFormValidation() {
    const formFields = [
        'school_id', 'age', 'gender', 'ethnicity', 'insurance_type', 'BMI', 
        'systolic_bp', 'diastolic_bp', 'heart_rate', 'cholesterol_total', 
        'blood_glucose', 'diabetes', 'hypertension', 'heart_disease',
        'smoking_status', 'alcohol_consumption', 'doctor_visits_per_year',
        'num_medications', 'medication_adherence', 'treatment_success_rate'
    ];
    
    formFields.forEach(fieldId => {
        formValidator.setupFieldValidation(fieldId);
    });
}

// ===== Info tooltip manager =====
function initializeInfoTooltips() {
    const icons = document.querySelectorAll('.info-icon');
    if (!icons || icons.length === 0) return;

    icons.forEach(icon => {
        const inner = icon.querySelector('.info-tooltip');
        const text = inner ? inner.innerHTML.trim() : null;
        if (!text) return;

        // Remove original inline tooltip to avoid double elements
        inner.remove();

        // Create a body-level tooltip element
        const bodyTip = document.createElement('div');
        bodyTip.className = 'info-tooltip-body';
        bodyTip.innerHTML = text;
        document.body.appendChild(bodyTip);

        let visible = false;

        function positionTooltip() {
            const rect = icon.getBoundingClientRect();
            const tipRect = bodyTip.getBoundingClientRect();
            const scrollX = window.scrollX || window.pageXOffset;
            const scrollY = window.scrollY || window.pageYOffset;

            // Default place below the icon, centered
            let top = scrollY + rect.bottom + 8;
            let left = scrollX + rect.left + (rect.width / 2) - (tipRect.width / 2);

            // Constrain within viewport with padding
            const padding = 8;
            const minLeft = scrollX + padding;
            const maxLeft = scrollX + document.documentElement.clientWidth - tipRect.width - padding;
            if (left < minLeft) left = minLeft;
            if (left > maxLeft) left = maxLeft;

            bodyTip.style.left = `${Math.round(left)}px`;
            bodyTip.style.top = `${Math.round(top)}px`;
        }

        function show() {
            positionTooltip();
            bodyTip.style.opacity = '1';
            bodyTip.style.pointerEvents = 'auto';
            visible = true;
        }

        function hide() {
            bodyTip.style.opacity = '0';
            bodyTip.style.pointerEvents = 'none';
            visible = false;
        }

        icon.addEventListener('mouseenter', show);
        icon.addEventListener('mouseleave', hide);
        icon.addEventListener('focus', show);
        icon.addEventListener('blur', hide);
        icon.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') hide();
        });

        // Reposition on scroll/resize while visible
        window.addEventListener('scroll', () => { if (visible) positionTooltip(); }, true);
        window.addEventListener('resize', () => { if (visible) positionTooltip(); });
    });
}

// ===== BMI unit/realism warning modal =====
// This exposes `window.maybeShowBMIUnitWarning(bmi, heightCm, weightKg)` so validation logic
// can call it after computing BMI. The modal offers simple conversion helpers.
function maybeShowBMIUnitWarning(bmi, heightCm, weightKg) {
    // thresholds for 'unrealistic' BMI
    const LOW_THRESHOLD = 12.0;
    const HIGH_THRESHOLD = 50.0;

    if (!Number.isFinite(bmi)) return;
    if (bmi >= LOW_THRESHOLD && bmi <= HIGH_THRESHOLD) return; // reasonable

    // Avoid showing multiple times in the same session
    if (document.body.dataset.bmiWarningShown === 'true') return;

    // Create modal
    const modal = document.createElement('div');
    modal.className = 'bmi-warning-modal';
    modal.setAttribute('role', 'dialog');
    modal.setAttribute('aria-modal', 'true');
    modal.style.position = 'fixed';
    modal.style.left = '0';
    modal.style.top = '0';
    modal.style.width = '100%';
    modal.style.height = '100%';
    modal.style.display = 'flex';
    modal.style.alignItems = 'center';
    modal.style.justifyContent = 'center';
    modal.style.zIndex = '10000';
    modal.style.background = 'rgba(2,6,23,0.6)';

    const panel = document.createElement('div');
    panel.style.background = 'linear-gradient(180deg, #0b1220, #071026)';
    panel.style.border = '1px solid rgba(148,163,184,0.12)';
    panel.style.borderRadius = '12px';
    panel.style.padding = '18px';
    panel.style.maxWidth = '560px';
    panel.style.color = '#E5E7EB';
    panel.style.boxShadow = '0 16px 40px rgba(2,6,23,0.7)';

    const title = document.createElement('h3');
    title.textContent = 'Check height & weight units';
    title.style.marginTop = '0';
    panel.appendChild(title);

    const p = document.createElement('p');
    p.style.margin = '8px 0 14px';
    p.style.color = '#C7D2FE';
    p.innerHTML = `Computed BMI is <strong>${Number.isFinite(bmi) ? bmi.toFixed(1) : 'N/A'}</strong>, which looks unrealistic. Please verify that height is in <strong>cm</strong> and weight is in <strong>kg</strong>.`; 
    panel.appendChild(p);

    const hintList = document.createElement('ul');
    hintList.style.margin = '0 0 12px 18px';
    hintList.style.color = '#9CA3AF';
    hintList.innerHTML = `
        <li>If you entered height in meters (e.g. <em>1.75</em>), convert to <em>175 cm</em>.</li>
        <li>If you entered weight in pounds (e.g. <em>150</em>), convert to <em>68.0 kg</em> using lbs → kg.</li>
    `;
    panel.appendChild(hintList);

    const buttons = document.createElement('div');
    buttons.style.display = 'flex';
    buttons.style.gap = '10px';
    buttons.style.justifyContent = 'flex-end';

    const convertHeightBtn = document.createElement('button');
    convertHeightBtn.type = 'button';
    convertHeightBtn.className = 'btn btn-secondary';
    convertHeightBtn.textContent = 'Convert height (m → cm)';

    const convertWeightBtn = document.createElement('button');
    convertWeightBtn.type = 'button';
    convertWeightBtn.className = 'btn btn-secondary';
    convertWeightBtn.textContent = 'Convert weight (lbs → kg)';

    const dismissBtn = document.createElement('button');
    dismissBtn.type = 'button';
    dismissBtn.className = 'btn btn-outline';
    dismissBtn.textContent = 'Dismiss';

    buttons.appendChild(convertHeightBtn);
    buttons.appendChild(convertWeightBtn);
    buttons.appendChild(dismissBtn);
    panel.appendChild(buttons);

    modal.appendChild(panel);
    document.body.appendChild(modal);

    // Focus management
    dismissBtn.focus();

    // Handlers
    convertHeightBtn.addEventListener('click', () => {
        const heightField = document.getElementById('height_cm');
        if (!heightField) return;
        const raw = parseFloat(heightField.value);
        // If height seems like meters (e.g., 1.7 or < 10), convert by *100
        let newVal;
        if (!isNaN(raw)) {
            if (raw > 0 && raw < 10) {
                newVal = +(raw * 100).toFixed(1);
            } else if (raw >= 10 && raw <= 250) {
                // It's already in cm-ish range — but still multiply if user insists
                newVal = raw;
            } else {
                newVal = raw;
            }
            heightField.value = newVal;
            heightField.dispatchEvent(new Event('input', { bubbles: true }));
        }
        closeModal();
    });

    convertWeightBtn.addEventListener('click', () => {
        const weightField = document.getElementById('weight_kg');
        if (!weightField) return;
        const raw = parseFloat(weightField.value);
        if (!isNaN(raw)) {
            const converted = +(raw / 2.20462).toFixed(1);
            weightField.value = converted;
            weightField.dispatchEvent(new Event('input', { bubbles: true }));
        }
        closeModal();
    });

    dismissBtn.addEventListener('click', () => {
        closeModal();
    });

    // Close when clicking outside or pressing Escape
    modal.addEventListener('click', (e) => {
        if (e.target === modal) closeModal();
    });
    function onKey(e) {
        if (e.key === 'Escape') closeModal();
    }
    window.addEventListener('keydown', onKey);

    function closeModal() {
        document.body.dataset.bmiWarningShown = 'true';
        window.removeEventListener('keydown', onKey);
        if (modal && modal.parentNode) modal.parentNode.removeChild(modal);
    }
}

// Expose globally for validation.js to call
window.maybeShowBMIUnitWarning = maybeShowBMIUnitWarning;

// Check if a patient ID already exists in the system
function findExistingPatient(patientId) {
    if (!patientId || !allPatientsData || allPatientsData.length === 0) {
        return null;
    }
    const searchId = String(patientId).trim().toLowerCase();
    return allPatientsData.find(patient => {
        const pid = String(patient.patient_id || '').trim().toLowerCase();
        const sid = String(patient.Student_No || '').trim().toLowerCase();
        return pid === searchId || sid === searchId;
    });
}

// Display existing patient data (as if searched)
function displayExistingPatient(patient) {
    // Transform existing patient data to match the format expected by displayResults
    const confidence = patient.confidence || patient.cluster_confidence || 0;
    const patientId = patient.patient_id || patient.Student_No || '-';
    
    // Get cluster profile for recommendations
    const profile = patient.cluster_profile || {};
    const recommendations = profile.recommendations || [];
    const summary = profile.care_focus || profile.risk_summary || getClusterCareRecommendation(patient.cluster);
    
    // Build a result object that mimics a fresh prediction response
    const resultData = {
        patient_id: patientId,
        cluster: patient.cluster,
        confidence: confidence,
        umap_coordinates: {
            x: patient.x || patient.umap_x || 0,
            y: patient.y || patient.umap_y || 0
        },
        // Use patient data as the snapshot
        patient_snapshot: {
            first_name: patient.first_name || patient.First_Name || '',
            last_name: patient.last_name || patient.Last_Name || '',
            year_level: patient.year_level,
            age: patient.age || patient.age_years,
            age_years: patient.age_years,
            gender: patient.gender || (patient.is_female === 1 ? 'Female' : patient.is_female === 0 ? 'Male' : null),
            is_female: patient.is_female,
            height_cm: patient.height_cm,
            weight_kg: patient.weight_kg,
            bmi: patient.bmi || patient.BMI,
            BMI: patient.BMI || patient.bmi,
            has_respiratory_issue: patient.has_respiratory_issue,
            has_pain: patient.has_pain,
            has_fever: patient.has_fever,
            has_allergy: patient.has_allergy,
            is_uti: patient.is_uti
        },
        // Build clinical intelligence matching API format (summary + recommendations as strings)
        clinical_intelligence: {
            summary: summary,
            alerts: [],  // No alerts for existing patients - use recommendations instead
            recommendations: recommendations
        },
        // Include cluster_profile for Cluster Details section
        cluster_profile: profile
    };
    
    // Use the same display function as a new prediction
    displayResults(resultData);
}

// Helper function to get default care recommendation for cluster
function getClusterCareRecommendation(cluster) {
    const recommendations = {
        0: "Routine preventive care recommended. Focus on maintaining current health status.",
        1: "Schedule follow-up and reinforce lifestyle coaching. Monitor for any changes.",
        2: "Consider escalated monitoring and multidisciplinary review. Prioritize symptom management."
    };
    return recommendations[cluster] || "Review patient context for personalized guidance.";
}


// Handle form submission
async function handleFormSubmit(event) {
    event.preventDefault();
    
    const predictBtn = document.getElementById('predictBtn');
    const originalText = predictBtn.innerHTML;
    
    // Collect form data
    const formData = new FormData(event.target);
    
    // Validate form
    const validationResult = formValidator.validateForm(formData);
    
    if (!validationResult.isValid) {
        // Show validation errors
        for (const [field, error] of Object.entries(validationResult.errors)) {
            formValidator.showFieldError(field, error);
        }
        const firstField = Object.keys(validationResult.errors)[0];
        if (firstField) {
            const firstInput = document.getElementById(firstField);
            if (firstInput) {
                firstInput.focus({ preventScroll: true });
                firstInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        }
        showNotification('Please fix the validation errors before submitting', 'error');
        return;
    }
    
    // Show loading state
    predictBtn.disabled = true;
    predictBtn.innerHTML = '<span class="btn-icon">⏳</span> Predicting...';
    
    // Convert form data to JSON
    const patientData = {};
    const numericFields = new Set([
        'age', 'BMI', 'bmi', 'year_level', 'age_years',
        'systolic_bp', 'diastolic_bp', 'heart_rate', 
        'cholesterol_total', 'blood_glucose', 'diabetes', 'hypertension', 
        'heart_disease', 'doctor_visits_per_year', 'num_medications', 
        'medication_adherence', 'treatment_success_rate', 'height_cm', 'weight_kg'
    ]);
    const booleanFlags = new Set(['has_respiratory_issue','has_pain','has_fever','has_allergy','is_uti']);
    
    for (let [key, value] of formData.entries()) {
        // Skip empty values
        if (value === '' || value === null || value === undefined) {
            continue;
        }
        
        if (numericFields.has(key)) {
            const parsed = parseFloat(value);
            // Only include if it's a valid number (not NaN)
            if (!isNaN(parsed) && isFinite(parsed)) {
                patientData[key] = parsed;
            }
        } else if (booleanFlags.has(key)) {
            // Convert checkbox flags to numeric 0/1
            const numeric = (value === '1' || value === 'on') ? 1 : 0;
            patientData[key] = numeric;
        } else {
            patientData[key] = value;
        }
    }
    
    // Ensure boolean flags are present as 0/1 even if unchecked
    ['has_respiratory_issue','has_pain','has_fever','has_allergy','is_uti'].forEach(flag => {
        if (!(flag in patientData)) {
            patientData[flag] = 0;
        }
    });
    
    // If height in cm provided, also include height in meters for backward compatibility
    if (patientData.height_cm !== undefined && !isNaN(patientData.height_cm)) {
        const hm = parseFloat(patientData.height_cm) / 100.0;
        patientData.height_m = parseFloat(hm.toFixed(3));
    }
    
    // Final sanitization: remove any NaN or undefined values
    Object.keys(patientData).forEach(key => {
        const val = patientData[key];
        if (val === undefined || val === null || (typeof val === 'number' && (isNaN(val) || !isFinite(val)))) {
            delete patientData[key];
        }
    });

    lastSubmittedPatient = { ...patientData };
    
    // Check if patient ID already exists
    const inputPatientId = patientData.school_id || patientData.patient_id || patientData.Student_No;
    if (inputPatientId) {
        const existingPatient = findExistingPatient(inputPatientId);
        if (existingPatient) {
            console.log('⚠️ Patient ID already exists:', inputPatientId);
            showNotification(`Patient ID "${inputPatientId}" already exists in the system. Showing existing record.`, 'warning');
            displayExistingPatient(existingPatient);
            
            // Reset button
            predictBtn.disabled = false;
            predictBtn.innerHTML = originalText;
            return;
        }
    }
    
    console.log('📤 Sending prediction request:', patientData);
    
    try {
        // Call prediction API
        const predictUrl = (typeof getApiUrl === 'function') ? getApiUrl('PREDICT') : `${API_BASE_URL}/predict`;
        console.log('%c[Predict] Calling API:', 'color: #93C5FD;', predictUrl);
        const response = await fetch(predictUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(patientData)
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            // Handle validation errors from backend
            if (result.validation_errors) {
                showNotification('Validation failed: ' + result.validation_errors.join(', '), 'error');
            } else {
                throw new Error(result.error || `HTTP error! status: ${response.status}`);
            }
            return;
        }
        
        console.log('✅ Prediction result:', result);
        
        if (result.success) {
            currentPrediction = result;
            displayResults(result);
            
            // Reload all patients data to include the new patient
            await loadAllPatientsData();
            
            showNotification(result.message || `Patient saved with ID ${result.patient_id}`, 'success');
        } else {
            throw new Error(result.error || 'Prediction failed');
        }
        
    } catch (error) {
        console.error('❌ Prediction error:', error);
        let userMessage = `Prediction failed: ${error.message}`;
        // Common fetch network error (e.g., server not running / CORS issue)
        if (error instanceof TypeError) {
            userMessage += ' — Unable to reach prediction API. Is the backend running? Check server and CORS settings.';
            const hint = `API URL attempted: ${ (typeof getApiUrl === 'function') ? getApiUrl('PREDICT') : (API_BASE_URL + '/predict') }`;
            console.warn('[Predict] Hint:', hint);
        }
        showNotification(userMessage, 'error');
    } finally {
        // Reset button
        predictBtn.disabled = false;
        predictBtn.innerHTML = originalText;
    }
}

// Display prediction results
function displayResults(result) {
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // School ID
    document.getElementById('assignedPatientId').textContent = result.patient_id;
    const patientIdSubtext = document.getElementById('patientIdSubtext');
    if (patientIdSubtext) {
        patientIdSubtext.textContent = result.patient_id.startsWith('SIM-') ? 'Auto-generated' : 'Student Identifier';
    }

    // Cluster
    const clusterElement = document.getElementById('predictedCluster');
    clusterElement.textContent = `Cluster ${result.cluster}`;
    clusterElement.style.color = CLUSTER_COLORS[result.cluster % CLUSTER_COLORS.length];

    const clusterNarrative = getClusterDescription(result.cluster);
    const clusterDescriptionEl = document.getElementById('clusterDescription');
    if (clusterDescriptionEl) {
        clusterDescriptionEl.textContent = clusterNarrative;
    }

    // Confidence
    if (result.confidence !== undefined) {
        const confidencePercent = (result.confidence * 100).toFixed(1);
        const confidenceElement = document.getElementById('confidenceScore');
        if (confidenceElement) {
            confidenceElement.textContent = `${confidencePercent}%`;
        }

        const confidenceBar = document.getElementById('confidenceBar');
        if (confidenceBar) {
            confidenceBar.style.width = `${confidencePercent}%`;
            let state = 'low';
            if (result.confidence >= 0.85) {
                state = 'high';
            } else if (result.confidence >= 0.65) {
                state = 'medium';
            }
            confidenceBar.dataset.state = state;
        }
    }

    // Get patient data from snapshot or last submitted
    const patientSnapshot = result.patient_snapshot || lastSubmittedPatient || {};

    // Populate Demographics Section
    populateDemographics(patientSnapshot);

    // Populate Diagnosis Flags Section
    populateDiagnosisFlags(patientSnapshot);

    // Clinical Intelligence (Recommendations) - always update
    const clinicalContainer = document.getElementById('clinicalAlertsContainer');
    if (clinicalContainer) {
        if (result.clinical_intelligence && typeof clinicalAlertsDisplay !== 'undefined') {
            console.log('📋 Using API clinical_intelligence:', result.clinical_intelligence);
            clinicalAlertsDisplay.displayClinicalIntelligence(result.clinical_intelligence);
        } else {
            // Generate fallback clinical summary based on cluster
            console.log('⚠️ No API clinical_intelligence, using fallback for cluster:', result.cluster);
            const fallbackIntelligence = generateFallbackClinicalIntelligence(result.cluster, patientSnapshot);
            if (typeof clinicalAlertsDisplay !== 'undefined') {
                clinicalAlertsDisplay.displayClinicalIntelligence(fallbackIntelligence);
            } else {
                // Direct DOM update if clinicalAlertsDisplay not available
                clinicalContainer.innerHTML = buildClinicalSummaryHTML(fallbackIntelligence);
            }
        }
        const summaryPanel = document.getElementById('clinicalSummaryPanel');
        if (summaryPanel) {
            summaryPanel.open = true;
        }
    }

    // Cluster Explanation - use API cluster_profile if available
    populateClusterExplanation(result.cluster, result.cluster_profile);

    // Scatter Plot
    createScatterPlot(result);
}

// Generate fallback clinical intelligence when API doesn't provide it
function generateFallbackClinicalIntelligence(cluster, patientData) {
    const clusterData = {
        0: {
            summary: 'Patient assigned to the mixed symptom group with metabolic considerations. This cluster includes the majority of students presenting with varying combinations of respiratory, fever, and pain symptoms alongside elevated BMI.',
            recommendations: [
                'Weight management counseling alongside symptom monitoring',
                'Screen for metabolic indicators at follow-up visits',
                'Provide action plans for managing febrile or respiratory flares'
            ],
            alerts: []
        },
        1: {
            summary: 'Patient assigned to the wellness/clearance group. This small cohort presents with no acute symptoms and visits primarily for routine health services or clearances.',
            recommendations: [
                'Continue routine annual screenings and preventive care',
                'Reinforce healthy lifestyle habits (nutrition, sleep, exercise)',
                'Educate on early symptom reporting for future visits'
            ],
            alerts: []
        },
        2: {
            summary: 'Patient assigned to the acute respiratory-febrile group. This cluster is characterized by concurrent respiratory difficulty and fever, requiring prompt assessment.',
            recommendations: [
                'Rapid respiratory assessment (respiratory rate, pulse oximetry)',
                'Initiate fever management and review contagion controls',
                'Escalate if symptoms persist beyond 48 hours or red flags emerge'
            ],
            alerts: ['Concurrent respiratory and fever symptoms detected']
        }
    };

    return clusterData[cluster] || {
        summary: `Patient assigned to Cluster ${cluster}. Review individual characteristics for personalized guidance.`,
        recommendations: ['Clinical management tailored to cluster characteristics', 'Regular monitoring and appropriate follow-up'],
        alerts: []
    };
}

// Build clinical summary HTML directly
function buildClinicalSummaryHTML(clinicalData) {
    const { alerts, recommendations, summary } = clinicalData;
    
    let html = `<div class="clinical-content">`;
    
    if (summary) {
        html += `<div class="clinical-summary-box"><p>${summary}</p></div>`;
    }

    if (recommendations && recommendations.length > 0) {
        html += `<div class="clinical-recommendations"><h4>Recommendations</h4><ul>${recommendations.map(rec => `<li>${rec}</li>`).join('')}</ul></div>`;
    }

    if (alerts && alerts.length > 0) {
        html += `<div class="clinical-alerts-list"><h4>Alerts</h4><ul>${alerts.map(alert => `<li>${alert}</li>`).join('')}</ul></div>`;
    }

    html += `</div>`;
    return html;
}

// Populate Demographics section
function populateDemographics(data) {
    // Name - Hidden for privacy
    const nameEl = document.getElementById('resultName');
    if (nameEl) nameEl.textContent = '(Hidden for Privacy)';

    // Year Level
    const yearLevel = data.year_level || '-';
    const yearLevelEl = document.getElementById('resultYearLevel');
    if (yearLevelEl) {
        if (yearLevel !== '-') {
            const yearLabels = { 1: '1st Year', 2: '2nd Year', 3: '3rd Year', 4: '4th Year', 5: '5th Year' };
            yearLevelEl.textContent = yearLabels[yearLevel] || `Year ${yearLevel}`;
        } else {
            yearLevelEl.textContent = '-';
        }
    }

    // Age
    const age = data.age || data.age_years || '-';
    const ageEl = document.getElementById('resultAge');
    if (ageEl) ageEl.textContent = age !== '-' ? `${age} yrs` : '-';

    // Gender
    const gender = data.gender || (data.is_female === 1 ? 'Female' : data.is_female === 0 ? 'Male' : '-');
    const genderEl = document.getElementById('resultGender');
    if (genderEl) genderEl.textContent = gender;

    // Height
    const height = data.height_cm || '-';
    const heightEl = document.getElementById('resultHeight');
    if (heightEl) heightEl.textContent = height !== '-' ? `${height} cm` : '-';

    // Weight
    const weight = data.weight_kg || '-';
    const weightEl = document.getElementById('resultWeight');
    if (weightEl) weightEl.textContent = weight !== '-' ? `${weight} kg` : '-';

    // BMI
    const bmi = data.BMI || data.bmi || '-';
    const bmiEl = document.getElementById('resultBMI');
    if (bmiEl) bmiEl.textContent = bmi !== '-' ? parseFloat(bmi).toFixed(1) : '-';
}

// Populate Diagnosis Flags section
function populateDiagnosisFlags(data) {
    const flags = [
        { id: 'diagRespiratory', key: 'has_respiratory_issue' },
        { id: 'diagPain', key: 'has_pain' },
        { id: 'diagFever', key: 'has_fever' },
        { id: 'diagAllergy', key: 'has_allergy' },
        { id: 'diagUTI', key: 'is_uti' }
    ];

    flags.forEach(({ id, key }) => {
        const el = document.getElementById(id);
        if (el) {
            const statusEl = el.querySelector('.diagnosis-status');
            const value = data[key];
            const isPositive = value === 1 || value === '1' || value === true;
            
            // Update status text and colors (Green = Yes, Red = No)
            if (statusEl) {
                statusEl.textContent = isPositive ? 'Yes' : 'No';
                statusEl.style.background = isPositive ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)';
                statusEl.style.color = isPositive ? '#10B981' : '#EF4444';
            }
            
            // Update border color
            el.style.borderLeftColor = isPositive ? '#10B981' : '#EF4444';
        }
    });
}

// Display clinical insights
function displayInsights(insights) {
    const insightsContainer = document.getElementById('clinicalInsights');
    if (!insightsContainer) return;
    
    insightsContainer.innerHTML = '';
    
    if (insights.bmi_category) {
        addInsightBadge(insightsContainer, 'BMI', insights.bmi_category);
    }
    
    if (insights.bp_category) {
        addInsightBadge(insightsContainer, 'Blood Pressure', insights.bp_category);
    }
    
    if (insights.diabetes_risk) {
        addInsightBadge(insightsContainer, 'Diabetes Risk', insights.diabetes_risk);
    }
}

// Add insight badge
function addInsightBadge(container, label, value) {
    const badge = document.createElement('div');
    badge.className = 'insight-badge';
    badge.innerHTML = `
        <span class="insight-label">${label}:</span>
        <span class="insight-value">${value}</span>
    `;
    container.appendChild(badge);
}

function updateMetricMeters(patientSnapshot = {}, insights = {}) {
    const metricConfigs = [
        {
            key: 'BMI',
            min: 10,
            max: 60,
            unit: 'kg/m^2',
            caption: insights.bmi_category || 'Healthy range: 18.5 - 24.9'
        },
        {
            key: 'systolic_bp',
            min: 70,
            max: 200,
            unit: 'mmHg',
            caption: insights.bp_category || 'Target: < 120/80'
        },
        {
            key: 'blood_glucose',
            min: 50,
            max: 400,
            unit: 'mg/dL',
            caption: insights.diabetes_risk || 'Fasting target: 70 - 99'
        }
    ];

    metricConfigs.forEach(({ key, min, max, unit, caption }) => {
        const value = Number(patientSnapshot[key]);
        const fill = document.getElementById(`meterFill-${key}`);
        const valueLabel = document.getElementById(`meterValue-${key}`);
        const captionLabel = document.getElementById(`meterCaption-${key}`);

        if (!fill || !valueLabel || !captionLabel) {
            return;
        }

        if (Number.isFinite(value)) {
            const percent = Math.min(100, Math.max(0, ((value - min) / (max - min)) * 100));
            fill.style.width = `${percent}%`;
            let state = 'good';
            if (percent >= 75) {
                state = 'alert';
            } else if (percent >= 55) {
                state = 'caution';
            }
            fill.dataset.state = state;
            valueLabel.textContent = `${value.toFixed(1)} ${unit}`;
            captionLabel.textContent = caption;
        } else {
            fill.style.width = '0%';
            fill.dataset.state = 'idle';
            valueLabel.textContent = '-';
            captionLabel.textContent = 'Awaiting data';
        }
    });
}

function describeConfidence(score = 0) {
    if (score >= 0.9) return 'Very high certainty';
    if (score >= 0.75) return 'High certainty';
    if (score >= 0.6) return 'Moderate certainty';
    return 'Needs clinician review';
}

// Get cluster description
function getClusterDescription(cluster) {
    const descriptions = {
        0: 'Mixed symptom group with metabolic considerations',
        1: 'Wellness/clearance group - no acute symptoms',
        2: 'Acute respiratory-febrile group'
    };
    
    return descriptions[cluster] || `Cluster ${cluster}`;
}

// Populate detailed cluster explanation - uses API cluster_profile when available
function populateClusterExplanation(cluster, clusterProfile = null) {
    // If API provides cluster_profile, use it directly
    if (clusterProfile && clusterProfile.key_characteristics) {
        console.log('📊 Using API cluster_profile for cluster details:', clusterProfile);
        const container = document.getElementById('clusterExplanationContent');
        if (container) {
            const characteristics = clusterProfile.key_characteristics || [];
            const recommendations = clusterProfile.recommendations || [];
            const riskSummary = clusterProfile.risk_summary || clusterProfile.care_focus || `Cluster ${cluster} profile`;
            
            const characteristicsHtml = characteristics.map(item => `<li>${item}</li>`).join('');
            const recommendationsHtml = recommendations.map(item => `<li>${item}</li>`).join('');
            
            container.innerHTML = `
                <div class="clinical-content">
                    <div class="clinical-summary-box">
                        <p>${riskSummary}</p>
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
        return;
    }
    
    // Fallback to hardcoded explanations if API doesn't provide cluster_profile
    console.log('⚠️ No API cluster_profile, using fallback for cluster:', cluster);
    const explanations = {
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
    
    const explanation = explanations[cluster] || defaultExplanation;
    
    // Build HTML matching the Clinical Summary style
    const container = document.getElementById('clusterExplanationContent');
    if (container) {
        const characteristicsHtml = explanation.characteristics.map(item => `<li>${item}</li>`).join('');
        const recommendationsHtml = explanation.recommendations.map(item => `<li>${item}</li>`).join('');
        
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
}

// Create scatter plot with Plotly
function createScatterPlot(prediction) {
    if (allPatientsData.length === 0) {
        document.getElementById('scatterPlot').innerHTML = 
            '<div class="plot-error">⚠️ Visualization data not available. Please run train.py first.</div>';
        return;
    }
    
    // Store prediction for feature selector
    currentPrediction = prediction;
    
    // Get selected feature
    const featureSelector = document.getElementById('featureSelector');
    const selectedFeature = featureSelector ? featureSelector.value : 'cluster';
    
    // Update plot based on selected feature
    updateScatterPlot(prediction, selectedFeature);
}

function updateScatterPlot(prediction, selectedFeature) {
    const traces = [];
    
    if (selectedFeature === 'cluster') {
        // Group patients by cluster (original behavior)
        const clusterGroups = {};
        allPatientsData.forEach(patient => {
            if (!clusterGroups[patient.cluster]) {
                clusterGroups[patient.cluster] = { x: [], y: [] };
            }
            clusterGroups[patient.cluster].x.push(getCoordinateValue(patient, 'x'));
            clusterGroups[patient.cluster].y.push(getCoordinateValue(patient, 'y'));
        });
        
        // Create traces for each cluster
        Object.keys(clusterGroups).sort((a, b) => parseInt(a) - parseInt(b)).forEach(cluster => {
            const clusterNum = parseInt(cluster);
            traces.push({
                x: clusterGroups[cluster].x,
                y: clusterGroups[cluster].y,
                mode: 'markers',
                type: 'scatter',
                name: `Cluster ${cluster}`,
                marker: {
                    size: 8,
                    color: CLUSTER_COLORS[clusterNum % CLUSTER_COLORS.length],
                    opacity: 0.6,
                    line: {
                        color: 'white',
                        width: 0.5
                    }
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
        const hoverText = allPatientsData.map(p => {
            const featVal = p.features ? p.features[selectedFeature] : (p[selectedFeature] !== undefined ? p[selectedFeature] : 'N/A');
            // For boolean flags, give readable text
            if (selectedFeature === 'has_respiratory_issue') {
                const yesNo = featVal ? 'Yes' : 'No';
                const explanation = featVal ? 'Respiratory issues include asthma, wheeze, shortness of breath.' : '';
                return `Cluster: ${p.cluster}<br>Respiratory Issue: ${yesNo}${explanation ? ' — ' + explanation : ''}`;
            }
            if (['has_pain','has_fever','has_allergy','is_uti'].includes(selectedFeature)) {
                const yesNo = featVal ? 'Yes' : 'No';
                return `Cluster: ${p.cluster}<br>${selectedFeature.replace(/_/g,' ')}: ${yesNo}`;
            }
            return `Cluster: ${p.cluster}<br>${selectedFeature}: ${featVal !== null && featVal !== undefined ? featVal : 'N/A'}`;
        });
        
        // Feature-specific colorbar settings
        const featureColorSettings = {
            bmi: { cmin: 15, cmax: 50, colorscale: 'Viridis', title: 'BMI (kg/m²)' },
            age_years: { cmin: 16, cmax: 25, colorscale: 'Blues', title: 'Age (years)' },
            year_level: { cmin: 1, cmax: 4, colorscale: 'Plasma', title: 'Year Level' }
        };
        const colorSettings = featureColorSettings[selectedFeature] || {};
        const featureTitle = colorSettings.title || selectedFeature.replace(/_/g, ' ');
        
        const markerConfig = {
            size: 8,
            color: featureValues,
            colorscale: colorSettings.colorscale || 'Viridis',
            showscale: true,
            opacity: 0.7,
            colorbar: {
                title: featureTitle,
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
        
        // Apply min/max limits if defined
        if (typeof colorSettings.cmin === 'number') {
            markerConfig.cmin = colorSettings.cmin;
            markerConfig.cauto = false;
        }
        if (typeof colorSettings.cmax === 'number') {
            markerConfig.cmax = colorSettings.cmax;
            markerConfig.cauto = false;
        }
        
        traces.push({
            x: x,
            y: y,
            mode: 'markers',
            type: 'scatter',
            name: 'All Patients',
            text: hoverText,
            hovertemplate: '%{text}<extra></extra>',
            marker: markerConfig,
            showlegend: false
        });
    }
    
    // Add the new patient as a highlighted point
    traces.push({
        x: [prediction.umap_coordinates.x],
        y: [prediction.umap_coordinates.y],
        mode: 'markers',
        type: 'scatter',
        name: 'Your Patient',
        marker: {
            size: 20,
            color: 'rgba(239, 68, 68, 0.8)',  // Red with 80% opacity
            symbol: 'star',
            line: {
                color: 'white',
                width: 2
            }
        },
        showlegend: true
    });
    
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
        showlegend: selectedFeature === 'cluster' || true,
        legend: {
            x: 1.02,
            y: 1,
            bgcolor: '#1F2937',
            bordercolor: '#374151',
            borderwidth: 1
        }
    };
    
    // Plot configuration
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['lasso2d', 'select2d']
    };
    
    // Create the plot
    Plotly.newPlot('scatterPlot', traces, layout, config);
}

// Add event listener for feature selector
document.addEventListener('DOMContentLoaded', function() {
    const featureSelector = document.getElementById('featureSelector');
    if (featureSelector) {
        featureSelector.addEventListener('change', function() {
            if (currentPrediction) {
                updateScatterPlot(currentPrediction, this.value);
            }
        });
    }
});

// Fill sample data for testing - matches ISU student health dataset
function fillSampleData() {
    // School ID (required, format: XX-XXXX)
    const schoolIdEl = document.getElementById('school_id'); 
    if (schoolIdEl) schoolIdEl.value = '22-0641';

    // Names (optional)
    const last = document.getElementById('last_name'); if (last) last.value = 'Dela Cruz';
    const first = document.getElementById('first_name'); if (first) first.value = 'Maria';

    // Demographics (required for model)
    const ageEl = document.getElementById('age'); if (ageEl) ageEl.value = 19;
    const yearLevelEl = document.getElementById('year_level'); if (yearLevelEl) yearLevelEl.value = '2';
    const genderEl = document.getElementById('gender'); if (genderEl) genderEl.value = 'Female';

    // Height in centimeters and weight in kg (used to compute BMI)
    const heightField = document.getElementById('height_cm');
    const weightField = document.getElementById('weight_kg');
    if (heightField) heightField.value = 160;
    if (weightField) weightField.value = 55.0;

    // Ensure BMI is computed/displayed
    const bmiField = document.getElementById('BMI');
    if (heightField && weightField && bmiField) {
        const h_cm = parseFloat(heightField.value);
        const h = !isNaN(h_cm) && h_cm > 0 ? (h_cm / 100.0) : NaN;
        const w = parseFloat(weightField.value);
        if (!isNaN(h) && h > 0 && !isNaN(w)) {
            const bmi = w / (h * h);
            bmiField.value = bmi.toFixed(1);
        }
    }

    // Diagnostic flags (the core features for clustering)
    const flags = ['has_respiratory_issue', 'has_pain', 'has_fever', 'has_allergy', 'is_uti'];
    flags.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.checked = false;
    });
    
    // Set sample diagnostic flags - simulating a student with respiratory issue and fever
    const respEl = document.getElementById('has_respiratory_issue'); if (respEl) respEl.checked = true;
    const feverEl = document.getElementById('has_fever'); if (feverEl) feverEl.checked = true;

    // Optional legacy fields (if they exist in the form)
    const ethnicityEl = document.getElementById('ethnicity'); if (ethnicityEl) ethnicityEl.value = 'Asian';
    const insuranceEl = document.getElementById('insurance_type'); if (insuranceEl) insuranceEl.value = 'Private';
    const sBP = document.getElementById('systolic_bp'); if (sBP) sBP.value = 120;
    const dBP = document.getElementById('diastolic_bp'); if (dBP) dBP.value = 80;
    const hr = document.getElementById('heart_rate'); if (hr) hr.value = 72;
    const chol = document.getElementById('cholesterol_total'); if (chol) chol.value = 180;
    const bg = document.getElementById('blood_glucose'); if (bg) bg.value = 95;
    const diabetesEl = document.getElementById('diabetes'); if (diabetesEl) diabetesEl.value = '0';
    const hyperEl = document.getElementById('hypertension'); if (hyperEl) hyperEl.value = '0';
    const heartEl = document.getElementById('heart_disease'); if (heartEl) heartEl.value = '0';
    const smokeEl = document.getElementById('smoking_status'); if (smokeEl) smokeEl.value = 'Never';
    const alcEl = document.getElementById('alcohol_consumption'); if (alcEl) alcEl.value = 'None';
    const visitsEl = document.getElementById('doctor_visits_per_year'); if (visitsEl) visitsEl.value = 2;
    const medsEl = document.getElementById('num_medications'); if (medsEl) medsEl.value = 0;
    const adherenceEl = document.getElementById('medication_adherence'); if (adherenceEl) adherenceEl.value = 1.0;
    const successEl = document.getElementById('treatment_success_rate'); if (successEl) successEl.value = 0.95;

    showNotification('Sample data filled! Ready to predict cluster.', 'info');
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

// Log application info
console.log('%c🔬 ClusterMed - Patient Clustering System', 'color: #4F46E5; font-size: 20px; font-weight: bold;');
console.log('%cAPI Endpoint: ' + API_BASE_URL, 'color: #10B981;');
console.log('%cMake sure your Flask server is running on port 5000', 'color: #F59E0B;');

// ===== BATCH PREDICT FUNCTIONALITY =====

let selectedFile = null;
let batchResults = null;
let activeBatchPatientId = null;

function resetBatchUpload() {
    selectedFile = null;
    batchResults = null;
    activeBatchPatientId = null;
    const fileInput = document.getElementById('csvFileInput');
    if (fileInput) fileInput.value = '';
    const fileInfo = document.getElementById('fileInfo');
    if (fileInfo) fileInfo.style.display = 'none';
    const progress = document.getElementById('batchProgress');
    if (progress) progress.style.display = 'none';
    const results = document.getElementById('batchResults');
    if (results) results.style.display = 'none';
    const startBtn = document.getElementById('startBatchPredict');
    if (startBtn) startBtn.disabled = true;
    const progressBar = document.getElementById('progressBar');
    if (progressBar) progressBar.style.width = '0%';
    const progressText = document.getElementById('progressText');
    if (progressText) progressText.textContent = '0%';
    const progressMessage = document.getElementById('progressMessage');
    if (progressMessage) progressMessage.textContent = 'Waiting for upload...';
    const tableBody = document.getElementById('batchResultsBody');
    if (tableBody) tableBody.innerHTML = '';
    const emptyState = document.getElementById('batchResultsEmpty');
    if (emptyState) emptyState.style.display = 'block';
    const resultsCount = document.getElementById('batchResultsCount');
    if (resultsCount) resultsCount.textContent = '';
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    if (!file.name.endsWith('.csv')) {
        showNotification('Please select a CSV file', 'error');
        return;
    }
    
    selectedFile = file;
    
    // Show file info
    document.getElementById('fileName').textContent = file.name;
    document.getElementById('fileSize').textContent = formatFileSize(file.size);
    document.getElementById('fileInfo').style.display = 'flex';
    document.getElementById('startBatchPredict').disabled = false;
    const progress = document.getElementById('batchProgress');
    if (progress) progress.style.display = 'none';
    const results = document.getElementById('batchResults');
    if (results) results.style.display = 'none';
}

function removeFile() {
    resetBatchUpload();
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
}

async function startBatchPrediction() {
    if (!selectedFile) {
        showNotification('Please select a file first', 'error');
        return;
    }
    
    // Show progress
    const progressContainer = document.getElementById('batchProgress');
    const resultsContainer = document.getElementById('batchResults');
    const startButton = document.getElementById('startBatchPredict');

    if (progressContainer) progressContainer.style.display = 'block';
    if (resultsContainer) resultsContainer.style.display = 'none';
    if (startButton) startButton.disabled = true;
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    try {
        updateProgress(10, 'Uploading file...');
        
        const response = await fetch(`${API_BASE_URL}/api/batch_predict`, {
            method: 'POST',
            body: formData
        });
        
        updateProgress(50, 'Processing patients...');

        const rawBody = await response.text();
        let data = null;
        if (rawBody) {
            try {
                data = JSON.parse(rawBody);
            } catch (parseError) {
                console.error('Failed to parse batch response JSON:', parseError, rawBody);
            }
        }

        if (!response.ok) {
            const statusInfo = `HTTP ${response.status}${response.statusText ? ' ' + response.statusText : ''}`;
            const reason = getBatchFailureReason(data);
            const errorMessage = reason
                ? `${statusInfo}: ${reason}`
                : `${statusInfo}: ${rawBody || 'Batch prediction failed'}`;
            if (progressContainer) progressContainer.style.display = 'none';
            if (startButton) startButton.disabled = false;
            showNotification(errorMessage, 'error');
            return;
        }

        if (!data) {
            const statusInfo = `HTTP ${response.status}${response.statusText ? ' ' + response.statusText : ''}`;
            throw new Error(`${statusInfo}: Batch prediction returned an empty or invalid JSON response.`);
        }

        updateProgress(100, 'Complete!');
        
        // Store results
        batchResults = data;

        // No successful predictions? Show reason and exit
        if (!data.successful || !data.results || data.results.length === 0) {
            const reason = getBatchFailureReason(data);
            if (progressContainer) progressContainer.style.display = 'none';
            if (startButton) startButton.disabled = false;
            showNotification(reason, 'error');
            return;
        }
        
        // Show batch summary
        setTimeout(() => {
            displayBatchResults(data);
        }, 500);
        
        // Surface the first successful prediction using the single-patient view
        showBatchPredictionResult(data.results[0]);
        
        // Refresh patient search data so batch patients are searchable
        await loadAllPatientsData();
        
        const successMessage = data.failed && data.failed > 0
            ? `Batch completed with ${data.successful} successes and ${data.failed} failures.`
            : 'Batch prediction completed successfully!';
        showNotification(successMessage, data.failed && data.failed > 0 ? 'warning' : 'success');
        if (startButton) startButton.disabled = false;
        
    } catch (error) {
        console.error('Batch prediction error:', error);
        showNotification(error.message || 'Failed to process batch prediction', 'error');
        if (progressContainer) progressContainer.style.display = 'none';
        if (startButton) startButton.disabled = false;
    }
}

function updateProgress(percentage, message) {
    document.getElementById('progressBar').style.width = percentage + '%';
    document.getElementById('progressText').textContent = percentage + '%';
    document.getElementById('progressMessage').textContent = message;
}

function displayBatchResults(data) {
    document.getElementById('batchProgress').style.display = 'none';
    document.getElementById('batchResults').style.display = 'block';
    
    // Update summary stats
    document.getElementById('totalProcessed').textContent = data.total_rows || data.total_patients || 0;
    document.getElementById('successCount').textContent = data.successful || 0;
    document.getElementById('savedCount').textContent = data.saved_to_database || 0;
    
    // Calculate failed count (invalid + duplicates + processing errors)
    const invalidCount = data.invalid_count || 0;
    const duplicateCount = data.duplicate_count || 0;
    const processingFailed = data.failed || 0;
    const totalFailed = invalidCount + duplicateCount + processingFailed;
    document.getElementById('failedCount').textContent = totalFailed;
    
    // Show validation errors if any
    displayBatchValidationErrors(data);
    
    renderBatchResultsTable(data.results || []);
    
    // Setup download button
    const downloadBtn = document.getElementById('downloadResults');
    downloadBtn.onclick = () => downloadBatchResults(data);
}

function displayBatchValidationErrors(data) {
    // Check if error display container exists, if not create it
    let errorContainer = document.getElementById('batchErrorsContainer');
    if (!errorContainer) {
        const resultsDiv = document.getElementById('batchResults');
        errorContainer = document.createElement('div');
        errorContainer.id = 'batchErrorsContainer';
        errorContainer.style.cssText = 'margin-bottom: 1rem;';
        resultsDiv.insertBefore(errorContainer, resultsDiv.querySelector('.batch-result-stats').nextSibling);
    }
    
    let html = '';
    
    // Show validation errors
    if (data.validation_errors && data.validation_errors.length > 0) {
        html += `
            <details class="batch-error-section" style="margin-bottom: 0.75rem; background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 0.5rem; padding: 0.75rem;">
                <summary style="cursor: pointer; color: #EF4444; font-weight: 600; display: flex; align-items: center; gap: 0.5rem;">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10"></circle>
                        <line x1="15" y1="9" x2="9" y2="15"></line>
                        <line x1="9" y1="9" x2="15" y2="15"></line>
                    </svg>
                    Invalid Rows (${data.validation_errors.length})
                </summary>
                <div style="margin-top: 0.75rem; max-height: 200px; overflow-y: auto;">
                    ${data.validation_errors.map(err => `
                        <div style="background: rgba(0,0,0,0.2); padding: 0.5rem; border-radius: 0.25rem; margin-bottom: 0.5rem; font-size: 0.875rem;">
                            <strong style="color: #FCA5A5;">Row ${err.row}:</strong>
                            <ul style="margin: 0.25rem 0 0 1rem; padding: 0; color: var(--text-secondary);">
                                ${err.errors.map(e => `<li>${e}</li>`).join('')}
                            </ul>
                        </div>
                    `).join('')}
                </div>
            </details>
        `;
    }
    
    // Show skipped duplicates
    if (data.skipped_duplicates && data.skipped_duplicates.length > 0) {
        html += `
            <details class="batch-error-section" style="margin-bottom: 0.75rem; background: rgba(251, 191, 36, 0.1); border: 1px solid rgba(251, 191, 36, 0.3); border-radius: 0.5rem; padding: 0.75rem;">
                <summary style="cursor: pointer; color: #FBBF24; font-weight: 600; display: flex; align-items: center; gap: 0.5rem;">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                        <line x1="12" y1="9" x2="12" y2="13"></line>
                        <line x1="12" y1="17" x2="12.01" y2="17"></line>
                    </svg>
                    Skipped Duplicates (${data.skipped_duplicates.length})
                </summary>
                <div style="margin-top: 0.75rem; max-height: 200px; overflow-y: auto;">
                    ${data.skipped_duplicates.map(dup => `
                        <div style="background: rgba(0,0,0,0.2); padding: 0.5rem; border-radius: 0.25rem; margin-bottom: 0.5rem; font-size: 0.875rem;">
                            <strong style="color: #FDE68A;">Row ${dup.row} - ${dup.student_id}:</strong>
                            <span style="color: var(--text-secondary); margin-left: 0.5rem;">${dup.reason}</span>
                        </div>
                    `).join('')}
                </div>
            </details>
        `;
    }
    
    errorContainer.innerHTML = html;
}

function renderBatchResultsTable(results) {
    const tableBody = document.getElementById('batchResultsBody');
    const emptyState = document.getElementById('batchResultsEmpty');
    const resultsCount = document.getElementById('batchResultsCount');
    if (!tableBody || !emptyState || !resultsCount) {
        return;
    }

    tableBody.innerHTML = '';

    const hasResults = Array.isArray(results) && results.length > 0;
    resultsCount.textContent = hasResults
        ? `${results.length} patient${results.length === 1 ? '' : 's'}`
        : '0 patients';
    emptyState.style.display = hasResults ? 'none' : 'block';

    if (!hasResults) {
        return;
    }

    const fragment = document.createDocumentFragment();
    results.forEach((result, index) => {
        const row = document.createElement('tr');
        row.dataset.patientId = String(result.patient_id ?? `row-${index}`);
        row.innerHTML = `
            <td>${result.row || index + 1}</td>
            <td>${result.patient_id ?? '-'}</td>
            <td>${formatClusterLabel(result.cluster)}</td>
            <td>${formatConfidenceValue(result.confidence)}</td>
            <td>${formatMetricValue(result.input_data ? result.input_data.BMI : undefined)}</td>
            <td>${formatMetricValue(result.input_data ? result.input_data.blood_glucose : undefined)}</td>
        `;
        row.addEventListener('click', () => {
            showBatchPredictionResult(result);
        });
        fragment.appendChild(row);
    });

    tableBody.appendChild(fragment);

    if (!activeBatchPatientId && results.length > 0) {
        activeBatchPatientId = String(results[0].patient_id ?? `row-0`);
    }
    highlightBatchResultRow(activeBatchPatientId);
}

function highlightBatchResultRow(patientId) {
    const rows = document.querySelectorAll('#batchResultsBody tr');
    rows.forEach(row => {
        const isActive = patientId && row.dataset.patientId === String(patientId);
        row.classList.toggle('active', Boolean(isActive));
    });
}

function formatClusterLabel(cluster) {
    if (cluster === null || cluster === undefined || cluster === '') {
        return '-';
    }
    return `Cluster ${cluster}`;
}

function formatConfidenceValue(value) {
    if (typeof value !== 'number' || Number.isNaN(value)) {
        return '-';
    }
    return `${(value * 100).toFixed(1)}%`;
}

function formatMetricValue(value) {
    if (value === null || value === undefined || value === '') {
        return '-';
    }
    if (typeof value === 'number') {
        return Number(value).toFixed(1);
    }
    return value;
}

function downloadBatchResults(data) {
    if (!data || !data.results) {
        showNotification('No results to download', 'error');
        return;
    }
    
    // Convert results to CSV
    const csvContent = convertResultsToCSV(data.results);
    
    // Create blob and download
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    
    link.setAttribute('href', url);
    link.setAttribute('download', `batch_predictions_${new Date().toISOString().slice(0,10)}.csv`);
    link.style.visibility = 'hidden';
    
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    showNotification('Results downloaded successfully!', 'success');
}

function convertResultsToCSV(results) {
    if (!results || results.length === 0) return '';
    
    // Get all keys from first result
    const headers = Object.keys(results[0]);
    
    // Create CSV header
    let csv = headers.join(',') + '\n';
    
    // Add data rows
    results.forEach(result => {
        const row = headers.map(header => {
            let value = result[header];
            
            // Handle values that might contain commas or quotes
            if (typeof value === 'string' && (value.includes(',') || value.includes('"') || value.includes('\n'))) {
                value = '"' + value.replace(/"/g, '""') + '"';
            }
            
            return value !== null && value !== undefined ? value : '';
        });
        
        csv += row.join(',') + '\n';
    });
    
    return csv;
}

function getBatchFailureReason(data) {
    if (!data) return 'Batch prediction failed.';
    if (data.error) {
        // Include counts if available
        let msg = data.error;
        if (data.invalid_count > 0 || data.duplicate_count > 0) {
            const parts = [];
            if (data.invalid_count > 0) parts.push(`${data.invalid_count} invalid rows`);
            if (data.duplicate_count > 0) parts.push(`${data.duplicate_count} duplicates`);
            msg += ` (${parts.join(', ')})`;
        }
        return msg;
    }
    if (data.message) return data.message;
    if (Array.isArray(data.schema_errors) && data.schema_errors.length > 0) {
        return data.schema_errors.join(' ');
    }
    if (data.summary) {
        if (data.summary.error) return data.summary.error;
        if (Array.isArray(data.summary.schema_errors) && data.summary.schema_errors.length > 0) {
            return data.summary.schema_errors.join(' ');
        }
    }
    if (Array.isArray(data.validation_errors) && data.validation_errors.length > 0) {
        const first = data.validation_errors[0];
        const rowInfo = first.row ? `Row ${first.row}: ` : '';
        if (Array.isArray(first.errors) && first.errors.length > 0) {
            return rowInfo + first.errors[0];
        }
    }
    if (Array.isArray(data.errors) && data.errors.length > 0) {
        const first = data.errors[0];
        const rowInfo = first.row ? `Row ${first.row}: ` : '';
        if (Array.isArray(first.errors) && first.errors.length > 0) {
            return rowInfo + first.errors[0];
        }
        if (typeof first.errors === 'string') {
            return rowInfo + first.errors;
        }
    }
    return 'Batch prediction failed. Please check your CSV and try again.';
}

function showBatchPredictionResult(batchResult) {
    if (!batchResult) return;
    
    const inputData = batchResult.input_data || {};
    
    // Build patient snapshot from input data
    const patientSnapshot = {
        first_name: inputData.First_Name || inputData.first_name || '',
        last_name: inputData.Last_Name || inputData.last_name || '',
        year_level: inputData.year_level,
        age: inputData.age_years || inputData.age,
        age_years: inputData.age_years || inputData.age,
        gender: inputData.gender || (inputData.is_female === 1 ? 'Female' : inputData.is_female === 0 ? 'Male' : null),
        is_female: inputData.is_female,
        height_cm: inputData.height_cm,
        weight_kg: inputData.weight_kg,
        bmi: inputData.bmi || inputData.BMI,
        BMI: inputData.BMI || inputData.bmi,
        has_respiratory_issue: inputData.has_respiratory_issue,
        has_pain: inputData.has_pain,
        has_fever: inputData.has_fever,
        has_allergy: inputData.has_allergy,
        is_uti: inputData.is_uti
    };
    
    // Use clinical intelligence from backend if available, otherwise build from cluster
    const clinicalIntelligence = batchResult.clinical_intelligence || {
        summary: getClusterCareRecommendation(batchResult.cluster),
        alerts: [],
        recommendations: getClusterRecommendations(batchResult.cluster)
    };
    
    const transformedResult = {
        patient_id: batchResult.patient_id,
        cluster: batchResult.cluster,
        confidence: batchResult.confidence,
        umap_coordinates: {
            x: batchResult.umap_x ?? 0,
            y: batchResult.umap_y ?? 0
        },
        patient_snapshot: patientSnapshot,
        clinical_intelligence: clinicalIntelligence
    };
    
    displayResults(transformedResult);
    activeBatchPatientId = batchResult.patient_id != null ? String(batchResult.patient_id) : null;
    highlightBatchResultRow(activeBatchPatientId);
}

// Helper to get cluster recommendations array
function getClusterRecommendations(cluster) {
    const recommendations = {
        0: [
            'Routine annual checkups and preventive care screenings',
            'Maintain current medication regimen and healthy lifestyle practices',
            'Focus on preventive measures such as vaccinations and wellness programs'
        ],
        1: [
            'Implement rapid respiratory assessment (respiratory rate, pulse oximetry, lung exam)',
            'Initiate fever management bundle and review exposure/contagion controls for shared housing',
            'Escalate to tele-infectious-disease consult when symptoms persist beyond 48 hours or red flags emerge'
        ],
        2: [
            'Comprehensive health assessment addressing all presenting symptoms',
            'Coordinate care between primary provider and relevant specialists',
            'Monitor for medication interactions and provide integrated treatment plan'
        ]
    };
    return recommendations[cluster] || [
        'Clinical management tailored to specific cluster needs',
        'Regular monitoring and appropriate medication management',
        'Lifestyle interventions based on individual characteristics'
    ];
}

function deriveInsightsFromInput(input) {
    if (!input) return null;
    const insights = {};
    if (input.BMI !== undefined && input.BMI !== null) {
        insights.bmi_category = categorizeBMILabel(input.BMI);
    }
    if (input.systolic_bp !== undefined && input.diastolic_bp !== undefined) {
        insights.bp_category = categorizeBloodPressureLabel(input.systolic_bp, input.diastolic_bp);
    }
    if (input.blood_glucose !== undefined && input.blood_glucose !== null) {
        insights.diabetes_risk = categorizeDiabetesRiskLabel(input.blood_glucose);
    }
    return Object.keys(insights).length ? insights : null;
}

function categorizeBMILabel(bmi) {
    if (bmi < 18.5) return 'Underweight';
    if (bmi < 25) return 'Normal';
    if (bmi < 30) return 'Overweight';
    if (bmi < 35) return 'Obesity Class I';
    if (bmi < 40) return 'Obesity Class II';
    return 'Obesity Class III';
}

function categorizeBloodPressureLabel(systolic, diastolic) {
    if (systolic < 120 && diastolic < 80) return 'Normal';
    if (systolic < 130 && diastolic < 80) return 'Elevated';
    if (systolic < 140 || diastolic < 90) return 'Stage 1 Hypertension';
    if (systolic < 180 || diastolic < 120) return 'Stage 2 Hypertension';
    return 'Hypertensive Crisis';
}

function categorizeDiabetesRiskLabel(glucose) {
    if (glucose < 100) return 'Low';
    if (glucose < 126) return 'Pre-Diabetic';
    return 'High';
}
