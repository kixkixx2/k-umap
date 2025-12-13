/**
 * Frontend validation utilities
 * Provides real-time form validation with user-friendly error messages
 */

const DEFAULT_FIELD_INFO = {
    // Required fields for the ISU student health clustering model
    age: { type: 'numeric', required: true, min: 1, max: 120, unit: 'years' },
    year_level: { type: 'numeric', required: true, min: 1, max: 5, unit: 'year' },
    gender: { type: 'categorical', required: true, options: ['Male', 'Female', 'Other'] },
    BMI: { type: 'numeric', required: false, min: 10, max: 60, unit: 'kg/m^2' },
    school_id: { type: 'string', required: true, pattern: '^\\d{2}-\\d{4}$', example: '22-0641' },
    // Boolean diagnostic flags (optional, default to 0)
    has_respiratory_issue: { type: 'categorical', required: false, options: [0, 1, '0', '1'] },
    has_pain: { type: 'categorical', required: false, options: [0, 1, '0', '1'] },
    has_fever: { type: 'categorical', required: false, options: [0, 1, '0', '1'] },
    has_allergy: { type: 'categorical', required: false, options: [0, 1, '0', '1'] },
    is_uti: { type: 'categorical', required: false, options: [0, 1, '0', '1'] },
   
};

class FormValidator {
    constructor() {
        this.fieldInfo = null;
        this.validationErrors = {};
    }

    /**
     * Load validation rules from backend
     */
    async loadValidationRules() {
        try {
            const response = await fetch(`${API_BASE_URL}/api/field_info`);
            const data = await response.json();
            if (data && data.success && data.fields) {
                this.fieldInfo = data.fields;
                console.log('✅ Validation rules loaded');
                return true;
            }
            throw new Error('Field info payload missing');
        } catch (error) {
            console.error('❌ Failed to load validation rules:', error);
            console.warn('⚠️ Falling back to built-in validation rules');
            this.fieldInfo = { ...DEFAULT_FIELD_INFO };
            return false;
        }
    }

    /**
     * Validate a single field
     * @param {string} fieldName - Name of the field
     * @param {any} value - Value to validate
     * @returns {object} { isValid, error }
     */
    validateField(fieldName, value) {
        const field = this.getFieldMetadata(fieldName);
        if (!field) {
            return { isValid: true, error: null };
        }

        // Check required fields
        if (field.required && (value === '' || value === null || value === undefined)) {
            return { 
                isValid: false, 
                error: `${this.formatFieldName(fieldName)} is required` 
            };
        }

        // If field is optional and empty, it's valid
        if (!field.required && (value === '' || value === null || value === undefined)) {
            return { isValid: true, error: null };
        }

        // Validate numeric fields
        if (field.type === 'numeric') {
            const numValue = parseFloat(value);
            
            if (isNaN(numValue)) {
                return { 
                    isValid: false, 
                    error: `${this.formatFieldName(fieldName)} must be a number` 
                };
            }

            if (numValue < 0) {
                return { 
                    isValid: false, 
                    error: `${this.formatFieldName(fieldName)} cannot be negative` 
                };
            }

            if (numValue < field.min || numValue > field.max) {
                return { 
                    isValid: false, 
                    error: `${this.formatFieldName(fieldName)} must be between ${field.min} and ${field.max} ${field.unit}` 
                };
            }
        }

        // Validate categorical fields
        if (field.type === 'categorical') {
            const validOptions = field.options.map(o => String(o));
            if (!validOptions.includes(String(value))) {
                return { 
                    isValid: false, 
                    error: `Invalid value for ${this.formatFieldName(fieldName)}` 
                };
            }
        }

        // Validate pattern if provided (e.g., school_id)
        if (field.pattern) {
            try {
                const re = new RegExp(field.pattern);
                if (!re.test(String(value))) {
                    return {
                        isValid: false,
                        error: `${this.formatFieldName(fieldName)} must match format ${field.example || field.pattern}`
                    };
                }
            } catch (e) {
                // invalid regex in metadata - ignore pattern
                console.warn('Invalid validation pattern for', fieldName, field.pattern);
            }
        }

        return { isValid: true, error: null };
    }

    /**
     * Safely fetch metadata for a field with fallback defaults
     */
    getFieldMetadata(fieldName) {
        if (this.fieldInfo && this.fieldInfo[fieldName]) {
            return this.fieldInfo[fieldName];
        }
        return DEFAULT_FIELD_INFO[fieldName] || null;
    }

    /**
     * Validate all form fields
     * @param {FormData} formData - Form data to validate
     * @returns {object} { isValid, errors }
     */
    validateForm(formData) {
        const errors = {};
        let isValid = true;

        for (let [key, value] of formData.entries()) {
            const result = this.validateField(key, value);
            if (!result.isValid) {
                errors[key] = result.error;
                isValid = false;
            }
        }

        // Cross-field validation
        const systolic = parseFloat(formData.get('systolic_bp'));
        const diastolic = parseFloat(formData.get('diastolic_bp'));
        
        if (!isNaN(systolic) && !isNaN(diastolic) && diastolic >= systolic) {
            errors['diastolic_bp'] = 'Diastolic BP should be lower than systolic BP';
            isValid = false;
        }

        this.validationErrors = errors;
        return { isValid, errors };
    }

    /**
     * Format field name for display
     */
    formatFieldName(fieldName) {
        return fieldName
            .split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }

    /**
     * Show validation error on field
     */
    showFieldError(fieldId, errorMessage) {
        const field = document.getElementById(fieldId);
        if (!field) return;
        const container = field.closest('.form-group') || field.parentNode;
        if (!container) return;

        // Remove any existing error first so we don't stack messages
        this.clearFieldError(fieldId);

        // Add error class
        field.classList.add('input-error');
        field.setAttribute('aria-invalid', 'true');

        // Add error message
        const errorDiv = document.createElement('div');
        errorDiv.className = 'field-error-message';
        errorDiv.textContent = errorMessage;
        errorDiv.id = `${fieldId}-error`;
        errorDiv.setAttribute('role', 'alert');
        errorDiv.setAttribute('aria-live', 'assertive');
        container.appendChild(errorDiv);
        field.setAttribute('aria-describedby', errorDiv.id);
    }

    /**
     * Clear validation error from field
     */
    clearFieldError(fieldId) {
        const field = document.getElementById(fieldId);
        if (!field) return;

        field.classList.remove('input-error');
        field.removeAttribute('aria-invalid');
        field.removeAttribute('aria-describedby');
        const container = field.closest('.form-group') || field.parentNode;
        const errorMsg = document.getElementById(`${fieldId}-error`);
        if (errorMsg && container && container.contains(errorMsg)) {
            errorMsg.remove();
        } else if (errorMsg && !container) {
            errorMsg.remove();
        }
    }

    /**
     * Setup real-time validation for a field
     */
    setupFieldValidation(fieldId) {
        const field = document.getElementById(fieldId);
        if (!field) return;

        const metadata = this.getFieldMetadata(fieldId);
        if (metadata && metadata.type === 'numeric') {
            if (metadata.min !== undefined) {
                field.setAttribute('aria-valuemin', metadata.min);
            }
            if (metadata.max !== undefined) {
                field.setAttribute('aria-valuemax', metadata.max);
            }
            field.setAttribute('inputmode', 'decimal');
        }

        const runValidation = () => {
            const constraintMessage = this.enforceNumericBounds(fieldId, field);
            if (constraintMessage) {
                this.showFieldError(fieldId, constraintMessage);
                return;
            }

            const result = this.validateField(fieldId, field.value);
            if (!result.isValid) {
                this.showFieldError(fieldId, result.error);
            } else {
                this.clearFieldError(fieldId);
                if (field.value !== '') {
                    field.dataset.lastValidValue = field.value;
                }
            }
        };

        // Validate on blur
        field.addEventListener('blur', runValidation);
        field.addEventListener('change', runValidation);

        // Clear error on input
        field.addEventListener('input', () => {
            this.clearFieldError(fieldId);
            if (metadata && metadata.type === 'numeric') {
                let sanitized = field.value.replace(/[^0-9.]/g, '');
                if (sanitized.includes('.')) {
                    const [whole, ...decimals] = sanitized.split('.');
                    sanitized = `${whole}.${decimals.join('')}`;
                }
                if (sanitized !== field.value) {
                    field.value = sanitized;
                }
            }
        });
    }

    /**
     * Enforce numeric min/max constraints on blur/change
     */
    enforceNumericBounds(fieldName, fieldElement) {
        const metadata = this.getFieldMetadata(fieldName);
        if (!metadata || metadata.type !== 'numeric') {
            return null;
        }

        const rawValue = fieldElement.value;
        if (rawValue === undefined || rawValue === null || rawValue === '') {
            return null;
        }

        const normalizedValue = typeof rawValue === 'string' ? rawValue.trim() : rawValue;
        if (normalizedValue === '') {
            return null;
        }

        const numericValue = Number(normalizedValue);
        if (Number.isNaN(numericValue)) {
            return `${this.formatFieldName(fieldName)} must be a number`;
        }

        const min = metadata.min;
        const max = metadata.max;

        if (numericValue < min || numericValue > max) {
            const lastValid = fieldElement.dataset.lastValidValue;
            if (lastValid !== undefined && lastValid !== '') {
                fieldElement.value = lastValid;
            } else {
                const clamped = Math.min(Math.max(numericValue, min), max);
                fieldElement.value = clamped;
            }
            const unit = metadata.unit ? ` ${metadata.unit}` : '';
            return `${this.formatFieldName(fieldName)} must be between ${min} and ${max}${unit}`;
        }

        fieldElement.dataset.lastValidValue = normalizedValue;
        return null;
    }

    /**
     * Add required indicators to form
     */
    addRequiredIndicators() {
        if (!this.fieldInfo) return;

        for (const [fieldName, info] of Object.entries(this.fieldInfo)) {
            if (info.required) {
                const field = document.getElementById(fieldName);
                if (field) {
                    const label = document.querySelector(`label[for="${fieldName}"]`);
                    if (label && !label.querySelector('.required-indicator')) {
                        const indicator = document.createElement('span');
                        indicator.className = 'required-indicator';
                        indicator.textContent = ' *';
                        indicator.style.color = '#EF4444';
                        label.appendChild(indicator);
                    }
                }
            }
        }
    }

    /**
     * Add unit indicators to numeric fields
     */
    addUnitIndicators() {
        if (!this.fieldInfo) return;

        for (const [fieldName, info] of Object.entries(this.fieldInfo)) {
            if (info.type === 'numeric' && info.unit) {
                const field = document.getElementById(fieldName);
                if (field && !field.parentNode.querySelector('.unit-indicator')) {
                    const unitSpan = document.createElement('span');
                    unitSpan.className = 'unit-indicator';
                    unitSpan.textContent = info.unit;
                    unitSpan.style.cssText = 'margin-left: 8px; color: #9CA3AF; font-size: 0.875rem;';
                    field.parentNode.appendChild(unitSpan);
                }
            }
        }
    }

    /**
     * Show loading state on button
     */
    setButtonLoading(buttonId, isLoading) {
        const button = document.getElementById(buttonId);
        if (!button) return;

        if (isLoading) {
            button.dataset.originalText = button.innerHTML;
            button.disabled = true;
            button.innerHTML = '<span class="spinner"></span> Processing...';
        } else {
            button.disabled = false;
            button.innerHTML = button.dataset.originalText || 'Submit';
        }
    }
}

// Create global validator instance
const formValidator = new FormValidator();

// Auto-calculate BMI if height and weight fields exist
function setupBMICalculator() {
    const heightField = document.getElementById('height_cm');
    const weightField = document.getElementById('weight_kg');
    const bmiField = document.getElementById('BMI');

    if (heightField && weightField && bmiField) {
        const calculateBMI = () => {
            // heightField is in cm; convert to meters
            const heightCm = parseFloat(heightField.value);
            const height = !isNaN(heightCm) && heightCm > 0 ? (heightCm / 100.0) : NaN;
            const weight = parseFloat(weightField.value);

            if (!isNaN(height) && !isNaN(weight) && height > 0) {
                const bmi = weight / (height * height);
                bmiField.value = bmi.toFixed(1);
                
                // Show category
                const category = categorizeBMI(bmi);
                showBMICategory(category);
                // If the page provides a handler, ask it to check for unrealistic BMI and prompt the user
                try {
                    if (typeof window !== 'undefined' && window.maybeShowBMIUnitWarning) {
                        window.maybeShowBMIUnitWarning(bmi, heightCm, weight);
                    }
                } catch (e) {
                    // ignore if window isn't available (e.g., server-side tests)
                }
            }
        };

        heightField.addEventListener('input', calculateBMI);
        weightField.addEventListener('input', calculateBMI);
    }
}

function categorizeBMI(bmi) {
    if (bmi < 18.5) return { text: 'Underweight', color: '#F59E0B' };
    if (bmi < 25) return { text: 'Normal', color: '#10B981' };
    if (bmi < 30) return { text: 'Overweight', color: '#F59E0B' };
    if (bmi < 35) return { text: 'Obesity I', color: '#EF4444' };
    if (bmi < 40) return { text: 'Obesity II', color: '#DC2626' };
    return { text: 'Obesity III', color: '#991B1B' };
}

function showBMICategory(category) {
    let categoryDiv = document.getElementById('bmi-category');
    
    if (!categoryDiv) {
        categoryDiv = document.createElement('div');
        categoryDiv.id = 'bmi-category';
        categoryDiv.style.cssText = 'margin-top: 4px; font-size: 0.875rem; font-weight: 500;';
        document.getElementById('BMI').parentNode.appendChild(categoryDiv);
    }

    categoryDiv.textContent = `Category: ${category.text}`;
    categoryDiv.style.color = category.color;
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { FormValidator, formValidator };
}
