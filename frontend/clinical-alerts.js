/**
 * Clinical Alerts & Intelligence Display Component
 * Shows summary and recommendations for student health clusters
 */

class ClinicalAlertsDisplay {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Container #${containerId} not found`);
        }
    }

    /**
     * Display clinical intelligence summary
     */
    displayClinicalIntelligence(clinicalData) {
        if (!this.container || !clinicalData) return;

        const { alerts, recommendations, summary } = clinicalData;
        
        // Build simple display
        let html = `<div class="clinical-content">`;
        
        // Summary
        if (summary) {
            html += `
                <div class="clinical-summary-box">
                    <p>${summary}</p>
                </div>
            `;
        }

        // Recommendations (only if present)
        if (recommendations && recommendations.length > 0) {
            html += `
                <div class="clinical-recommendations">
                    <h4>Recommendations</h4>
                    <ul>
                        ${recommendations.map(rec => `<li>${rec}</li>`).join('')}
                    </ul>
                </div>
            `;
        }

        // Alerts (only if present and meaningful)
        if (alerts && alerts.length > 0) {
            html += `
                <div class="clinical-alerts-list">
                    <h4>Alerts</h4>
                    <ul>
                        ${alerts.map(alert => `<li>${alert}</li>`).join('')}
                    </ul>
                </div>
            `;
        }

        html += `</div>`;

        this.container.innerHTML = html;
    }

    /**
     * Clear all displayed content
     */
    clear() {
        if (this.container) {
            this.container.innerHTML = '';
        }
    }

    /**
     * Show loading state
     */
    showLoading() {
        if (this.container) {
            this.container.innerHTML = `
                <div class="clinical-loading">
                    <p>Analyzing clinical data...</p>
                </div>
            `;
        }
    }
}

// Global instance
const clinicalAlertsDisplay = new ClinicalAlertsDisplay('clinicalAlertsContainer');
