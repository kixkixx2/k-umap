/**
 * Visualization Enhancement Tools
 * Provides chart export, enhanced tooltips, and 3D visualization
 */

const RISK_LABELS = {
    0: 'Cluster 0',
    1: 'Cluster 1',
    2: 'Cluster 2'
};

function resolveCoordinate(patient, axis = 'x') {
    if (typeof getCoordinateValue === 'function') {
        return getCoordinateValue(patient, axis);
    }
    const candidates = axis === 'x'
        ? ['x', 'umap_x', 'umapX', 'umap1']
        : ['y', 'umap_y', 'umapY', 'umap2'];
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

function resolveNumericFeature(patient, key, fallback = 0) {
    if (!key) return fallback;
    if (patient.features && patient.features[key] !== undefined) {
        return patient.features[key];
    }
    const directValue = patient[key];
    if (directValue === undefined || directValue === null || directValue === '') {
        return fallback;
    }
    const numeric = Number(directValue);
    return Number.isNaN(numeric) ? fallback : numeric;
}

class VisualizationEnhancements {
    constructor() {
        this.currentPlotDiv = null;
        this.is3DMode = false;
    }

    /**
     * Export plot to image
     */
    exportPlot(plotDiv, format = 'png', filename = 'cluster-plot') {
        if (!plotDiv) {
            console.error('No plot element provided');
            return;
        }

        const validFormats = ['png', 'svg', 'jpeg', 'webp'];
        if (!validFormats.includes(format)) {
            console.error(`Invalid format: ${format}. Use one of: ${validFormats.join(', ')}`);
            return;
        }

        // Prepare download options
        const downloadOptions = {
            format: format,
            width: 1200,
            height: 800,
            filename: filename
        };

        // Use Plotly's built-in download function
        Plotly.downloadImage(plotDiv, downloadOptions)
            .then(() => {
                console.log(`✅ Plot exported as ${format}`);
                this.showNotification(`Plot exported as ${format.toUpperCase()}`, 'success');
            })
            .catch(error => {
                console.error('Export failed:', error);
                this.showNotification(`Export failed: ${error.message}`, 'error');
            });
    }

    /**
     * Add export buttons to a plot container
     */
    addExportButtons(plotContainerId, plotDiv) {
        const container = document.getElementById(plotContainerId);
        if (!container) return;

        // Check if export buttons already exist
        if (container.querySelector('.viz-export-toolbar')) {
            return;
        }

        const toolbar = document.createElement('div');
        toolbar.className = 'viz-export-toolbar';
        toolbar.innerHTML = `
            <button class="viz-tool-btn" data-format="png" title="Export as PNG">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                    <polyline points="7 10 12 15 17 10"></polyline>
                    <line x1="12" y1="15" x2="12" y2="3"></line>
                </svg>
                PNG
            </button>
            <button class="viz-tool-btn" data-format="svg" title="Export as SVG">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                    <polyline points="7 10 12 15 17 10"></polyline>
                    <line x1="12" y1="15" x2="12" y2="3"></line>
                </svg>
                SVG
            </button>
            <button class="viz-tool-btn" data-format="jpeg" title="Export as JPEG">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                    <polyline points="7 10 12 15 17 10"></polyline>
                    <line x1="12" y1="15" x2="12" y2="3"></line>
                </svg>
                JPG
            </button>
            <button class="viz-tool-btn viz-tool-reset" id="resetZoom" title="Reset View">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"></path>
                    <path d="M21 3v5h-5"></path>
                    <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"></path>
                    <path d="M3 21v-5h5"></path>
                </svg>
                Reset
            </button>
        `;

        // Insert toolbar at the top of container
        container.insertBefore(toolbar, container.firstChild);

        // Add event listeners
        toolbar.querySelectorAll('[data-format]').forEach(btn => {
            btn.addEventListener('click', () => {
                const format = btn.getAttribute('data-format');
                this.exportPlot(plotDiv, format, `cluster-visualization-${Date.now()}`);
            });
        });

        // Reset zoom button
        const resetBtn = toolbar.querySelector('#resetZoom');
        if (resetBtn) {
            resetBtn.addEventListener('click', () => {
                Plotly.relayout(plotDiv, {
                    'xaxis.autorange': true,
                    'yaxis.autorange': true
                });
                this.showNotification('View reset', 'info');
            });
        }
    }

    /**
     * Create enhanced tooltip template
     */
    createEnhancedTooltip(patient) {
        const bmiValue = resolveNumericFeature(patient, 'bmi', patient.BMI || 0);
        const ageValue = patient.age ?? patient.age_years ?? 'N/A';
        const diabetesValue = patient.diabetes ?? patient.has_diabetes;
        const hypertensionValue = patient.hypertension ?? patient.has_hypertension;
        return `
            <b>Patient ID:</b> ${patient.patient_id}<br>
            <b>Cluster:</b> ${patient.cluster}<br>
            <b>Age:</b> ${ageValue} years<br>
            <b>Gender:</b> ${patient.gender || 'N/A'}<br>
            <b>BMI:</b> ${bmiValue ? bmiValue.toFixed(1) : 'N/A'}<br>
            <b>Blood Pressure:</b> ${patient.systolic_bp || 'N/A'}/${patient.diastolic_bp || 'N/A'}<br>
            <b>Diabetes:</b> ${diabetesValue === undefined ? 'N/A' : diabetesValue ? 'Yes' : 'No'}<br>
            <b>Hypertension:</b> ${hypertensionValue === undefined ? 'N/A' : hypertensionValue ? 'Yes' : 'No'}
            <extra></extra>
        `;
    }

    /**
     * Add cluster statistics overlay
     */
    addClusterStatsOverlay(plotContainerId, patientsData) {
        const container = document.getElementById(plotContainerId);
        if (!container || container.querySelector('.cluster-stats-overlay')) {
            return;
        }

        // Calculate cluster statistics
        const clusterStats = this.calculateClusterStats(patientsData);

        const overlay = document.createElement('div');
        overlay.className = 'cluster-stats-overlay';
        overlay.innerHTML = `
            <h4>Cluster Statistics</h4>
            <div class="cluster-stats-grid">
                ${clusterStats.map(stat => {
                    const label = RISK_LABELS[stat.cluster];
                    const suffix = (label && !/^(Cluster\s*\d+)/i.test(label)) ? ' · ' + label : '';
                    return `
                    <div class="cluster-stat-item" style="border-left-color: ${stat.color}">
                        <span class="stat-cluster">Cluster ${stat.cluster}${suffix}&nbsp;</span>
                        <span class="stat-count">${stat.count} patients</span>
                        <span class="stat-percent">${stat.percentage}%</span>
                    </div>
                `}).join('')}
            </div>
        `;

        container.appendChild(overlay);
    }

    /**
     * Calculate cluster statistics
     */
    calculateClusterStats(patientsData) {
        const CLUSTER_COLORS = [
            '#10B981', '#F59E0B', '#EF4444', '#8B5CF6',
            '#EC4899', '#14B8A6', '#F97316', '#6366F1', '#A855F7'
        ];

        const clusterCounts = {};
        patientsData.forEach(p => {
            clusterCounts[p.cluster] = (clusterCounts[p.cluster] || 0) + 1;
        });

        const total = patientsData.length;
        return Object.entries(clusterCounts).map(([cluster, count]) => ({
            cluster: parseInt(cluster),
            count: count,
            percentage: ((count / total) * 100).toFixed(1),
            color: CLUSTER_COLORS[parseInt(cluster) % CLUSTER_COLORS.length]
        })).sort((a, b) => a.cluster - b.cluster);
    }

    /**
     * Create 3D visualization
     */
    create3DVisualization(plotDiv, patientsData, umapModel) {
        if (!patientsData || patientsData.length === 0) {
            console.error('No patient data for 3D visualization');
            return;
        }

        const CLUSTER_COLORS = [
            '#10B981', '#F59E0B', '#EF4444', '#8B5CF6',
            '#EC4899', '#14B8A6', '#F97316', '#6366F1', '#A855F7'
        ];

        // Group data by cluster
        const clusterGroups = {};
        patientsData.forEach(patient => {
            if (!clusterGroups[patient.cluster]) {
                clusterGroups[patient.cluster] = {
                    x: [],
                    y: [],
                    z: [], // For 3D, use a third feature or synthesize one
                    ids: [],
                    text: []
                };
            }
            clusterGroups[patient.cluster].x.push(resolveCoordinate(patient, 'x'));
            clusterGroups[patient.cluster].y.push(resolveCoordinate(patient, 'y'));
            // Use BMI as Z-axis or default to 0
            clusterGroups[patient.cluster].z.push(resolveNumericFeature(patient, 'bmi', patient.BMI || 0));
            clusterGroups[patient.cluster].ids.push(patient.patient_id);
            clusterGroups[patient.cluster].text.push(this.createEnhancedTooltip(patient));
        });

        // Create 3D traces
        const traces = Object.keys(clusterGroups).map(cluster => ({
            x: clusterGroups[cluster].x,
            y: clusterGroups[cluster].y,
            z: clusterGroups[cluster].z,
            mode: 'markers',
            type: 'scatter3d',
            name: `Cluster ${cluster}`,
            marker: {
                size: 5,
                color: CLUSTER_COLORS[parseInt(cluster) % CLUSTER_COLORS.length],
                opacity: 0.8,
                line: {
                    color: 'white',
                    width: 0.5
                }
            },
            text: clusterGroups[cluster].text,
            hovertemplate: '%{text}',
            ids: clusterGroups[cluster].ids
        }));

        const layout = {
            title: {
                text: 'Patient Clusters - 3D Visualization',
                font: { size: 20, color: '#f8fafc', family: 'Inter, sans-serif' }
            },
            scene: {
                xaxis: { title: 'UMAP Component 1', gridcolor: '#334155', color: '#cbd5e1' },
                yaxis: { title: 'UMAP Component 2', gridcolor: '#334155', color: '#cbd5e1' },
                zaxis: { title: 'BMI', gridcolor: '#334155', color: '#cbd5e1' },
                bgcolor: '#0f172a'
            },
            paper_bgcolor: '#1e293b',
            plot_bgcolor: '#0f172a',
            font: { color: '#cbd5e1', family: 'Inter, sans-serif' },
            showlegend: true,
            legend: {
                bgcolor: '#1e293b',
                bordercolor: '#334155',
                borderwidth: 1
            },
            hovermode: 'closest',
            margin: { l: 0, r: 0, b: 0, t: 50 }
        };

        const config = {
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['lasso2d', 'select2d']
        };

        Plotly.newPlot(plotDiv, traces, layout, config);
        this.is3DMode = true;
        console.log('✅ 3D visualization created');
    }

    /**
     * Toggle between 2D and 3D
     */
    toggle3DMode(plotDiv, patientsData) {
        if (this.is3DMode) {
            // Switch back to 2D - need to call original render function
            console.log('Switching to 2D mode...');
            this.is3DMode = false;
            return false; // Signal to re-render with 2D
        } else {
            // Switch to 3D
            this.create3DVisualization(plotDiv, patientsData);
            return true;
        }
    }

    /**
     * Add 3D toggle button
     */
    add3DToggle(containerId, plotDiv, patientsData, render2DCallback) {
        const container = document.getElementById(containerId);
        if (!container || container.querySelector('.viz-3d-toggle')) {
            return;
        }

        const toggle = document.createElement('button');
        toggle.className = 'viz-tool-btn viz-3d-toggle';
        toggle.innerHTML = `
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M12 2L2 7l10 5 10-5-10-5z"></path>
                <path d="M2 17l10 5 10-5"></path>
                <path d="M2 12l10 5 10-5"></path>
            </svg>
            <span class="toggle-text">3D View</span>
        `;

        // Find toolbar or create one
        let toolbar = container.querySelector('.viz-export-toolbar');
        if (!toolbar) {
            toolbar = document.createElement('div');
            toolbar.className = 'viz-export-toolbar';
            container.insertBefore(toolbar, container.firstChild);
        }

        toolbar.appendChild(toggle);

        // Add click handler
        toggle.addEventListener('click', () => {
            if (this.is3DMode) {
                // Switch to 2D
                this.is3DMode = false;
                toggle.querySelector('.toggle-text').textContent = '3D View';
                render2DCallback();
                this.showNotification('Switched to 2D view', 'info');
            } else {
                // Switch to 3D
                this.create3DVisualization(plotDiv, patientsData);
                toggle.querySelector('.toggle-text').textContent = '2D View';
                this.showNotification('Switched to 3D view', 'info');
            }
        });
    }

    /**
     * Show notification
     */
    showNotification(message, type = 'info') {
        // Check if global notification function exists
        if (typeof showNotification === 'function') {
            showNotification(message, type);
        } else {
            console.log(`[${type.toUpperCase()}] ${message}`);
        }
    }
}

// Global instance
const vizEnhancements = new VisualizationEnhancements();
