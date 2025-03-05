/**
 * Kompot Interactive Report JavaScript
 * 
 * This file contains all the client-side functionality for the Kompot report,
 * including loading data, generating interactive plots, and handling user interactions.
 */

// Global variables to store data and state
let geneData = [];
let umapData = null;
let comparisonData = null;
let genePlotData = null;
let currentComparisonIndex = 0;
let currentGene = null;
let selectedGenes = new Set();
let reportId = null;

// Initialize the report
function initReport(id) {
    console.log("Initializing Kompot report...");
    reportId = id;
    
    // Load all data files
    Promise.all([
        fetch('data/gene_data.json').then(response => response.json()),
        fetch('data/umap_data.json').then(response => {
            if (response.ok) {
                return response.json();
            } else {
                return null;
            }
        }),
        fetch('data/comparison_data.json').then(response => {
            if (response.ok) {
                return response.json();
            } else {
                return null;
            }
        }),
        fetch('data/gene_plots.json').then(response => {
            if (response.ok) {
                return response.json();
            } else {
                return null;
            }
        })
    ])
    .then(([geneDataResponse, umapDataResponse, comparisonDataResponse, genePlotDataResponse]) => {
        // Store the data in global variables
        geneData = geneDataResponse;
        umapData = umapDataResponse;
        comparisonData = comparisonDataResponse;
        genePlotData = genePlotDataResponse;
        
        // Initialize the UI
        initUI();
        
        // Show/hide sections based on available data
        if (umapData) {
            document.getElementById('umap-plots').style.display = 'block';
        }
        
        if (genePlotData) {
            document.getElementById('gene-specific').style.display = 'block';
        }
        
        if (comparisonData) {
            document.getElementById('method-comparison').style.display = 'block';
        }
        
        // Generate summary statistics
        generateSummaryStats();
        
        // Load the first comparison by default
        if (geneData.length > 0) {
            loadGeneTable(0);
            initVolcanoPlot(0);
        }
        
        // Initialize UMAP plot if data is available
        if (umapData) {
            initUMAPPlot();
        }
        
        // Initialize method comparison if data is available
        if (comparisonData) {
            initMethodComparison();
        }
    })
    .catch(error => {
        console.error("Error loading data:", error);
        document.getElementById('summary-stats').innerHTML = 
            '<div class="error">Error loading data. Please check the console for details.</div>';
    });
}

// Initialize the UI components
function initUI() {
    // Populate condition selector
    const conditionSelect = document.getElementById('condition-select');
    geneData.forEach((data, index) => {
        const option = document.createElement('option');
        option.value = index;
        option.textContent = `${data.condition1} vs ${data.condition2}`;
        conditionSelect.appendChild(option);
    });
    
    // Setup condition selector event
    conditionSelect.addEventListener('change', function() {
        const index = parseInt(this.value);
        currentComparisonIndex = index;
        loadGeneTable(index);
        updateVolcanoPlot(index);
    });
    
    // Setup volcano plot metric selectors
    document.getElementById('volcano-x-metric').addEventListener('change', function() {
        updateVolcanoPlot(currentComparisonIndex);
    });
    
    document.getElementById('volcano-y-metric').addEventListener('change', function() {
        updateVolcanoPlot(currentComparisonIndex);
    });
    
    // Setup UMAP annotation selector
    if (umapData) {
        const umapAnnotationSelect = document.getElementById('umap-annotation');
        for (const anno in umapData.annotations) {
            const option = document.createElement('option');
            option.value = anno;
            option.textContent = anno;
            umapAnnotationSelect.appendChild(option);
            
            // Set default selection
            if (anno === umapData.default_annotation) {
                option.selected = true;
            }
        }
        
        umapAnnotationSelect.addEventListener('change', function() {
            updateUMAPPlot(this.value);
        });
    }
    
    // Setup gene search
    if (genePlotData) {
        document.getElementById('search-gene-btn').addEventListener('click', function() {
            const geneSearch = document.getElementById('gene-search');
            const geneName = geneSearch.value.trim();
            if (geneName && genePlotData[geneName]) {
                currentGene = geneName;
                generateGenePlots(geneName);
                highlightGeneInTable(geneName);
                highlightGeneInPlot(geneName);
            } else {
                alert(`Gene "${geneName}" not found in the dataset.`);
            }
        });
        
        // Allow pressing Enter to search
        document.getElementById('gene-search').addEventListener('keyup', function(event) {
            if (event.key === 'Enter') {
                document.getElementById('search-gene-btn').click();
            }
        });
    }
    
    // Setup method comparison selectors
    if (comparisonData) {
        const comparisonSelect = document.getElementById('comparison-select');
        const methodSelect = document.getElementById('method-select');
        
        comparisonData.forEach((data, index) => {
            const option = document.createElement('option');
            option.value = index;
            option.textContent = data.name;
            comparisonSelect.appendChild(option);
        });
        
        // Set methods for the first comparison
        populateMethodSelector(0);
        
        comparisonSelect.addEventListener('change', function() {
            const index = parseInt(this.value);
            populateMethodSelector(index);
            updateComparisonPlot(index, methodSelect.value);
        });
        
        methodSelect.addEventListener('change', function() {
            updateComparisonPlot(parseInt(comparisonSelect.value), this.value);
        });
    }
}

// Generate summary statistics
function generateSummaryStats() {
    let summaryHtml = '';
    
    // Count total number of comparisons
    const comparisonsCount = geneData.length;
    
    // Get total number of genes
    const uniqueGenes = new Set();
    geneData.forEach(data => {
        data.genes.forEach(gene => {
            uniqueGenes.add(gene.gene);
        });
    });
    
    // Count total significant genes (abs(mahalanobis) > 3)
    let significantGenes = 0;
    geneData.forEach(data => {
        significantGenes += data.genes.filter(gene => Math.abs(gene.mahalanobis_distance) > 3).length;
    });
    
    summaryHtml += `
        <div class="summary-card">
            <h3>Dataset Summary</h3>
            <ul>
                <li><strong>Comparisons:</strong> ${comparisonsCount}</li>
                <li><strong>Unique Genes:</strong> ${uniqueGenes.size}</li>
                <li><strong>Significant Genes:</strong> ${significantGenes} <span class="note">(abs(mahalanobis) > 3)</span></li>
            </ul>
        </div>
    `;
    
    // Add comparison-specific summaries
    summaryHtml += '<div class="summary-comparisons">';
    geneData.forEach(data => {
        const upregulated = data.genes.filter(gene => gene.log2FoldChange > 0).length;
        const downregulated = data.genes.filter(gene => gene.log2FoldChange < 0).length;
        
        summaryHtml += `
            <div class="summary-comparison">
                <h4>${data.condition1} vs ${data.condition2}</h4>
                <p><span class="up">▲ ${upregulated}</span> | <span class="down">▼ ${downregulated}</span></p>
            </div>
        `;
    });
    summaryHtml += '</div>';
    
    document.getElementById('summary-stats').innerHTML = summaryHtml;
}

// Load gene table for the selected comparison
function loadGeneTable(comparisonIndex) {
    const data = geneData[comparisonIndex];
    const container = document.getElementById('gene-table-container');
    
    // Clear previous table
    container.innerHTML = '';
    
    // Create table element
    const table = document.createElement('table');
    table.id = 'genes-table';
    table.className = 'display';
    
    // Create table header
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    const columns = [
        { title: 'Gene', data: 'gene' },
        { title: 'Log2FC', data: 'log2FoldChange', render: formatNumber },
        { title: 'Z-score', data: 'z_score', render: formatNumber },
        { title: 'Mahalanobis', data: 'mahalanobis_distance', render: formatNumber },
        { title: 'Weighted FC', data: 'weighted_fold_change', render: formatNumber },
        { title: 'FC Std', data: 'fold_change_std', render: formatNumber }
    ];
    
    columns.forEach(col => {
        const th = document.createElement('th');
        th.textContent = col.title;
        headerRow.appendChild(th);
    });
    
    thead.appendChild(headerRow);
    table.appendChild(thead);
    
    // Create table body
    const tbody = document.createElement('tbody');
    data.genes.forEach(gene => {
        const row = document.createElement('tr');
        row.dataset.gene = gene.gene;
        row.addEventListener('click', function() {
            const geneName = this.dataset.gene;
            if (genePlotData && genePlotData[geneName]) {
                currentGene = geneName;
                generateGenePlots(geneName);
                
                // Highlight selected row
                const rows = tbody.querySelectorAll('tr');
                rows.forEach(r => r.classList.remove('selected-gene'));
                this.classList.add('selected-gene');
                
                // Highlight in plot
                highlightGeneInPlot(geneName);
                
                // Update gene search box
                document.getElementById('gene-search').value = geneName;
            }
        });
        
        columns.forEach(col => {
            const td = document.createElement('td');
            if (col.render) {
                td.textContent = col.render(gene[col.data]);
            } else {
                td.textContent = gene[col.data];
            }
            row.appendChild(td);
        });
        
        tbody.appendChild(row);
    });
    
    table.appendChild(tbody);
    container.appendChild(table);
    
    // Initialize DataTable
    $(table).DataTable({
        paging: true,
        searching: true,
        ordering: true,
        info: true,
        pageLength: 15,
        order: [[3, 'desc']] // Sort by Mahalanobis distance by default
    });
}

// Initialize volcano plot
function initVolcanoPlot(comparisonIndex) {
    const data = geneData[comparisonIndex];
    const container = document.getElementById('volcano-plot-container');
    
    // Get metrics for axes
    const xMetric = document.getElementById('volcano-x-metric').value;
    const yMetric = document.getElementById('volcano-y-metric').value;
    
    // Extract data for plotting
    const x = data.genes.map(gene => gene[xMetric]);
    const y = data.genes.map(gene => gene[yMetric]);
    const text = data.genes.map(gene => gene.gene);
    
    // Create Plotly trace
    const trace = {
        x: x,
        y: y,
        mode: 'markers',
        type: 'scatter',
        text: text,
        hoverinfo: 'text',
        marker: {
            size: 7,
            color: y.map(val => Math.abs(val)),
            colorscale: 'Viridis',
            colorbar: {
                title: yMetric
            }
        }
    };
    
    // Create layout
    const layout = {
        title: `${data.condition1} vs ${data.condition2}`,
        xaxis: { title: formatAxisLabel(xMetric) },
        yaxis: { title: formatAxisLabel(yMetric) },
        hovermode: 'closest'
    };
    
    // Plot
    Plotly.newPlot(container, [trace], layout);
    
    // Add click event
    container.on('plotly_click', function(eventData) {
        const pointIndex = eventData.points[0].pointIndex;
        const geneName = data.genes[pointIndex].gene;
        
        if (genePlotData && genePlotData[geneName]) {
            currentGene = geneName;
            generateGenePlots(geneName);
            highlightGeneInTable(geneName);
            
            // Update gene search box
            document.getElementById('gene-search').value = geneName;
        }
    });
}

// Update volcano plot for a different comparison or metrics
function updateVolcanoPlot(comparisonIndex) {
    const data = geneData[comparisonIndex];
    const container = document.getElementById('volcano-plot-container');
    
    // Get metrics for axes
    const xMetric = document.getElementById('volcano-x-metric').value;
    const yMetric = document.getElementById('volcano-y-metric').value;
    
    // Extract data for plotting
    const x = data.genes.map(gene => gene[xMetric]);
    const y = data.genes.map(gene => gene[yMetric]);
    const text = data.genes.map(gene => gene.gene);
    
    // Update data
    const update = {
        x: [x],
        y: [y],
        text: [text],
        'marker.color': [y.map(val => Math.abs(val))],
        'marker.colorbar.title': yMetric
    };
    
    // Update layout
    const layout = {
        title: `${data.condition1} vs ${data.condition2}`,
        xaxis: { title: formatAxisLabel(xMetric) },
        yaxis: { title: formatAxisLabel(yMetric) }
    };
    
    // Update plot
    Plotly.update(container, update, layout);
    
    // Highlight current gene if any
    if (currentGene) {
        highlightGeneInPlot(currentGene);
    }
}

// Initialize UMAP plot
function initUMAPPlot() {
    const container = document.getElementById('umap-plot-container');
    const annotation = document.getElementById('umap-annotation').value;
    
    // Extract data
    const x = umapData.coordinates.map(coord => coord[0]);
    const y = umapData.coordinates.map(coord => coord[1]);
    const values = umapData.annotations[annotation].values;
    const categories = umapData.annotations[annotation].categories;
    
    // Create a trace for each category
    const traces = [];
    
    for (let i = 0; i < categories.length; i++) {
        const categoryIndices = values.map((val, idx) => val === i ? idx : null).filter(idx => idx !== null);
        
        traces.push({
            x: categoryIndices.map(idx => x[idx]),
            y: categoryIndices.map(idx => y[idx]),
            mode: 'markers',
            type: 'scatter',
            name: categories[i],
            marker: {
                size: 5,
                opacity: 0.7
            },
            hoverinfo: 'text',
            text: categoryIndices.map(idx => `${annotation}: ${categories[i]}`)
        });
    }
    
    // Create layout
    const layout = {
        title: `UMAP - Colored by ${annotation}`,
        xaxis: { title: 'UMAP 1' },
        yaxis: { title: 'UMAP 2' },
        hovermode: 'closest',
        showlegend: true,
        legend: {
            title: { text: annotation },
            orientation: 'v',
            xanchor: 'right',
            y: 1
        }
    };
    
    // Plot
    Plotly.newPlot(container, traces, layout);
}

// Update UMAP plot with different annotation
function updateUMAPPlot(annotation) {
    const container = document.getElementById('umap-plot-container');
    
    // Extract data
    const x = umapData.coordinates.map(coord => coord[0]);
    const y = umapData.coordinates.map(coord => coord[1]);
    const values = umapData.annotations[annotation].values;
    const categories = umapData.annotations[annotation].categories;
    
    // Create a trace for each category
    const traces = [];
    
    for (let i = 0; i < categories.length; i++) {
        const categoryIndices = values.map((val, idx) => val === i ? idx : null).filter(idx => idx !== null);
        
        traces.push({
            x: categoryIndices.map(idx => x[idx]),
            y: categoryIndices.map(idx => y[idx]),
            mode: 'markers',
            type: 'scatter',
            name: categories[i],
            marker: {
                size: 5,
                opacity: 0.7
            },
            hoverinfo: 'text',
            text: categoryIndices.map(idx => `${annotation}: ${categories[i]}`)
        });
    }
    
    // Update layout
    const layout = {
        title: `UMAP - Colored by ${annotation}`,
        legend: {
            title: { text: annotation }
        }
    };
    
    // Reset and plot
    Plotly.purge(container);
    Plotly.newPlot(container, traces, layout);
}

// Generate plots for a specific gene
function generateGenePlots(geneName) {
    if (!genePlotData || !genePlotData[geneName]) {
        console.error(`No data available for gene: ${geneName}`);
        return;
    }
    
    const container = document.getElementById('gene-plots-container');
    container.innerHTML = '';
    
    // Get gene-specific data
    const gene = genePlotData[geneName];
    
    // Create header
    const header = document.createElement('h3');
    header.textContent = `Gene: ${geneName}`;
    container.appendChild(header);
    
    // Create grid for plots
    const grid = document.createElement('div');
    grid.className = 'plot-grid';
    container.appendChild(grid);
    
    // Create plots for each comparison
    Object.keys(gene).forEach(comparison => {
        const compData = gene[comparison];
        const plotDiv = document.createElement('div');
        plotDiv.className = 'gene-plot';
        grid.appendChild(plotDiv);
        
        const cond1 = compData.condition1;
        const cond2 = compData.condition2;
        
        // Create traces for both conditions
        const trace1 = {
            y: cond1.values,
            type: 'violin',
            name: cond1.name,
            box: { visible: true },
            meanline: { visible: true },
            side: 'negative'
        };
        
        const trace2 = {
            y: cond2.values,
            type: 'violin',
            name: cond2.name,
            box: { visible: true },
            meanline: { visible: true },
            side: 'positive'
        };
        
        const layout = {
            title: comparison.replace('_vs_', ' vs '),
            height: 300,
            width: 400,
            margin: { l: 60, r: 20, t: 40, b: 60 }
        };
        
        Plotly.newPlot(plotDiv, [trace1, trace2], layout);
    });
}

// Initialize method comparison plots
function initMethodComparison() {
    const comparisonSelect = document.getElementById('comparison-select');
    const methodSelect = document.getElementById('method-select');
    
    if (comparisonSelect.options.length > 0 && methodSelect.options.length > 0) {
        const compIndex = parseInt(comparisonSelect.value);
        const method = methodSelect.value;
        updateComparisonPlot(compIndex, method);
    }
}

// Populate method selector based on selected comparison
function populateMethodSelector(comparisonIndex) {
    const methodSelect = document.getElementById('method-select');
    methodSelect.innerHTML = '';
    
    const methods = Object.keys(comparisonData[comparisonIndex].methods);
    methods.forEach(method => {
        const option = document.createElement('option');
        option.value = method;
        option.textContent = method;
        methodSelect.appendChild(option);
    });
}

// Update comparison plot
function updateComparisonPlot(comparisonIndex, method) {
    const container = document.getElementById('method-comparison-container');
    const data = comparisonData[comparisonIndex];
    
    // Get Kompot and method data
    const kompotX = data.kompot.log2FoldChange;
    const kompotY = data.kompot.mahalanobis_distance;
    const kompotGenes = data.kompot.genes;
    
    const methodData = data.methods[method];
    let methodX = methodData.log2FoldChange.values;
    const methodGenes = methodData.log2FoldChange.genes;
    
    // Find a metric for y-axis - prefer pvalue, then padj, then statistic
    let methodY;
    let yLabel = "";
    
    if (methodData.pvalue) {
        methodY = methodData.pvalue.values.map(p => -Math.log10(p));
        yLabel = "-log10(p-value)";
    } else if (methodData.padj) {
        methodY = methodData.padj.values.map(p => -Math.log10(p));
        yLabel = "-log10(adjusted p-value)";
    } else if (methodData.statistic) {
        methodY = methodData.statistic.values;
        yLabel = "Statistic";
    } else {
        methodY = methodData.log2FoldChange.values;
        yLabel = "Log2 Fold Change";
    }
    
    // Create traces
    const trace1 = {
        x: kompotX,
        y: kompotY,
        mode: 'markers',
        type: 'scatter',
        name: 'Kompot',
        text: kompotGenes,
        marker: {
            size: 8,
            opacity: 0.7,
            color: 'blue'
        }
    };
    
    const trace2 = {
        x: methodX,
        y: methodY,
        mode: 'markers',
        type: 'scatter',
        name: method,
        text: methodGenes,
        marker: {
            size: 8,
            opacity: 0.7,
            color: 'red'
        }
    };
    
    // Create layout
    const layout = {
        title: `Method Comparison: Kompot vs ${method}`,
        xaxis: { title: 'Log2 Fold Change' },
        yaxis: { title: 'Kompot: Mahalanobis Dist. / ' + method + ': ' + yLabel },
        grid: { rows: 1, columns: 2, pattern: 'independent' },
        showlegend: true
    };
    
    // Plot
    Plotly.newPlot(container, [trace1, trace2], layout);
    
    // Add click event
    container.on('plotly_click', function(eventData) {
        const pointIndex = eventData.points[0].pointIndex;
        
        if (eventData.points[0].data.name === 'Kompot') {
            const geneName = kompotGenes[pointIndex];
            if (genePlotData && genePlotData[geneName]) {
                currentGene = geneName;
                generateGenePlots(geneName);
                highlightGeneInTable(geneName);
                
                // Update gene search box
                document.getElementById('gene-search').value = geneName;
            }
        } else {
            const geneName = methodGenes[pointIndex];
            if (genePlotData && genePlotData[geneName]) {
                currentGene = geneName;
                generateGenePlots(geneName);
                highlightGeneInTable(geneName);
                
                // Update gene search box
                document.getElementById('gene-search').value = geneName;
            }
        }
    });
}

// Highlight a gene in the table
function highlightGeneInTable(geneName) {
    const table = document.getElementById('genes-table');
    if (!table) return;
    
    // Remove highlight from all rows
    const rows = table.querySelectorAll('tbody tr');
    rows.forEach(row => row.classList.remove('selected-gene'));
    
    // Find and highlight the row with the given gene
    const targetRow = Array.from(rows).find(row => row.dataset.gene === geneName);
    if (targetRow) {
        targetRow.classList.add('selected-gene');
        
        // Scroll to the row if it's not visible
        const dataTable = $(table).DataTable();
        const rowIndex = dataTable.row(targetRow).index();
        const pageInfo = dataTable.page.info();
        const pageSize = pageInfo.length;
        const targetPage = Math.floor(rowIndex / pageSize);
        
        if (pageInfo.page !== targetPage) {
            dataTable.page(targetPage).draw(false);
        }
    }
}

// Highlight a gene in the volcano plot
function highlightGeneInPlot(geneName) {
    const container = document.getElementById('volcano-plot-container');
    if (!container) return;
    
    const data = geneData[currentComparisonIndex];
    const geneIndex = data.genes.findIndex(gene => gene.gene === geneName);
    
    if (geneIndex === -1) return;
    
    // Get the current data
    const plotData = container.data;
    
    // Create a new trace for the highlighted point
    const highlightTrace = {
        x: [data.genes[geneIndex][document.getElementById('volcano-x-metric').value]],
        y: [data.genes[geneIndex][document.getElementById('volcano-y-metric').value]],
        mode: 'markers',
        type: 'scatter',
        marker: {
            size: 12,
            color: 'red',
            line: {
                width: 2,
                color: 'darkred'
            }
        },
        text: [geneName],
        hoverinfo: 'text',
        showlegend: false
    };
    
    // Update the plot with both traces
    Plotly.react(container, [plotData[0], highlightTrace], container.layout);
}

// Helper function to format numbers
function formatNumber(num) {
    if (typeof num !== 'number') return num;
    return num.toFixed(4);
}

// Helper function to format axis labels
function formatAxisLabel(metric) {
    switch(metric) {
        case 'log2FoldChange': return 'Log2 Fold Change';
        case 'z_score': return 'Z-score';
        case 'mahalanobis_distance': return 'Mahalanobis Distance';
        case 'weighted_fold_change': return 'Weighted Fold Change';
        case 'fold_change_std': return 'Fold Change Std';
        case 'bidirectionality': return 'Bidirectionality';
        default: return metric;
    }
}