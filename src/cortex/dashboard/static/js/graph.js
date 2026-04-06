/* Cortex Knowledge Graph — Cytoscape.js Visualization */

let cy = null;

const TYPE_COLORS = {
    decision: '#2563eb',
    lesson:   '#059669',
    fix:      '#dc2626',
    session:  '#d97706',
    research: '#7c3aed',
    source:   '#4f46e5',
    synthesis:'#db2777',
    idea:     '#16a34a',
    'entity:technology': '#0891b2',
    'entity:project':    '#6d28d9',
    'entity:pattern':    '#ca8a04',
    'entity:concept':    '#64748b',
};

const REL_STYLES = {
    causedBy:    { lineColor: '#ef4444', lineStyle: 'solid' },
    ledTo:       { lineColor: '#ef4444', lineStyle: 'dashed' },
    contradicts: { lineColor: '#dc2626', lineStyle: 'dotted' },
    supports:    { lineColor: '#059669', lineStyle: 'solid' },
    supersedes:  { lineColor: '#6b7280', lineStyle: 'dashed' },
    dependsOn:   { lineColor: '#d97706', lineStyle: 'solid' },
    implements:  { lineColor: '#2563eb', lineStyle: 'solid' },
    mentions:    { lineColor: '#9ca3af', lineStyle: 'dotted' },
};

var currentOffset = 0;

function initGraph(elements) {
    cy = cytoscape({
        container: document.getElementById('cy'),
        elements: elements,
        style: [
            {
                selector: 'node',
                style: {
                    'label': 'data(label)',
                    'background-color': function(ele) {
                        return TYPE_COLORS[ele.data('type')] || '#6b7280';
                    },
                    'color': '#3d3833',
                    'font-size': '10px',
                    'text-valign': 'bottom',
                    'text-margin-y': 5,
                    'width': function(ele) {
                        return Math.max(20, Math.min(50, 20 + ele.degree() * 5));
                    },
                    'height': function(ele) {
                        return Math.max(20, Math.min(50, 20 + ele.degree() * 5));
                    },
                    'border-width': 2,
                    'border-color': '#e5e0da',
                },
            },
            {
                selector: 'edge',
                style: {
                    'curve-style': 'bezier',
                    'target-arrow-shape': 'triangle',
                    'arrow-scale': 0.8,
                    'width': 1.5,
                    'line-color': function(ele) {
                        var rs = REL_STYLES[ele.data('rel_type')];
                        return rs ? rs.lineColor : '#9ca3af';
                    },
                    'target-arrow-color': function(ele) {
                        var rs = REL_STYLES[ele.data('rel_type')];
                        return rs ? rs.lineColor : '#9ca3af';
                    },
                    'line-style': function(ele) {
                        var rs = REL_STYLES[ele.data('rel_type')];
                        return rs ? rs.lineStyle : 'solid';
                    },
                    'label': 'data(rel_type)',
                    'font-size': '8px',
                    'color': '#9b9590',
                    'text-rotation': 'autorotate',
                },
            },
            {
                selector: 'node:selected',
                style: {
                    'border-width': 3,
                    'border-color': '#2563eb',
                    'background-color': '#dbeafe',
                },
            },
        ],
        layout: {
            name: 'cose',
            idealEdgeLength: 120,
            nodeOverlap: 20,
            refresh: 20,
            fit: true,
            padding: 30,
            randomize: false,
            componentSpacing: 100,
            nodeRepulsion: 8000,
            edgeElasticity: 100,
            nestingFactor: 5,
            gravity: 80,
            numIter: 1000,
        },
    });

    // Click handler
    cy.on('tap', 'node', function(evt) {
        var node = evt.target;
        var detail = document.getElementById('node-detail');
        var info = document.getElementById('node-info');
        detail.style.display = 'block';
        info.innerHTML =
            '<strong>' + node.data('label') + '</strong><br>' +
            '<span class="badge badge-' + node.data('type') + '">' + node.data('type') + '</span><br>' +
            'Project: ' + (node.data('project') || '—') + '<br>' +
            'Connections: ' + node.degree() + '<br>' +
            '<a href="/documents/' + node.data('id') + '">View details &rarr;</a>';
    });
}

function loadGraph(append) {
    var project = document.getElementById('gf-project').value;
    var type = document.getElementById('gf-type').value;
    var limit = 500;
    var offset = append ? currentOffset : 0;
    var url = '/api/graph-data?project=' + encodeURIComponent(project) +
              '&doc_type=' + encodeURIComponent(type) +
              '&limit=' + limit + '&offset=' + offset;

    fetch(url)
        .then(function(r) { return r.json(); })
        .then(function(data) {
            var elements = data.nodes.concat(data.edges);
            if (!append || !cy) {
                if (cy) { cy.destroy(); }
                initGraph(elements);
            } else {
                cy.add(elements);
                cy.layout({name: 'cose', animate: false, fit: true}).run();
            }
            currentOffset = offset + data.nodes.length;
            var btn = document.getElementById('load-more');
            if (btn) {
                btn.style.display = (currentOffset < data.total) ? 'inline-block' : 'none';
            }
        })
        .catch(function(err) { console.error('Failed to load graph:', err); });
}

// Auto-load on page ready
document.addEventListener('DOMContentLoaded', function() { loadGraph(false); });
