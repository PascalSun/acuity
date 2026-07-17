"""Schema visualization for foreign key relationships."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from talk2metadata.core.schema.schema import SchemaMetadata
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


def generate_html_visualization(
    schema: SchemaMetadata,
    output_path: str | Path,
    title: str = "Schema Visualization",
) -> Path:
    """Generate HTML visualization of schema with foreign key relationships.

    Args:
        schema: SchemaMetadata object
        output_path: Path to save HTML file
        title: Title for the visualization

    Returns:
        Path to generated HTML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate graph data
    nodes = []
    edges = []

    # Add nodes (tables)
    for table_name, table_meta in schema.tables.items():
        is_target = table_name == schema.target_table
        node = {
            "id": table_name,
            "label": table_name,
            "title": f"<b>{table_name}</b><br/>Rows: {table_meta.row_count:,}<br/>Columns: {len(table_meta.columns)}<br/>PK: {table_meta.primary_key or 'None'}",
            "group": "target" if is_target else "normal",
            "row_count": table_meta.row_count,
            "column_count": len(table_meta.columns),
            "primary_key": table_meta.primary_key or "None",
        }
        nodes.append(node)

    # Add edges (foreign keys)
    for fk in schema.foreign_keys:
        edge = {
            "from": fk.child_table,
            "to": fk.parent_table,
            "label": f"{fk.child_column}",
            "title": f"<b>FK</b>: {fk.child_table}.{fk.child_column} → {fk.parent_table}.{fk.parent_column}<br/>Coverage: {fk.coverage:.1%}",
            "coverage": fk.coverage,
        }
        edges.append(edge)

    # Calculate statistics
    total_rows = sum(t.row_count for t in schema.tables.values())
    avg_coverage = (
        sum(fk.coverage for fk in schema.foreign_keys) / len(schema.foreign_keys)
        if schema.foreign_keys
        else 0
    )
    target_incoming_fks = len(
        [fk for fk in schema.foreign_keys if fk.parent_table == schema.target_table]
    )

    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Talk2Metadata - {title}</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f7fa;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}

        .header {{
            padding: 20px 30px;
            border-bottom: 1px solid #e1e4e8;
        }}

        .header h1 {{
            font-size: 24px;
            color: #24292e;
            margin-bottom: 5px;
        }}

        .header .subtitle {{
            font-size: 14px;
            color: #586069;
        }}

        .stats {{
            display: flex;
            gap: 20px;
            padding: 15px 30px;
            background: #fafbfc;
            border-bottom: 1px solid #e1e4e8;
            flex-wrap: wrap;
        }}

        .stat {{
            flex: 1;
            min-width: 150px;
        }}

        .stat-label {{
            font-size: 12px;
            color: #586069;
            margin-bottom: 3px;
        }}

        .stat-value {{
            font-size: 20px;
            font-weight: 600;
            color: #24292e;
        }}

        .main {{
            display: flex;
            min-height: 600px;
        }}

        .sidebar {{
            width: 250px;
            border-right: 1px solid #e1e4e8;
            padding: 20px;
            background: #fafbfc;
        }}

        .sidebar h3 {{
            font-size: 13px;
            color: #24292e;
            margin-bottom: 10px;
            font-weight: 600;
        }}

        .control-section {{
            margin-bottom: 20px;
        }}

        .search-input {{
            width: 100%;
            padding: 8px 10px;
            border: 1px solid #d1d5da;
            border-radius: 6px;
            font-size: 13px;
        }}

        .search-input:focus {{
            outline: none;
            border-color: #0366d6;
            box-shadow: 0 0 0 3px rgba(3, 102, 214, 0.1);
        }}

        .btn {{
            width: 100%;
            padding: 7px 12px;
            margin-bottom: 6px;
            border: 1px solid #d1d5da;
            border-radius: 6px;
            background: white;
            color: #24292e;
            font-size: 13px;
            cursor: pointer;
            text-align: left;
        }}

        .btn:hover {{
            background: #f3f4f6;
            border-color: #c9ccd1;
        }}

        .table-list {{
            max-height: 300px;
            overflow-y: auto;
        }}

        .table-item {{
            padding: 8px 10px;
            margin-bottom: 4px;
            border-radius: 6px;
            font-size: 13px;
            cursor: pointer;
            background: white;
            border: 1px solid #d1d5da;
        }}

        .table-item:hover {{
            background: #f6f8fa;
        }}

        .table-item.target {{
            background: #0366d6;
            color: white;
            border-color: #0366d6;
            font-weight: 500;
        }}

        .graph-area {{
            flex: 1;
            padding: 20px;
            position: relative;
        }}

        #network {{
            width: 100%;
            height: 600px;
            border: 1px solid #e1e4e8;
            border-radius: 6px;
            background: #ffffff;
        }}

        .zoom-controls {{
            position: absolute;
            top: 30px;
            right: 30px;
            display: flex;
            gap: 5px;
            background: white;
            border: 1px solid #d1d5da;
            border-radius: 6px;
            padding: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}

        .zoom-btn {{
            width: 32px;
            height: 32px;
            border: none;
            background: transparent;
            font-size: 16px;
            cursor: pointer;
            border-radius: 4px;
            color: #586069;
        }}

        .zoom-btn:hover {{
            background: #f6f8fa;
            color: #24292e;
        }}

        .legend {{
            margin-top: 15px;
            padding: 15px;
            background: #f6f8fa;
            border-radius: 6px;
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            font-size: 13px;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .legend-box {{
            width: 20px;
            height: 20px;
            border-radius: 3px;
            border: 1px solid #d1d5da;
        }}

        .legend-line {{
            width: 30px;
            height: 3px;
            border-radius: 2px;
        }}

        .details {{
            padding: 30px;
            background: #fafbfc;
            border-top: 1px solid #e1e4e8;
        }}

        .details h2 {{
            font-size: 18px;
            color: #24292e;
            margin-bottom: 20px;
        }}

        .table-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 15px;
        }}

        .table-card {{
            background: white;
            border: 1px solid #d1d5da;
            border-radius: 6px;
            padding: 16px;
        }}

        .table-card.target {{
            border-color: #0366d6;
            border-width: 2px;
        }}

        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            padding-bottom: 12px;
            border-bottom: 1px solid #e1e4e8;
        }}

        .card-title {{
            font-size: 16px;
            font-weight: 600;
            color: #24292e;
        }}

        .target-badge {{
            padding: 3px 8px;
            background: #0366d6;
            color: white;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 500;
        }}

        .card-info {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 12px;
        }}

        .info-row {{
            font-size: 13px;
        }}

        .info-label {{
            color: #586069;
            margin-bottom: 3px;
        }}

        .info-value {{
            color: #24292e;
            font-weight: 500;
        }}

        .pk-value {{
            color: #d73a49;
            font-family: 'SFMono-Regular', Consolas, monospace;
            font-size: 12px;
            background: #ffeef0;
            padding: 2px 6px;
            border-radius: 3px;
        }}

        .fk-section {{
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px solid #e1e4e8;
        }}

        .fk-title {{
            font-size: 12px;
            color: #586069;
            margin-bottom: 6px;
            font-weight: 500;
        }}

        .fk-item {{
            padding: 6px 10px;
            margin-bottom: 4px;
            background: #f6f8fa;
            border-radius: 4px;
            font-size: 12px;
            font-family: 'SFMono-Regular', Consolas, monospace;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .fk-item.low-coverage {{
            background: #ffeef0;
        }}

        .coverage {{
            font-size: 11px;
            padding: 2px 6px;
            border-radius: 3px;
            font-weight: 500;
        }}

        .coverage.high {{
            background: #dcffe4;
            color: #22863a;
        }}

        .coverage.low {{
            background: #ffeef0;
            color: #d73a49;
        }}

        .columns-section {{
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px solid #e1e4e8;
        }}

        .columns-title {{
            font-size: 12px;
            color: #586069;
            margin-bottom: 6px;
            font-weight: 500;
        }}

        .columns-text {{
            font-size: 12px;
            color: #586069;
            line-height: 1.8;
            max-height: 120px;
            overflow-y: auto;
        }}

        .column-tag {{
            display: inline-block;
            padding: 3px 8px;
            margin: 2px 3px 2px 0;
            background: #f1f8ff;
            border: 1px solid #c8e1ff;
            border-radius: 3px;
            font-size: 11px;
            font-family: 'SFMono-Regular', Consolas, monospace;
            color: #0366d6;
        }}

        ::-webkit-scrollbar {{
            width: 6px;
            height: 6px;
        }}

        ::-webkit-scrollbar-track {{
            background: #f6f8fa;
        }}

        ::-webkit-scrollbar-thumb {{
            background: #c9ccd1;
            border-radius: 3px;
        }}

        ::-webkit-scrollbar-thumb:hover {{
            background: #959da5;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Talk2Metadata - {title}</h1>
            <div class="subtitle">Database schema with foreign key relationships</div>
        </div>

        <div class="stats">
            <div class="stat">
                <div class="stat-label">Target Table</div>
                <div class="stat-value">{schema.target_table}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Tables</div>
                <div class="stat-value">{len(schema.tables)}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Foreign Keys</div>
                <div class="stat-value">{len(schema.foreign_keys)}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Total Rows</div>
                <div class="stat-value">{total_rows:,}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Avg Coverage</div>
                <div class="stat-value">{avg_coverage:.1%}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Connected</div>
                <div class="stat-value">{target_incoming_fks}/{len(schema.tables)-1}</div>
            </div>
        </div>

        <div class="main">
            <div class="sidebar">
                <div class="control-section">
                    <h3>Search</h3>
                    <input type="text" class="search-input" id="searchInput" placeholder="Search tables..." />
                </div>

                <div class="control-section">
                    <h3>Controls</h3>
                    <button class="btn" onclick="fitView()">Fit to View</button>
                    <button class="btn" onclick="resetLayout()">Reset Layout</button>
                    <button class="btn" onclick="focusTarget()">Focus Target</button>
                </div>

                <div class="control-section">
                    <h3>Tables</h3>
                    <div class="table-list" id="tableList">
                        {_generate_table_list(schema)}
                    </div>
                </div>
            </div>

            <div class="graph-area">
                <div class="zoom-controls">
                    <button class="zoom-btn" onclick="zoomIn()">+</button>
                    <button class="zoom-btn" onclick="zoomOut()">−</button>
                    <button class="zoom-btn" onclick="fitView()">⊙</button>
                </div>
                <div id="network"></div>
                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-box" style="background: #0366d6;"></div>
                        <span>Target Table</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-box" style="background: #6a737d;"></div>
                        <span>Dimension Table</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-line" style="background: #28a745;"></div>
                        <span>High Coverage (≥90%)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-line" style="background: #d73a49;"></div>
                        <span>Low Coverage (&lt;90%)</span>
                    </div>
                </div>
            </div>
        </div>

        <div class="details">
            <h2>Table Details</h2>
            <div class="table-grid">
                {_generate_table_cards(schema)}
            </div>
        </div>
    </div>

    <script type="text/javascript">
        var nodes = new vis.DataSet({json.dumps(nodes)});
        var edges = new vis.DataSet({json.dumps(edges)});

        var container = document.getElementById('network');
        var data = {{
            nodes: nodes,
            edges: edges
        }};

        var options = {{
            nodes: {{
                shape: 'box',
                font: {{
                    size: 14,
                    face: 'Arial'
                }},
                borderWidth: 2,
                shadow: false,
                margin: 10,
                widthConstraint: {{
                    minimum: 100,
                    maximum: 200
                }},
                heightConstraint: {{
                    minimum: 50
                }}
            }},
            edges: {{
                arrows: {{
                    to: {{
                        enabled: true,
                        scaleFactor: 1
                    }}
                }},
                font: {{
                    size: 11,
                    align: 'middle'
                }},
                smooth: {{
                    enabled: true,
                    type: 'cubicBezier',
                    roundness: 0.2
                }},
                width: 2,
                shadow: false
            }},
            physics: {{
                enabled: true,
                stabilization: {{
                    iterations: 300
                }},
                barnesHut: {{
                    gravitationalConstant: -3000,
                    centralGravity: 0.3,
                    springLength: 200,
                    springConstant: 0.04,
                    damping: 0.12
                }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 200,
                zoomView: true,
                dragView: true
            }}
        }};

        // Style nodes
        nodes.forEach(function(node) {{
            if (node.group === 'target') {{
                node.color = {{
                    border: '#0366d6',
                    background: '#0366d6',
                    highlight: {{ border: '#044289', background: '#044289' }},
                    hover: {{ border: '#0366d6', background: '#0366d6' }}
                }};
                node.font = {{ size: 15, bold: true, color: '#ffffff' }};
            }} else {{
                node.color = {{
                    border: '#586069',
                    background: '#6a737d',
                    highlight: {{ border: '#24292e', background: '#24292e' }},
                    hover: {{ border: '#586069', background: '#586069' }}
                }};
                node.font = {{ size: 13, color: '#ffffff' }};
            }}
        }});

        // Style edges
        edges.forEach(function(edge) {{
            if (edge.coverage >= 0.9) {{
                edge.color = {{ color: '#28a745', highlight: '#28a745' }};
            }} else {{
                edge.color = {{ color: '#d73a49', highlight: '#d73a49' }};
                edge.dashes = [5, 5];
            }}
        }});

        var network = new vis.Network(container, data, options);

        // Disable physics after stabilization
        network.on("stabilizationIterationsDone", function () {{
            setTimeout(function() {{
                network.setOptions({{ physics: false }});
            }}, 500);
        }});

        // Click to scroll to card
        network.on("click", function (params) {{
            if (params.nodes.length > 0) {{
                var nodeId = params.nodes[0];
                var element = document.getElementById('card-' + nodeId);
                if (element) {{
                    element.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                    element.style.boxShadow = '0 0 0 2px #0366d6';
                    setTimeout(function() {{
                        element.style.boxShadow = '';
                    }}, 1000);
                }}
            }}
        }});

        // Search functionality
        document.getElementById('searchInput').addEventListener('input', function(e) {{
            var term = e.target.value.toLowerCase();
            document.querySelectorAll('.table-item').forEach(function(item) {{
                var name = item.textContent.toLowerCase();
                item.style.display = name.includes(term) ? 'block' : 'none';
            }});

            if (term) {{
                var matching = [];
                nodes.forEach(function(node) {{
                    if (node.id.toLowerCase().includes(term)) matching.push(node.id);
                }});
                network.selectNodes(matching);
            }} else {{
                network.unselectAll();
            }}
        }});

        // Table list clicks
        document.querySelectorAll('.table-item').forEach(function(item) {{
            item.addEventListener('click', function() {{
                var tableId = this.getAttribute('data-table');
                network.focus(tableId, {{ scale: 1.3, animation: {{ duration: 500 }} }});
                network.selectNodes([tableId]);
            }});
        }});

        // Control functions
        function zoomIn() {{
            var scale = network.getScale();
            network.moveTo({{ scale: scale * 1.2, animation: {{ duration: 200 }} }});
        }}

        function zoomOut() {{
            var scale = network.getScale();
            network.moveTo({{ scale: scale * 0.8, animation: {{ duration: 200 }} }});
        }}

        function fitView() {{
            network.fit({{ animation: {{ duration: 500 }} }});
        }}

        function resetLayout() {{
            network.setOptions({{ physics: true }});
            setTimeout(function() {{ network.stabilize(); }}, 100);
        }}

        function focusTarget() {{
            network.focus('{schema.target_table}', {{ scale: 1.3, animation: {{ duration: 500 }} }});
            network.selectNodes(['{schema.target_table}']);
        }}

        // Auto-focus target after load
        setTimeout(focusTarget, 1000);
    </script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"Generated schema visualization at {output_path}")
    return output_path


def _generate_table_list(schema: SchemaMetadata) -> str:
    """Generate HTML for table list in sidebar."""
    items = []

    # Target table first
    if schema.target_table in schema.tables:
        items.append(
            f'<div class="table-item target" data-table="{schema.target_table}">{schema.target_table}</div>'
        )

    # Other tables
    for table_name in sorted(schema.tables.keys()):
        if table_name != schema.target_table:
            items.append(
                f'<div class="table-item" data-table="{table_name}">{table_name}</div>'
            )

    return "\n".join(items)


def _generate_table_cards(schema: SchemaMetadata) -> str:
    """Generate HTML for table detail cards."""
    cards = []

    # Target table first
    table_names = [schema.target_table] + sorted(
        [t for t in schema.tables.keys() if t != schema.target_table]
    )

    for table_name in table_names:
        if table_name not in schema.tables:
            continue

        table_meta = schema.tables[table_name]
        is_target = table_name == schema.target_table

        # Get foreign keys
        outgoing_fks = [
            fk for fk in schema.foreign_keys if fk.child_table == table_name
        ]
        incoming_fks = [
            fk for fk in schema.foreign_keys if fk.parent_table == table_name
        ]

        # Build FK HTML
        fk_html = ""
        if outgoing_fks:
            fk_html += '<div class="fk-section"><div class="fk-title">Outgoing Foreign Keys</div>'
            for fk in outgoing_fks:
                coverage_class = "high" if fk.coverage >= 0.9 else "low"
                item_class = "low-coverage" if fk.coverage < 0.9 else ""
                fk_html += f"""
                    <div class="fk-item {item_class}">
                        <span>{fk.child_column} → {fk.parent_table}.{fk.parent_column}</span>
                        <span class="coverage {coverage_class}">{fk.coverage:.1%}</span>
                    </div>
                """
            fk_html += "</div>"

        if incoming_fks:
            fk_html += '<div class="fk-section"><div class="fk-title">Incoming Foreign Keys</div>'
            for fk in incoming_fks:
                coverage_class = "high" if fk.coverage >= 0.9 else "low"
                fk_html += f"""
                    <div class="fk-item">
                        <span>{fk.child_table}.{fk.child_column} → {fk.parent_column}</span>
                        <span class="coverage {coverage_class}">{fk.coverage:.1%}</span>
                    </div>
                """
            fk_html += "</div>"

        # Columns as tags
        columns_html = "".join(
            [
                f'<span class="column-tag">{col}</span>'
                for col in table_meta.columns.keys()
            ]
        )

        card = f"""
            <div class="table-card {'target' if is_target else ''}" id="card-{table_name}">
                <div class="card-header">
                    <div class="card-title">{table_name}</div>
                    {f'<div class="target-badge">TARGET</div>' if is_target else ''}
                </div>
                <div class="card-info">
                    <div class="info-row">
                        <div class="info-label">Rows</div>
                        <div class="info-value">{table_meta.row_count:,}</div>
                    </div>
                    <div class="info-row">
                        <div class="info-label">Columns</div>
                        <div class="info-value">{len(table_meta.columns)}</div>
                    </div>
                </div>
                <div class="info-row">
                    <div class="info-label">Primary Key</div>
                    <div class="pk-value">{table_meta.primary_key or "None"}</div>
                </div>
                {fk_html}
                <div class="columns-section">
                    <div class="columns-title">Columns</div>
                    <div class="columns-text">{columns_html}</div>
                </div>
            </div>
        """
        cards.append(card)

    return "\n".join(cards)


def validate_schema(schema: SchemaMetadata) -> Dict[str, List[str]]:
    """Validate schema and return errors/warnings.

    Args:
        schema: SchemaMetadata to validate

    Returns:
        Dict with 'errors' and 'warnings' lists
    """
    errors = []
    warnings = []

    # Check if target table exists
    if schema.target_table not in schema.tables:
        errors.append(f"Target table '{schema.target_table}' not found in tables")

    # Check foreign keys
    for fk in schema.foreign_keys:
        # Check child table exists
        if fk.child_table not in schema.tables:
            errors.append(f"FK references non-existent child table: {fk.child_table}")

        # Check parent table exists
        if fk.parent_table not in schema.tables:
            errors.append(f"FK references non-existent parent table: {fk.parent_table}")

        # Check child column exists
        if fk.child_table in schema.tables:
            if fk.child_column not in schema.tables[fk.child_table].columns:
                errors.append(
                    f"FK references non-existent child column: "
                    f"{fk.child_table}.{fk.child_column}"
                )

        # Check parent column exists
        if fk.parent_table in schema.tables:
            if fk.parent_column not in schema.tables[fk.parent_table].columns:
                errors.append(
                    f"FK references non-existent parent column: "
                    f"{fk.parent_table}.{fk.parent_column}"
                )

        # Check coverage
        if fk.coverage < 0.5:
            errors.append(
                f"FK {fk.child_table}.{fk.child_column} → "
                f"{fk.parent_table}.{fk.parent_column} has very low coverage "
                f"({fk.coverage:.1%}), may be incorrect"
            )
        elif fk.coverage < 0.9:
            warnings.append(
                f"FK {fk.child_table}.{fk.child_column} → "
                f"{fk.parent_table}.{fk.parent_column} has low coverage "
                f"({fk.coverage:.1%})"
            )

    # Check for tables with no relationships
    table_names = set(schema.tables.keys())
    related_tables = set()
    for fk in schema.foreign_keys:
        related_tables.add(fk.child_table)
        related_tables.add(fk.parent_table)

    isolated_tables = table_names - related_tables
    if len(isolated_tables) > 1:  # Allow target table to be isolated
        warnings.append(
            f"Found {len(isolated_tables)} isolated tables (no FK relationships): "
            f"{', '.join(isolated_tables)}"
        )

    return {"errors": errors, "warnings": warnings}
