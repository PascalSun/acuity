"""Render quota/shortfall summaries as standalone PGFPlots figures."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def load_generation_summary(summary_path: Path) -> dict[str, Any]:
    """Load an aggregated generation summary JSON file."""

    return json.loads(Path(summary_path).read_text())


def build_quota_shortfall_figure_tex(
    summary: dict[str, Any], title: str = "Quota Fulfillment and Shortfall Reasons"
) -> str:
    """Build a standalone TeX document with quota/shortfall plots."""

    strategy_rows = summary.get("per_strategy", [])
    shortfall_reasons = summary.get("shortfall_reason_counts", {})

    target_coords = " ".join(
        f"({row['strategy']},{int(row['target_quota_total'])})" for row in strategy_rows
    )
    realized_coords = " ".join(
        f"({row['strategy']},{int(row['accepted_total'])})" for row in strategy_rows
    )
    reason_labels = list(shortfall_reasons.keys()) or ["none"]
    reason_coords = " ".join(
        f"({int(shortfall_reasons.get(label, 0))},{_escape_pgf_label(label)})"
        for label in reason_labels
    )

    symbolic_x = ",".join(_escape_pgf_label(row["strategy"]) for row in strategy_rows)
    symbolic_y = ",".join(_escape_pgf_label(label) for label in reason_labels)

    headline = (
        f"Reports={int(summary.get('report_count', 0))}, "
        f"target={int(summary.get('target_total', 0))}, "
        f"realized={int(summary.get('realized_total', 0))}, "
        f"shortfall={int(summary.get('shortfall_total', 0))}, "
        f"fulfillment={float(summary.get('overall_fulfillment_rate', 0.0)) * 100:.1f}%"
    )

    return f"""\\documentclass[tikz,border=4pt]{{standalone}}
\\usepackage{{pgfplots}}
\\pgfplotsset{{compat=1.18}}
\\usepgfplotslibrary{{groupplots}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{lmodern}}

\\definecolor{{FlexBlue}}{{HTML}}{{355C7D}}
\\definecolor{{FlexTeal}}{{HTML}}{{2A9D8F}}
\\definecolor{{FlexGold}}{{HTML}}{{E9C46A}}
\\definecolor{{FlexRed}}{{HTML}}{{D1495B}}
\\definecolor{{FlexGray}}{{HTML}}{{5C677D}}

\\begin{{document}}
\\begin{{tikzpicture}}
\\begin{{groupplot}}[
    group style={{group size=2 by 1, horizontal sep=2.1cm}},
    width=9.2cm,
    height=6.8cm,
    ymajorgrids=true,
    grid style={{dashed, gray!25}},
    tick align=outside,
    title style={{font=\\bfseries\\small}},
    label style={{font=\\small}},
    tick label style={{font=\\scriptsize}},
    legend style={{font=\\scriptsize, draw=none, fill=none}},
    every axis title shift=2pt,
]
\\nextgroupplot[
    title={{{_escape_latex(title)}}},
    symbolic x coords={{{symbolic_x}}},
    xtick=data,
    x tick label style={{rotate=45, anchor=east}},
    ymin=0,
    ymax={_strategy_ymax(summary)},
    ylabel={{QA pairs}},
    xlabel={{Strategy code}},
    legend columns=2,
    legend to name=quotafiglegend,
]
\\addplot[ybar, bar width=7pt, fill=FlexBlue, draw=none] coordinates {{{target_coords}}};
\\addlegendentry{{Target quota}}
\\addplot[ybar, bar width=7pt, fill=FlexTeal, draw=none] coordinates {{{realized_coords}}};
\\addlegendentry{{Realized}}

\\node[anchor=north west, align=left, font=\\scriptsize, text=FlexGray]
at (rel axis cs:0.02,0.98) {{{_escape_latex(headline)}}};

\\nextgroupplot[
    title={{Shortfall Reason Counts}},
    xbar,
    symbolic y coords={{{symbolic_y}}},
    ytick=data,
    xmin=0,
    xmax={_reason_xmax(summary)},
    xlabel={{Count}},
    enlarge y limits=0.2,
    nodes near coords,
    nodes near coords align={{horizontal}},
]
\\addplot[fill=FlexGold, draw=none] coordinates {{{reason_coords}}};
\\end{{groupplot}}
\\node[anchor=north] at ($(group c1r1.south)!0.5!(group c2r1.south)+(0,-1.35cm)$) {{\\pgfplotslegendfromname{{quotafiglegend}}}};
\\end{{tikzpicture}}
\\end{{document}}
"""


def write_quota_shortfall_figure(
    summary: dict[str, Any], output_tex_path: Path, title: str | None = None
) -> Path:
    """Write the standalone TeX source for quota/shortfall plots."""

    tex = build_quota_shortfall_figure_tex(
        summary, title=title or "Quota Fulfillment and Shortfall Reasons"
    )
    output_tex_path.parent.mkdir(parents=True, exist_ok=True)
    output_tex_path.write_text(tex)
    return output_tex_path


def compile_tex_to_pdf(tex_path: Path) -> Path:
    """Compile a standalone TeX figure into PDF."""

    if shutil.which("pdflatex") is None:
        raise RuntimeError("pdflatex is not available on PATH")

    tex_path = Path(tex_path)
    output_dir = tex_path.parent
    cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-output-directory",
        str(output_dir),
        str(tex_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return tex_path.with_suffix(".pdf")


def _strategy_ymax(summary: dict[str, Any]) -> int:
    max_target = max(
        [
            int(row.get("target_quota_total", 0))
            for row in summary.get("per_strategy", [])
        ]
        + [1]
    )
    max_realized = max(
        [int(row.get("accepted_total", 0)) for row in summary.get("per_strategy", [])]
        + [1]
    )
    return max(max_target, max_realized, 1) + 1


def _reason_xmax(summary: dict[str, Any]) -> int:
    max_reason = max(
        [int(v) for v in summary.get("shortfall_reason_counts", {}).values()] + [0]
    )
    return max(max_reason, 1) + 1


def _escape_latex(value: str) -> str:
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
    }
    escaped = value
    for old, new in replacements.items():
        escaped = escaped.replace(old, new)
    return escaped


def _escape_pgf_label(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}").replace("_", "\\_").replace(",", "{,}")
    )
