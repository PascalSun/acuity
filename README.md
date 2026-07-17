# Acuity — Artifact

Artifact for **"A Benchmark for Every Database: Annotation-Free, Coverage-Calibrated Evaluation of Conjunctive Text-to-SQL with Acuity"** (submitted to PVLDB Vol. 20, Experiment, Analysis & Benchmark track).

Acuity turns any relational schema into an annotation-free, coverage-calibrated Text-to-SQL benchmark for the conjunctive filter–join (record-localization) workload: it derives schema-feasible structure classes from FK topology, apportions per-class quotas, synthesizes execution-verified SQL by anchor-row sampling, and releases every pair in dual form (a provably faithful canonical question + a gate-certified natural paraphrase).

## Contents

| Directory | What it is |
|---|---|
| `generator/` | Full generator + evaluation-harness source (`talk2metadata` Python package) |
| `benchmarks/` | The released dual-form benchmarks: Spider (6,599 pairs, 158 DBs), BIRD (3,000 pairs, 72 DBs), WAMEX (500), ACYWA (500) — every pair carries `question` (natural), `canonical_question`, `sql`, `answer_row_ids`, strategy/tier labels, and provenance |
| `records/` | All >103,000 per-question model evaluations: `spider_natural/` (6 closed models), `spider_canonical/` (5 models), `spider_openweight/` (OmniSQL-7B + Qwen2.5-Coder-7B, natural + canonical), `bird_natural/`, `bird_canonical/`, `wamex_controlled/`, `acywa_controlled/`, `{wamex,acywa}_operational/` |
| `results/` | Every aggregated analysis JSON behind every number printed in the paper |
| `scripts/` | Analysis + figure scripts (offline; run against `records/` and `results/`) |
| `prompts/` | The three prompts, verbatim: evaluation, paraphraser, faithfulness judge |

**Data availability.** **WAMEX ships in full**: the complete SQLite snapshot (115,174-row `wamex_reports` hub + 8 satellite tables, 407 MB / 69 MB compressed) is attached to the [`wamex-db-v1` release](https://github.com/PascalSun/acuity/releases/tag/wamex-db-v1) — decompress with `zstd -d`, SHA-256 in the release notes. With it, WAMEX benchmark generation, Gate 1 re-execution, and both evaluation regimes (Section 7) are reproducible end-to-end:

```bash
gh release download wamex-db-v1 --repo PascalSun/acuity && zstd -d wamex.sqlite.zst
python -m talk2metadata.cli analysis wamex generate-qa --db wamex.sqlite --mode flexbench --pairs 500
```

**ACYWA** is held under a government data-sharing agreement covering child-and-youth wellbeing statistics and cannot be redistributed: its released pairs and records contain question text, SQL, answer *row IDs*, provenance, and per-question verdicts, with raw row contents redacted.

## Table/Figure → script manifest

| Paper element | Data | Script |
|---|---|---|
| Table 2 (E1 saturation) | `results/e2_FINAL_*.json`, `benchmarks/*/generation_report.json` | `scripts/e2_analyze.py` |
| Table 3 (E2 resolution) | `results/e2_FINAL_{spider,bird}.json`, `results/e2_final_matched4.json`, `results/e2_std_*.json` | `scripts/e2_analyze.py`, `scripts/e2_analyze_matched4.py` |
| Composition ablation (Result 1) | `results/quota_matched_ablation_spider.json` | `scripts/quota_matched_ablation.py` |
| BIRD power subsample (Result 1) | `results/bird_power_subsample.json` | inline in commit history; recompute via `scripts/e2_analyze.py` on n=425 draws |
| Figure 2 (difficulty surface) | `results/surface_pattern_x_npred*.json` | `scripts/make_fig_surface.py` (3D), `scripts/make_fig_surface2d.py` (flat numeric view) |
| Figure 7 (per-class sawtooth) | `results/e2_FINAL_spider.json` | `scripts/make_fig3_lockstep.py` |
| Figure 8 (dual-form dumbbell) | `results/dualform_certified_permodel.json`, `results/dualform_paired_table.json`, `results/openweight_gap_analysis.json` | `scripts/make_fig_dumbbell.py`, `scripts/dualform_certified_permodel.py` |
| Figure 9 (failure-mode taxonomy) | `results/error_taxonomy.json` | `scripts/error_taxonomy.py`, `scripts/make_fig_errors.py` |
| Register control (Result 3) | `records/spider_openweight/../e2_rewrite*` inputs | `scripts/rewrite_htier_set.py` |
| Open-weight pair (Result 3) | `results/openweight_gap_analysis.json` | `scripts/openweight_gap_analysis.py` |
| Table 4/5 (real schemas) | `results/regimes_clean_final.json`, `results/tab_difficulty_operational.json`, `results/e2_realdb_*.json` | `scripts/tab_difficulty_operational.py` |
| Cross-vendor judge check | `results/crossvendor_judge_spider.json` | `scripts/crossvendor_judge.py` |
| Cost accounting | `results/cost_reconstruction.json` | `scripts/cost_reconstruction.py` |
| Memorization probe | `results/contamination_probe_full.json` | see `generator/` probe module |

## Reproducing

```bash
# environment
pip install -e generator/            # or: uv sync

# regenerate a benchmark for a Spider database (seeded, deterministic SQL side)
python -m talk2metadata.cli analysis spider generate-qa \
  --db-dir <spider sqlite dir> --mode flexbench --pairs-per-db 50

# evaluate any OpenAI-compatible model on a released set
python scripts/e2_resolution_eval.py \
  --benchmark spider --set-dir benchmarks/spider --set-name myrun \
  --models "openai:MODEL" --base-url <endpoint>/v1 --output-dir out/

# aggregate: spread / ceiling / separable pairs (cluster bootstrap + BH)
python scripts/e2_analyze.py --input out/ --benchmark spider
```

Generation is seeded on the SQL side; LLM paraphrase calls are provider-dependent (model IDs and evaluation timestamps ship in the per-pair provenance blocks).

## License

Code: MIT. Released benchmark data and per-question records: CC BY 4.0.
