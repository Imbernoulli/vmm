#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


FIGURES = {
    "merge_landscape": Path("results/digits_merge/figures/merge_landscape.png"),
    "per_task_basin_overlay": Path("results/digits_merge/figures/per_task_basin_overlay.png"),
    "lambda_sweep": Path("results/digits_merge/figures/lambda_sweep.png"),
    "method_overlay": Path("results/digits_merge/figures/method_overlay.png"),
    "interference_heatmap": Path("results/digits_merge/figures/interference_heatmap.png"),
    "pairwise_heatmaps": Path("results/digit_pairwise_experts/pairwise_heatmaps.png"),
    "conflict_vs_drop": Path("results/digit_pairwise_experts/conflict_vs_drop.png"),
    "layer_conflict_atlas": Path("results/digit_pairwise_experts/layer_conflict_atlas.png"),
    "alignment": Path("results/alignment_barrier/interpolation_alignment.png"),
    "cifar_landscape": Path("results/cifar_merge/figures/merge_landscape.png"),
    "cifar_methods": Path("results/cifar_merge/figures/method_overlay.png"),
    "cifar_lambda": Path("results/cifar_merge/figures/lambda_sweep.png"),
    "cifar_interference": Path("results/cifar_merge/figures/interference_heatmap.png"),
    "vit_landscape": Path("results/cifar100_vit_merge/figures/merge_landscape.png"),
    "vit_methods": Path("results/cifar100_vit_merge/figures/method_overlay.png"),
    "vit_lambda": Path("results/cifar100_vit_merge/figures/lambda_sweep.png"),
    "vit_interference": Path("results/cifar100_vit_merge/figures/interference_heatmap.png"),
    "vit_pca": Path("results/cifar100_vit_merge/figures/pca_task_vectors.png"),
    "pre_vit_landscape": Path("results/pretrained_vit_transfer_merge/figures/merge_landscape.png"),
    "pre_vit_methods": Path("results/pretrained_vit_transfer_merge/figures/method_overlay.png"),
    "pre_vit_lambda": Path("results/pretrained_vit_transfer_merge/figures/lambda_sweep.png"),
    "pre_vit_interference": Path("results/pretrained_vit_transfer_merge/figures/interference_heatmap.png"),
    "qwen_path": Path("results/qwen_path_sweep/qwen_path_sweep.png"),
    "qwen_deltas": Path("results/qwen_path_sweep/delta_norms.png"),
    "qwen_gsm8k": Path("results/qwen_gsm8k_slice/gsm8k_exact_match.png"),
    "qwen_mmlu": Path("results/qwen_mmlu_slice/mmlu_accuracy.png"),
    "qwen_humaneval": Path("results/qwen_humaneval_nll_slice/humaneval_nll.png"),
    "qwen_safety": Path("results/qwen_safety_refusal_slice/safety_refusal_nll.png"),
    "qwen_multi_grid": Path("results/qwen_multi_expert_merge/figures/merge_grid.png"),
    "qwen_multi_path": Path("results/qwen_multi_expert_merge/figures/diagonal_path.png"),
    "qwen_multi_conflict": Path("results/qwen_multi_expert_merge/figures/pairwise_conflict.png"),
}


def read_csv(path: Path) -> list[dict[str, object]]:
    return pd.read_csv(path).where(pd.notnull(pd.read_csv(path)), None).to_dict(orient="records")


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def copy_assets(out_dir: Path) -> dict[str, str]:
    assets = out_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for name, source in FIGURES.items():
        if not source.exists():
            raise FileNotFoundError(source)
        target = assets / f"{name}.png"
        shutil.copyfile(source, target)
        paths[name] = f"assets/{target.name}"
    return paths


def load_data() -> dict[str, object]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "digits_summary": read_json(Path("results/digits_merge/summary.json")),
        "alignment_summary": read_json(Path("results/alignment_barrier/summary.json")),
        "cifar_summary": read_json(Path("results/cifar_merge/summary.json")),
        "vit_summary": read_json(Path("results/cifar100_vit_merge/summary.json")),
        "pretrained_vit_summary": read_json(Path("results/pretrained_vit_transfer_merge/summary.json")),
        "pairwise_summary": read_json(Path("results/digit_pairwise_experts/summary.json")),
        "qwen_summary": read_json(Path("results/qwen_path_sweep/summary.json")),
        "qwen_gsm8k_summary": read_json(Path("results/qwen_gsm8k_slice/summary.json")),
        "qwen_mmlu_summary": read_json(Path("results/qwen_mmlu_slice/summary.json")),
        "qwen_humaneval_summary": read_json(Path("results/qwen_humaneval_nll_slice/summary.json")),
        "qwen_safety_summary": read_json(Path("results/qwen_safety_refusal_slice/summary.json")),
        "qwen_multi_summary": read_json(Path("results/qwen_multi_expert_merge/summary.json")),
        "grid_rows": read_csv(Path("results/digits_merge/grid_metrics.csv")),
        "method_rows": read_csv(Path("results/digits_merge/method_metrics.csv")),
        "cifar_grid_rows": read_csv(Path("results/cifar_merge/grid_metrics.csv")),
        "cifar_method_rows": read_csv(Path("results/cifar_merge/method_metrics.csv")),
        "cifar_lambda_rows": read_csv(Path("results/cifar_merge/lambda_sweep.csv")),
        "vit_grid_rows": read_csv(Path("results/cifar100_vit_merge/grid_metrics.csv")),
        "vit_method_rows": read_csv(Path("results/cifar100_vit_merge/method_metrics.csv")),
        "vit_lambda_rows": read_csv(Path("results/cifar100_vit_merge/lambda_sweep.csv")),
        "pretrained_vit_grid_rows": read_csv(Path("results/pretrained_vit_transfer_merge/grid_metrics.csv")),
        "pretrained_vit_method_rows": read_csv(Path("results/pretrained_vit_transfer_merge/method_metrics.csv")),
        "pretrained_vit_lambda_rows": read_csv(Path("results/pretrained_vit_transfer_merge/lambda_sweep.csv")),
        "pairwise_rows": read_csv(Path("results/digit_pairwise_experts/pairwise_metrics.csv")),
        "alignment_rows": read_csv(Path("results/alignment_barrier/path_metrics.csv")),
        "qwen_rows": read_csv(Path("results/qwen_path_sweep/path_metrics.csv")),
        "qwen_gsm8k_rows": read_csv(Path("results/qwen_gsm8k_slice/metrics.csv")),
        "qwen_mmlu_rows": read_csv(Path("results/qwen_mmlu_slice/metrics.csv")),
        "qwen_humaneval_rows": read_csv(Path("results/qwen_humaneval_nll_slice/metrics.csv")),
        "qwen_safety_rows": read_csv(Path("results/qwen_safety_refusal_slice/metrics.csv")),
        "qwen_multi_grid_rows": read_csv(Path("results/qwen_multi_expert_merge/grid_metrics.csv")),
        "qwen_multi_method_rows": read_csv(Path("results/qwen_multi_expert_merge/method_metrics.csv")),
        "qwen_multi_conflict_rows": read_csv(Path("results/qwen_multi_expert_merge/pairwise_conflict.csv")),
    }


def html_template(data: dict[str, object], figures: dict[str, str]) -> str:
    payload = json.dumps(data, indent=2)
    figure_payload = json.dumps(figures, indent=2)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Visualizing Model Merging Dashboard</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #1f2933;
      --muted: #65717e;
      --line: #d7dde4;
      --blue: #2563a8;
      --teal: #2a9d8f;
      --coral: #e76f51;
      --ink: #111827;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }}
    header {{
      padding: 18px 24px 12px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      position: sticky;
      top: 0;
      z-index: 10;
    }}
    h1 {{
      font-size: 22px;
      margin: 0 0 10px;
      font-weight: 650;
    }}
    h2 {{
      font-size: 17px;
      margin: 0 0 12px;
      font-weight: 650;
    }}
    h3 {{
      font-size: 14px;
      margin: 0 0 8px;
      font-weight: 650;
    }}
    .tabs {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .tab {{
      border: 1px solid var(--line);
      background: #f9fafb;
      color: var(--text);
      border-radius: 7px;
      padding: 7px 11px;
      cursor: pointer;
      font: inherit;
    }}
    .tab.active {{
      background: var(--blue);
      color: white;
      border-color: var(--blue);
    }}
    main {{
      padding: 18px 24px 32px;
      max-width: 1440px;
      margin: 0 auto;
    }}
    section {{ display: none; }}
    section.active {{ display: block; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-height: 84px;
    }}
    .metric .label {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }}
    .metric .value {{
      font-size: 22px;
      font-weight: 700;
      color: var(--ink);
    }}
    .metric .note {{
      color: var(--muted);
      font-size: 12px;
      margin-top: 5px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      margin-bottom: 14px;
    }}
    .split {{
      display: grid;
      grid-template-columns: minmax(0, 1.15fr) minmax(360px, 0.85fr);
      gap: 14px;
      align-items: start;
    }}
    .figure-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }}
    figure {{
      margin: 0;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
    }}
    figcaption {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 8px;
    }}
    img {{
      display: block;
      width: 100%;
      height: auto;
      border-radius: 5px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    th, td {{
      padding: 7px 8px;
      border-bottom: 1px solid #edf0f3;
      text-align: right;
      white-space: nowrap;
    }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{
      color: var(--muted);
      font-weight: 650;
      background: #fbfcfd;
      position: sticky;
      top: 0;
    }}
    .table-wrap {{
      overflow: auto;
      max-height: 520px;
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .controls {{
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 10px;
      color: var(--muted);
    }}
    select {{
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 6px 8px;
      background: white;
      color: var(--text);
      font: inherit;
    }}
    .slider-row {{
      display: grid;
      grid-template-columns: 88px minmax(0, 1fr) 64px;
      gap: 10px;
      align-items: center;
      margin: 10px 0;
      color: var(--muted);
    }}
    input[type="range"] {{
      width: 100%;
      accent-color: var(--blue);
    }}
    .explorer-wrap {{
      min-height: 520px;
    }}
    .selected-detail {{
      color: var(--muted);
      margin-bottom: 12px;
    }}
    svg {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .legend {{
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 12px;
      margin-top: 8px;
    }}
    .swatch {{
      width: 12px;
      height: 12px;
      display: inline-block;
      border-radius: 3px;
      margin-right: 5px;
      vertical-align: middle;
    }}
    .text-block {{
      color: var(--muted);
      max-width: 980px;
      margin: 0 0 14px;
    }}
    @media (max-width: 980px) {{
      .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .split {{ grid-template-columns: 1fr; }}
      .figure-grid {{ grid-template-columns: 1fr; }}
      main {{ padding: 14px; }}
      header {{ padding: 14px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Visualizing Model Merging</h1>
    <nav class="tabs" aria-label="Dashboard sections">
      <button class="tab active" data-tab="overview">Overview</button>
      <button class="tab" data-tab="merge">Merge Plane</button>
      <button class="tab" data-tab="cifar">CIFAR</button>
      <button class="tab" data-tab="vit">ViT</button>
      <button class="tab" data-tab="pairwise">Pairwise Experts</button>
      <button class="tab" data-tab="alignment">Alignment</button>
      <button class="tab" data-tab="qwen">Qwen / LLM</button>
    </nav>
  </header>
  <main>
    <section id="overview" class="active">
      <div class="grid" id="overviewMetrics"></div>
      <div class="panel">
        <h2>Evidence Map</h2>
        <div class="table-wrap"><table id="evidenceTable"></table></div>
      </div>
      <div class="figure-grid">
        <figure><figcaption>Task-vector merge landscape</figcaption><img id="overviewMerge"></figure>
        <figure><figcaption>Qwen multi-expert merge grid</figcaption><img id="overviewQwen"></figure>
      </div>
    </section>

    <section id="merge">
      <div class="split">
        <div class="panel">
          <h2>Interactive Plane</h2>
          <div class="controls">
            <label for="explorerDataset">Task Pair</label>
            <select id="explorerDataset"></select>
            <label for="explorerMetric">Objective</label>
            <select id="explorerMetric"></select>
            <label for="planeScale">Plane</label>
            <select id="planeScale">
              <option value="raw">Raw</option>
              <option value="normalized">Normalized</option>
            </select>
          </div>
          <div class="explorer-wrap" id="planeExplorer"></div>
        </div>
        <div class="panel">
          <h2>Selected Point</h2>
          <div class="selected-detail" id="selectedPointLabel"></div>
          <div class="slider-row">
            <label for="alphaSlider">alpha</label>
            <input id="alphaSlider" type="range" min="0" max="1" step="0.001">
            <span id="alphaValue"></span>
          </div>
          <div class="slider-row">
            <label for="betaSlider">beta</label>
            <input id="betaSlider" type="range" min="0" max="1" step="0.001">
            <span id="betaValue"></span>
          </div>
          <div class="slider-row">
            <label for="lambdaSlider">lambda</label>
            <input id="lambdaSlider" type="range" min="0" max="1" step="0.001">
            <span id="lambdaValue"></span>
          </div>
          <div class="controls">
            <label for="explorerMethod">Method</label>
            <select id="explorerMethod"></select>
          </div>
          <div class="table-wrap"><table id="selectedPointTable"></table></div>
        </div>
      </div>
      <div class="split">
        <div class="panel">
          <h2>Merge Method Table</h2>
          <div class="table-wrap"><table id="methodTable"></table></div>
        </div>
        <div class="panel">
          <h2>Layer Interference</h2>
          <img id="mergeInterference">
        </div>
      </div>
      <div class="figure-grid">
        <figure><figcaption>Loss surfaces and method points</figcaption><img id="mergeLandscape"></figure>
        <figure><figcaption>Per-task basin overlay</figcaption><img id="mergeOverlay"></figure>
        <figure><figcaption>Lambda sweep</figcaption><img id="mergeLambda"></figure>
        <figure><figcaption>Method overlay</figcaption><img id="mergeMethods"></figure>
      </div>
    </section>

    <section id="pairwise">
      <div class="split">
        <div class="panel">
          <h2>Digit-Pair Matrix</h2>
          <div class="controls">
            <label for="pairMetric">Metric</label>
            <select id="pairMetric">
              <option value="linear_worst_acc">Linear worst accuracy</option>
              <option value="linear_drop_from_base">Drop from base</option>
              <option value="cosine">Task-vector cosine</option>
              <option value="weighted_conflict">Weighted sign conflict</option>
            </select>
          </div>
          <div id="pairHeatmap"></div>
        </div>
        <div class="panel">
          <h2>Worst Pairs</h2>
          <div class="table-wrap"><table id="worstPairTable"></table></div>
        </div>
      </div>
      <div class="figure-grid">
        <figure><figcaption>Pairwise heatmaps from the run</figcaption><img id="pairwiseHeatmaps"></figure>
        <figure><figcaption>Conflict metrics vs merge drop</figcaption><img id="pairwiseScatter"></figure>
        <figure><figcaption>Average layer conflict atlas</figcaption><img id="pairwiseLayer"></figure>
      </div>
    </section>

    <section id="cifar">
      <div class="grid" id="cifarMetrics"></div>
      <div class="split">
        <div class="panel">
          <h2>CIFAR Method Table</h2>
          <div class="table-wrap"><table id="cifarMethodTable"></table></div>
        </div>
        <div class="panel">
          <h2>CIFAR Lambda Path</h2>
          <div id="cifarChart"></div>
        </div>
      </div>
      <div class="figure-grid">
        <figure><figcaption>CIFAR vehicle/animal merge landscape</figcaption><img id="cifarLandscape"></figure>
        <figure><figcaption>CIFAR method overlay</figcaption><img id="cifarMethods"></figure>
        <figure><figcaption>CIFAR lambda sweep</figcaption><img id="cifarLambda"></figure>
        <figure><figcaption>CIFAR interference atlas</figcaption><img id="cifarInterference"></figure>
      </div>
    </section>

    <section id="vit">
      <div class="grid" id="vitMetrics"></div>
      <div class="split">
        <div class="panel">
          <h2>ViT-Style Method Table</h2>
          <div class="table-wrap"><table id="vitMethodTable"></table></div>
        </div>
        <div class="panel">
          <h2>ViT-Style Lambda Path</h2>
          <div id="vitChart"></div>
        </div>
      </div>
      <div class="split">
        <div class="panel">
          <h2>Pretrained ViT Transfer Methods</h2>
          <div class="table-wrap"><table id="pretrainedVitMethodTable"></table></div>
        </div>
        <div class="panel">
          <h2>Pretrained ViT Lambda Path</h2>
          <div id="pretrainedVitChart"></div>
        </div>
      </div>
      <div class="figure-grid">
        <figure><figcaption>CIFAR100 ViT-style merge landscape</figcaption><img id="vitLandscape"></figure>
        <figure><figcaption>CIFAR100 ViT-style method overlay</figcaption><img id="vitMethods"></figure>
        <figure><figcaption>CIFAR100 ViT-style lambda sweep</figcaption><img id="vitLambda"></figure>
        <figure><figcaption>CIFAR100 ViT-style interference atlas</figcaption><img id="vitInterference"></figure>
        <figure><figcaption>CIFAR100 ViT-style task-vector PCA</figcaption><img id="vitPca"></figure>
        <figure><figcaption>Pretrained ViT transfer landscape</figcaption><img id="pretrainedVitLandscape"></figure>
        <figure><figcaption>Pretrained ViT transfer methods</figcaption><img id="pretrainedVitMethods"></figure>
        <figure><figcaption>Pretrained ViT transfer lambda sweep</figcaption><img id="pretrainedVitLambda"></figure>
        <figure><figcaption>Pretrained ViT head conflict</figcaption><img id="pretrainedVitInterference"></figure>
      </div>
    </section>

    <section id="alignment">
      <div class="grid" id="alignmentMetrics"></div>
      <div class="panel">
        <h2>Interpolation Path</h2>
        <div id="alignmentChart"></div>
      </div>
      <figure><figcaption>Generated alignment figure</figcaption><img id="alignmentFigure"></figure>
    </section>

    <section id="qwen">
      <div class="grid" id="qwenMetrics"></div>
      <div class="split">
        <div class="panel">
          <h2>Lambda Path</h2>
          <div id="qwenChart"></div>
        </div>
        <div class="panel">
          <h2>Path Metrics</h2>
          <div class="table-wrap"><table id="qwenTable"></table></div>
        </div>
      </div>
      <div class="split">
        <div class="panel">
          <h2>GSM8K Slice</h2>
          <div class="table-wrap"><table id="qwenGsm8kTable"></table></div>
        </div>
        <div class="panel">
          <h2>MMLU Slice</h2>
          <div class="table-wrap"><table id="qwenMmluTable"></table></div>
        </div>
      </div>
      <div class="split">
        <div class="panel">
          <h2>HumanEval NLL Slice</h2>
          <div class="table-wrap"><table id="qwenHumanEvalTable"></table></div>
        </div>
        <div class="panel">
          <h2>Safety / Refusal Slice</h2>
          <div class="table-wrap"><table id="qwenSafetyTable"></table></div>
        </div>
      </div>
      <div class="split">
        <div class="panel">
          <h2>Expert Conflict</h2>
          <div class="table-wrap"><table id="qwenConflictTable"></table></div>
        </div>
      </div>
      <div class="split">
        <div class="panel">
          <h2>Multi-Expert Methods</h2>
          <div class="table-wrap"><table id="qwenMultiTable"></table></div>
        </div>
      </div>
      <div class="figure-grid">
        <figure><figcaption>Qwen path sweep</figcaption><img id="qwenFigure"></figure>
        <figure><figcaption>Largest base-to-instruct deltas</figcaption><img id="qwenDeltaFigure"></figure>
        <figure><figcaption>GSM8K exact-match benchmark slice</figcaption><img id="qwenGsm8kFigure"></figure>
        <figure><figcaption>MMLU multiple-choice benchmark slice</figcaption><img id="qwenMmluFigure"></figure>
        <figure><figcaption>HumanEval canonical-solution NLL slice</figcaption><img id="qwenHumanEvalFigure"></figure>
        <figure><figcaption>BeaverTails safety/refusal NLL slice</figcaption><img id="qwenSafetyFigure"></figure>
        <figure><figcaption>Qwen instruct+coder merge grid</figcaption><img id="qwenMultiGridFigure"></figure>
        <figure><figcaption>Qwen multi-expert diagonal path</figcaption><img id="qwenMultiPathFigure"></figure>
        <figure><figcaption>Qwen instruct/coder conflict</figcaption><img id="qwenMultiConflictFigure"></figure>
      </div>
    </section>
  </main>

  <script>
    const DATA = {payload};
    const FIGURES = {figure_payload};

    const fmt = (value, digits = 3) => {{
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "";
      const number = Number(value);
      if (Math.abs(number) >= 1000) return number.toFixed(0);
      return number.toFixed(digits);
    }};

    function setImage(id, name) {{
      document.getElementById(id).src = FIGURES[name];
    }}

    function metricCard(label, value, note = "") {{
      return `<div class="metric"><div class="label">${{label}}</div><div class="value">${{value}}</div><div class="note">${{note}}</div></div>`;
    }}

    function renderTable(id, rows, columns) {{
      const table = document.getElementById(id);
      const head = `<thead><tr>${{columns.map(c => `<th>${{c.label}}</th>`).join("")}}</tr></thead>`;
      const body = rows.map(row => `<tr>${{columns.map(c => `<td>${{c.format ? c.format(row[c.key], row) : row[c.key]}}</td>`).join("")}}</tr>`).join("");
      table.innerHTML = head + `<tbody>${{body}}</tbody>`;
    }}

    function colorRamp(value, min, max, reverse = false) {{
      if (max <= min) return "#d7dde4";
      let t = (value - min) / (max - min);
      t = Math.max(0, Math.min(1, reverse ? 1 - t : t));
      const r = Math.round(38 + t * 210);
      const g = Math.round(99 + t * 110);
      const b = Math.round(168 - t * 120);
      return `rgb(${{r}},${{g}},${{b}})`;
    }}

    function renderPairHeatmap(metric) {{
      const rows = DATA.pairwise_rows;
      const values = rows.map(r => Number(r[metric])).filter(v => !Number.isNaN(v));
      const min = Math.min(...values);
      const max = Math.max(...values);
      const cell = 42;
      const pad = 46;
      let rects = "";
      for (let i = 0; i < 10; i++) {{
        for (let j = 0; j < 10; j++) {{
          const found = rows.find(r => (Number(r.left_digit) === i && Number(r.right_digit) === j) || (Number(r.left_digit) === j && Number(r.right_digit) === i));
          const x = pad + j * cell;
          const y = pad + i * cell;
          if (!found || i === j) {{
            rects += `<rect x="${{x}}" y="${{y}}" width="${{cell - 2}}" height="${{cell - 2}}" fill="#eef1f4"/>`;
            continue;
          }}
          const value = Number(found[metric]);
          const fill = colorRamp(value, min, max, metric === "linear_worst_acc");
          rects += `<rect x="${{x}}" y="${{y}}" width="${{cell - 2}}" height="${{cell - 2}}" fill="${{fill}}"/>`;
          rects += `<text x="${{x + cell / 2}}" y="${{y + cell / 2 + 4}}" text-anchor="middle" font-size="10" fill="white">${{fmt(value, 2)}}</text>`;
        }}
      }}
      let labels = "";
      for (let i = 0; i < 10; i++) {{
        labels += `<text x="${{pad + i * cell + cell / 2}}" y="30" text-anchor="middle" font-size="12">${{i}}</text>`;
        labels += `<text x="28" y="${{pad + i * cell + cell / 2 + 4}}" text-anchor="middle" font-size="12">${{i}}</text>`;
      }}
      document.getElementById("pairHeatmap").innerHTML = `<svg viewBox="0 0 490 490" role="img">${{labels}}${{rects}}<text x="245" y="480" text-anchor="middle" font-size="12" fill="#65717e">min ${{fmt(min, 3)}} / max ${{fmt(max, 3)}}</text></svg>`;
    }}

    function renderLineChart(id, rows, xKey, series, yLabel) {{
      const width = 720;
      const height = 330;
      const pad = {{ left: 58, right: 18, top: 24, bottom: 42 }};
      const xs = rows.map(r => Number(r[xKey]));
      const ys = series.flatMap(s => rows.map(r => Number(r[s.key])));
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const minY = Math.min(...ys);
      const maxY = Math.max(...ys);
      const sx = x => pad.left + ((x - minX) / Math.max(1e-9, maxX - minX)) * (width - pad.left - pad.right);
      const sy = y => height - pad.bottom - ((y - minY) / Math.max(1e-9, maxY - minY)) * (height - pad.top - pad.bottom);
      const lines = series.map(s => {{
        const points = rows.map(r => `${{sx(Number(r[xKey]))}},${{sy(Number(r[s.key]))}}`).join(" ");
        const dots = rows.map(r => `<circle cx="${{sx(Number(r[xKey]))}}" cy="${{sy(Number(r[s.key]))}}" r="3.5" fill="${{s.color}}"/>`).join("");
        return `<polyline points="${{points}}" fill="none" stroke="${{s.color}}" stroke-width="2.2"/>${{dots}}`;
      }}).join("");
      const legend = series.map((s, i) => `<g transform="translate(${{pad.left + i * 145}},${{height - 12}})"><rect width="10" height="10" fill="${{s.color}}"/><text x="16" y="10" font-size="12">${{s.label}}</text></g>`).join("");
      const xTicks = xs.map(x => `<text x="${{sx(x)}}" y="${{height - pad.bottom + 18}}" text-anchor="middle" font-size="11">${{fmt(x, 2)}}</text>`).join("");
      const yTicks = [minY, (minY + maxY) / 2, maxY].map(y => `<text x="${{pad.left - 8}}" y="${{sy(y) + 4}}" text-anchor="end" font-size="11">${{fmt(y, 2)}}</text><line x1="${{pad.left}}" x2="${{width - pad.right}}" y1="${{sy(y)}}" y2="${{sy(y)}}" stroke="#edf0f3"/>`).join("");
      document.getElementById(id).innerHTML = `<svg viewBox="0 0 ${{width}} ${{height}}" role="img">${{yTicks}}<line x1="${{pad.left}}" y1="${{height - pad.bottom}}" x2="${{width - pad.right}}" y2="${{height - pad.bottom}}" stroke="#9aa5b1"/><line x1="${{pad.left}}" y1="${{pad.top}}" x2="${{pad.left}}" y2="${{height - pad.bottom}}" stroke="#9aa5b1"/>${{xTicks}}<text x="16" y="${{height / 2}}" transform="rotate(-90 16 ${{height / 2}})" font-size="12" fill="#65717e">${{yLabel}}</text>${{lines}}${{legend}}</svg>`;
    }}

    const explorerState = {{
      dataset: "digits",
      metric: null,
      alpha: 0.5,
      beta: 0.5,
      method: "linear_average",
      scale: "raw",
    }};
    let explorerDragging = false;

    const EXPLORERS = {{
      digits: {{
        label: "Digits 0-4 / 5-9",
        rows: DATA.grid_rows,
        methods: DATA.method_rows,
        metrics: [
          {{ key: "worst_acc", label: "Worst accuracy" }},
          {{ key: "avg_acc", label: "Average accuracy" }},
          {{ key: "task_a_acc", label: "Task A accuracy" }},
          {{ key: "task_b_acc", label: "Task B accuracy" }},
          {{ key: "worst_loss", label: "Worst loss" }},
          {{ key: "avg_loss", label: "Average loss" }},
          {{ key: "task_a_loss", label: "Task A loss" }},
          {{ key: "task_b_loss", label: "Task B loss" }},
        ],
        detailColumns: [
          {{ key: "task_a_acc", label: "task A accuracy" }},
          {{ key: "task_b_acc", label: "task B accuracy" }},
          {{ key: "avg_acc", label: "average accuracy" }},
          {{ key: "worst_acc", label: "worst accuracy" }},
          {{ key: "avg_loss", label: "average loss" }},
          {{ key: "worst_loss", label: "worst loss" }},
        ],
      }},
      cifar: {{
        label: "CIFAR vehicles / animals",
        rows: DATA.cifar_grid_rows,
        methods: DATA.cifar_method_rows,
        metrics: [
          {{ key: "worst_acc", label: "Worst accuracy" }},
          {{ key: "avg_acc", label: "Average accuracy" }},
          {{ key: "vehicle_acc", label: "Vehicle accuracy" }},
          {{ key: "animal_acc", label: "Animal accuracy" }},
          {{ key: "worst_loss", label: "Worst loss" }},
          {{ key: "avg_loss", label: "Average loss" }},
          {{ key: "vehicle_loss", label: "Vehicle loss" }},
          {{ key: "animal_loss", label: "Animal loss" }},
        ],
        detailColumns: [
          {{ key: "vehicle_acc", label: "vehicle accuracy" }},
          {{ key: "animal_acc", label: "animal accuracy" }},
          {{ key: "avg_acc", label: "average accuracy" }},
          {{ key: "worst_acc", label: "worst accuracy" }},
          {{ key: "avg_loss", label: "average loss" }},
          {{ key: "worst_loss", label: "worst loss" }},
        ],
      }},
      vit: {{
        label: "CIFAR100 living / object",
        rows: DATA.vit_grid_rows,
        methods: DATA.vit_method_rows,
        metrics: [
          {{ key: "worst_acc", label: "Worst accuracy" }},
          {{ key: "avg_acc", label: "Average accuracy" }},
          {{ key: "living_acc", label: "Living accuracy" }},
          {{ key: "object_acc", label: "Object accuracy" }},
          {{ key: "worst_loss", label: "Worst loss" }},
          {{ key: "avg_loss", label: "Average loss" }},
          {{ key: "living_loss", label: "Living loss" }},
          {{ key: "object_loss", label: "Object loss" }},
        ],
        detailColumns: [
          {{ key: "living_acc", label: "living accuracy" }},
          {{ key: "object_acc", label: "object accuracy" }},
          {{ key: "avg_acc", label: "average accuracy" }},
          {{ key: "worst_acc", label: "worst accuracy" }},
          {{ key: "avg_loss", label: "average loss" }},
          {{ key: "worst_loss", label: "worst loss" }},
        ],
      }},
      qwen_multi: {{
        label: "Qwen instruct / coder",
        rows: DATA.qwen_multi_grid_rows,
        methods: DATA.qwen_multi_method_rows,
        metrics: [
          {{ key: "avg_nll", label: "Average NLL" }},
          {{ key: "worst_nll", label: "Worst NLL" }},
          {{ key: "general_nll", label: "General NLL" }},
          {{ key: "instruction_nll", label: "Instruction NLL" }},
          {{ key: "code_nll", label: "Code NLL" }},
        ],
        detailColumns: [
          {{ key: "general_nll", label: "general NLL" }},
          {{ key: "instruction_nll", label: "instruction NLL" }},
          {{ key: "code_nll", label: "code NLL" }},
          {{ key: "avg_nll", label: "average NLL" }},
          {{ key: "worst_nll", label: "worst NLL" }},
        ],
      }},
      pretrained_vit: {{
        label: "Pretrained ViT living / object",
        rows: DATA.pretrained_vit_grid_rows,
        methods: DATA.pretrained_vit_method_rows,
        metrics: [
          {{ key: "worst_acc", label: "Worst accuracy" }},
          {{ key: "avg_acc", label: "Average accuracy" }},
          {{ key: "living_acc", label: "Living accuracy" }},
          {{ key: "object_acc", label: "Object accuracy" }},
          {{ key: "worst_loss", label: "Worst loss" }},
          {{ key: "avg_loss", label: "Average loss" }},
          {{ key: "living_loss", label: "Living loss" }},
          {{ key: "object_loss", label: "Object loss" }},
        ],
        detailColumns: [
          {{ key: "living_acc", label: "living accuracy" }},
          {{ key: "object_acc", label: "object accuracy" }},
          {{ key: "avg_acc", label: "average accuracy" }},
          {{ key: "worst_acc", label: "worst accuracy" }},
          {{ key: "avg_loss", label: "average loss" }},
          {{ key: "worst_loss", label: "worst loss" }},
        ],
      }},
    }};

    function metricHigherIsBetter(metric) {{
      return metric.endsWith("_acc") || metric === "avg_acc" || metric === "worst_acc";
    }}

    function finiteNumber(value) {{
      const number = Number(value);
      return Number.isFinite(number) ? number : null;
    }}

    function explorerBounds(rows) {{
      const alphas = rows.map(r => Number(r.alpha)).filter(Number.isFinite);
      const betas = rows.map(r => Number(r.beta)).filter(Number.isFinite);
      return {{
        minAlpha: Math.min(...alphas),
        maxAlpha: Math.max(...alphas),
        minBeta: Math.min(...betas),
        maxBeta: Math.max(...betas),
      }};
    }}

    function nearestGridRow(rows, alpha, beta) {{
      let best = rows[0];
      let bestDistance = Infinity;
      rows.forEach(row => {{
        const da = Number(row.alpha) - alpha;
        const db = Number(row.beta) - beta;
        const distance = da * da + db * db;
        if (distance < bestDistance) {{
          best = row;
          bestDistance = distance;
        }}
      }});
      return best;
    }}

    function setExplorerPointFromEvent(event) {{
      const svg = document.getElementById("planeExplorerSvg");
      if (!svg) return;
      const config = EXPLORERS[explorerState.dataset];
      const bounds = explorerBounds(config.rows);
      const width = 640;
      const height = 540;
      const pad = {{ left: 70, right: 28, top: 26, bottom: 62 }};
      const plotW = width - pad.left - pad.right;
      const plotH = height - pad.top - pad.bottom;
      const rect = svg.getBoundingClientRect();
      const x = (event.clientX - rect.left) / Math.max(1, rect.width) * width;
      const y = (event.clientY - rect.top) / Math.max(1, rect.height) * height;
      const alpha = bounds.minAlpha + ((x - pad.left) / plotW) * (bounds.maxAlpha - bounds.minAlpha);
      const beta = bounds.maxBeta - ((y - pad.top) / plotH) * (bounds.maxBeta - bounds.minBeta);
      setExplorerPoint(alpha, beta, "__custom");
    }}

    function methodRows(config) {{
      return config.methods
        .filter(row => finiteNumber(row.alpha) !== null && finiteNumber(row.beta) !== null)
        .map(row => ({{
          ...row,
          alpha: Number(row.alpha),
          beta: Number(row.beta),
        }}));
    }}

    function setExplorerPoint(alpha, beta, method = "__custom") {{
      const config = EXPLORERS[explorerState.dataset];
      const bounds = explorerBounds(config.rows);
      explorerState.alpha = Math.max(bounds.minAlpha, Math.min(bounds.maxAlpha, alpha));
      explorerState.beta = Math.max(bounds.minBeta, Math.min(bounds.maxBeta, beta));
      explorerState.method = method;
      renderPlaneExplorer();
    }}

    function updateExplorerControls() {{
      const config = EXPLORERS[explorerState.dataset];
      const bounds = explorerBounds(config.rows);
      const metricSelect = document.getElementById("explorerMetric");
      const methodSelect = document.getElementById("explorerMethod");
      const alphaSlider = document.getElementById("alphaSlider");
      const betaSlider = document.getElementById("betaSlider");
      const lambdaSlider = document.getElementById("lambdaSlider");
      metricSelect.innerHTML = config.metrics.map(metric => `<option value="${{metric.key}}">${{metric.label}}</option>`).join("");
      if (!config.metrics.some(metric => metric.key === explorerState.metric)) {{
        explorerState.metric = config.metrics[0].key;
      }}
      metricSelect.value = explorerState.metric;
      const methods = methodRows(config);
      methodSelect.innerHTML = `<option value="__custom">Selected point</option>` + methods.map(row => `<option value="${{row.method}}">${{row.method}}</option>`).join("");
      if (!methods.some(row => row.method === explorerState.method)) {{
        const fallback = methods.find(row => row.method === "linear_average") || methods[0];
        explorerState.method = fallback ? fallback.method : "__custom";
        if (fallback) {{
          explorerState.alpha = fallback.alpha;
          explorerState.beta = fallback.beta;
        }}
      }}
      methodSelect.value = explorerState.method;
      [alphaSlider, betaSlider].forEach(slider => {{
        slider.min = String(bounds.minAlpha);
        slider.max = String(bounds.maxAlpha);
        slider.step = String(Math.max(0.001, (bounds.maxAlpha - bounds.minAlpha) / 400));
      }});
      const lambdaMin = Math.max(bounds.minAlpha, bounds.minBeta);
      const lambdaMax = Math.min(bounds.maxAlpha, bounds.maxBeta);
      lambdaSlider.min = String(lambdaMin);
      lambdaSlider.max = String(lambdaMax);
      lambdaSlider.step = String(Math.max(0.001, (lambdaMax - lambdaMin) / 400));
    }}

    function renderPlaneExplorer() {{
      const config = EXPLORERS[explorerState.dataset];
      const bounds = explorerBounds(config.rows);
      const metric = explorerState.metric || config.metrics[0].key;
      const rows = config.rows;
      const values = rows.map(row => Number(row[metric])).filter(Number.isFinite);
      const minValue = Math.min(...values);
      const maxValue = Math.max(...values);
      const width = 640;
      const height = 540;
      const pad = {{ left: 70, right: 28, top: 26, bottom: 62 }};
      const plotW = width - pad.left - pad.right;
      const plotH = height - pad.top - pad.bottom;
      const alphaValues = [...new Set(rows.map(row => Number(row.alpha)))].sort((a, b) => a - b);
      const betaValues = [...new Set(rows.map(row => Number(row.beta)))].sort((a, b) => a - b);
      const cellW = plotW / Math.max(1, alphaValues.length);
      const cellH = plotH / Math.max(1, betaValues.length);
      const sx = alpha => pad.left + ((alpha - bounds.minAlpha) / Math.max(1e-9, bounds.maxAlpha - bounds.minAlpha)) * plotW;
      const sy = beta => pad.top + ((bounds.maxBeta - beta) / Math.max(1e-9, bounds.maxBeta - bounds.minBeta)) * plotH;
      const selected = nearestGridRow(rows, explorerState.alpha, explorerState.beta);
      explorerState.alpha = Number(selected.alpha);
      explorerState.beta = Number(selected.beta);
      const higherBetter = metricHigherIsBetter(metric);
      const reverse = !higherBetter;
      const cells = rows.map(row => {{
        const value = Number(row[metric]);
        const x = sx(Number(row.alpha)) - cellW / 2;
        const y = sy(Number(row.beta)) - cellH / 2;
        const fill = colorRamp(value, minValue, maxValue, reverse);
        return `<rect x="${{x}}" y="${{y}}" width="${{cellW + 0.5}}" height="${{cellH + 0.5}}" fill="${{fill}}"><title>alpha ${{fmt(row.alpha, 3)}}, beta ${{fmt(row.beta, 3)}}, ${{metric}} ${{fmt(value, 3)}}</title></rect>`;
      }}).join("");
      const methods = methodRows(config);
      const methodPoints = methods.map(row => {{
        const selectedMethod = row.method === explorerState.method;
        const radius = selectedMethod ? 5.5 : 3.5;
        const stroke = selectedMethod ? "#111827" : "white";
        return `<circle cx="${{sx(row.alpha)}}" cy="${{sy(row.beta)}}" r="${{radius}}" fill="#111827" stroke="${{stroke}}" stroke-width="1.6"><title>${{row.method}} (${{fmt(row.alpha, 2)}}, ${{fmt(row.beta, 2)}})</title></circle>`;
      }}).join("");
      const point = `<circle cx="${{sx(explorerState.alpha)}}" cy="${{sy(explorerState.beta)}}" r="8" fill="#e76f51" stroke="white" stroke-width="2.5"/><circle cx="${{sx(explorerState.alpha)}}" cy="${{sy(explorerState.beta)}}" r="13" fill="transparent" stroke="#e76f51" stroke-width="1.5"/>`;
      const formatAxis = value => explorerState.scale === "normalized" ? fmt((value - bounds.minAlpha) / Math.max(1e-9, bounds.maxAlpha - bounds.minAlpha), 2) : fmt(value, 2);
      const formatBetaAxis = value => explorerState.scale === "normalized" ? fmt((value - bounds.minBeta) / Math.max(1e-9, bounds.maxBeta - bounds.minBeta), 2) : fmt(value, 2);
      const xTicks = [bounds.minAlpha, (bounds.minAlpha + bounds.maxAlpha) / 2, bounds.maxAlpha].map(x => `<text x="${{sx(x)}}" y="${{height - pad.bottom + 22}}" text-anchor="middle" font-size="11">${{formatAxis(x)}}</text>`).join("");
      const yTicks = [bounds.minBeta, (bounds.minBeta + bounds.maxBeta) / 2, bounds.maxBeta].map(y => `<text x="${{pad.left - 9}}" y="${{sy(y) + 4}}" text-anchor="end" font-size="11">${{formatBetaAxis(y)}}</text><line x1="${{pad.left}}" x2="${{width - pad.right}}" y1="${{sy(y)}}" y2="${{sy(y)}}" stroke="rgba(255,255,255,0.55)"/>`).join("");
      const summary = `${{config.label}} | ${{metric}} range ${{fmt(minValue)}} to ${{fmt(maxValue)}}`;
      document.getElementById("planeExplorer").innerHTML = `<svg id="planeExplorerSvg" viewBox="0 0 ${{width}} ${{height}}" role="img"><rect x="${{pad.left}}" y="${{pad.top}}" width="${{plotW}}" height="${{plotH}}" fill="#eef1f4"/>${{cells}}${{yTicks}}${{xTicks}}<line x1="${{pad.left}}" y1="${{height - pad.bottom}}" x2="${{width - pad.right}}" y2="${{height - pad.bottom}}" stroke="#9aa5b1"/><line x1="${{pad.left}}" y1="${{pad.top}}" x2="${{pad.left}}" y2="${{height - pad.bottom}}" stroke="#9aa5b1"/><text x="${{pad.left + plotW / 2}}" y="${{height - 18}}" text-anchor="middle" font-size="12" fill="#65717e">alpha</text><text x="17" y="${{pad.top + plotH / 2}}" transform="rotate(-90 17 ${{pad.top + plotH / 2}})" font-size="12" fill="#65717e">beta</text>${{methodPoints}}${{point}}<text x="${{pad.left}}" y="16" font-size="12" fill="#65717e">${{summary}}</text></svg>`;
      const label = explorerState.method === "__custom" ? "Selected point" : explorerState.method;
      document.getElementById("selectedPointLabel").textContent = `${{config.label}} | ${{label}} | alpha=${{fmt(explorerState.alpha, 3)}}, beta=${{fmt(explorerState.beta, 3)}}`;
      document.getElementById("alphaSlider").value = String(explorerState.alpha);
      document.getElementById("betaSlider").value = String(explorerState.beta);
      document.getElementById("lambdaSlider").value = String(Math.max(Number(document.getElementById("lambdaSlider").min), Math.min(Number(document.getElementById("lambdaSlider").max), (explorerState.alpha + explorerState.beta) / 2)));
      document.getElementById("alphaValue").textContent = fmt(explorerState.alpha, 3);
      document.getElementById("betaValue").textContent = fmt(explorerState.beta, 3);
      document.getElementById("lambdaValue").textContent = fmt(Number(document.getElementById("lambdaSlider").value), 3);
      const detailRows = [
        {{ metric: "alpha", value: fmt(selected.alpha, 3) }},
        {{ metric: "beta", value: fmt(selected.beta, 3) }},
        ...config.detailColumns.map(column => ({{ metric: column.label, value: fmt(selected[column.key]) }})),
      ];
      renderTable("selectedPointTable", detailRows, [
        {{ key: "metric", label: "Metric" }},
        {{ key: "value", label: "Value" }},
      ]);
      const svg = document.getElementById("planeExplorerSvg");
      svg.addEventListener("pointerdown", event => {{
        explorerDragging = true;
        setExplorerPointFromEvent(event);
      }});
    }}

    function initExplorer() {{
      const datasetSelect = document.getElementById("explorerDataset");
      datasetSelect.innerHTML = Object.entries(EXPLORERS).map(([key, config]) => `<option value="${{key}}">${{config.label}}</option>`).join("");
      datasetSelect.value = explorerState.dataset;
      updateExplorerControls();
      renderPlaneExplorer();
      datasetSelect.addEventListener("change", event => {{
        explorerState.dataset = event.target.value;
        explorerState.metric = null;
        explorerState.method = "linear_average";
        updateExplorerControls();
        renderPlaneExplorer();
      }});
      document.getElementById("explorerMetric").addEventListener("change", event => {{
        explorerState.metric = event.target.value;
        renderPlaneExplorer();
      }});
      document.getElementById("planeScale").addEventListener("change", event => {{
        explorerState.scale = event.target.value;
        renderPlaneExplorer();
      }});
      document.getElementById("explorerMethod").addEventListener("change", event => {{
        const config = EXPLORERS[explorerState.dataset];
        const method = methodRows(config).find(row => row.method === event.target.value);
        if (method) {{
          setExplorerPoint(method.alpha, method.beta, method.method);
        }} else {{
          explorerState.method = "__custom";
          renderPlaneExplorer();
        }}
      }});
      document.getElementById("alphaSlider").addEventListener("input", event => {{
        setExplorerPoint(Number(event.target.value), explorerState.beta, "__custom");
      }});
      document.getElementById("betaSlider").addEventListener("input", event => {{
        setExplorerPoint(explorerState.alpha, Number(event.target.value), "__custom");
      }});
      document.getElementById("lambdaSlider").addEventListener("input", event => {{
        const value = Number(event.target.value);
        setExplorerPoint(value, value, "__custom");
      }});
      window.addEventListener("pointermove", event => {{
        if (explorerDragging) setExplorerPointFromEvent(event);
      }});
      window.addEventListener("pointerup", () => {{
        explorerDragging = false;
      }});
    }}

    function init() {{
      document.querySelectorAll(".tab").forEach(button => {{
        button.addEventListener("click", () => {{
          document.querySelectorAll(".tab").forEach(item => item.classList.remove("active"));
          document.querySelectorAll("section").forEach(item => item.classList.remove("active"));
          button.classList.add("active");
          document.getElementById(button.dataset.tab).classList.add("active");
        }});
      }});

      setImage("overviewMerge", "merge_landscape");
      setImage("overviewQwen", "qwen_multi_grid");
      setImage("mergeInterference", "interference_heatmap");
      setImage("mergeLandscape", "merge_landscape");
      setImage("mergeOverlay", "per_task_basin_overlay");
      setImage("mergeLambda", "lambda_sweep");
      setImage("mergeMethods", "method_overlay");
      setImage("pairwiseHeatmaps", "pairwise_heatmaps");
      setImage("pairwiseScatter", "conflict_vs_drop");
      setImage("pairwiseLayer", "layer_conflict_atlas");
      setImage("cifarLandscape", "cifar_landscape");
      setImage("cifarMethods", "cifar_methods");
      setImage("cifarLambda", "cifar_lambda");
      setImage("cifarInterference", "cifar_interference");
      setImage("vitLandscape", "vit_landscape");
      setImage("vitMethods", "vit_methods");
      setImage("vitLambda", "vit_lambda");
      setImage("vitInterference", "vit_interference");
      setImage("vitPca", "vit_pca");
      setImage("pretrainedVitLandscape", "pre_vit_landscape");
      setImage("pretrainedVitMethods", "pre_vit_methods");
      setImage("pretrainedVitLambda", "pre_vit_lambda");
      setImage("pretrainedVitInterference", "pre_vit_interference");
      setImage("alignmentFigure", "alignment");
      setImage("qwenFigure", "qwen_path");
      setImage("qwenDeltaFigure", "qwen_deltas");
      setImage("qwenGsm8kFigure", "qwen_gsm8k");
      setImage("qwenMmluFigure", "qwen_mmlu");
      setImage("qwenHumanEvalFigure", "qwen_humaneval");
      setImage("qwenSafetyFigure", "qwen_safety");
      setImage("qwenMultiGridFigure", "qwen_multi_grid");
      setImage("qwenMultiPathFigure", "qwen_multi_path");
      setImage("qwenMultiConflictFigure", "qwen_multi_conflict");
      initExplorer();

      const bestMethod = [...DATA.method_rows].sort((a, b) => Number(b.worst_acc) - Number(a.worst_acc))[0];
      const worstPair = [...DATA.pairwise_rows].sort((a, b) => Number(a.linear_worst_acc) - Number(b.linear_worst_acc))[0];
      const bestQwen = [...DATA.qwen_rows].sort((a, b) => Number(a.avg_nll) - Number(b.avg_nll))[0];
      const bestQwenMulti = [...DATA.qwen_multi_method_rows].sort((a, b) => Number(a.avg_nll) - Number(b.avg_nll))[0];
      document.getElementById("overviewMetrics").innerHTML = [
        metricCard("Digits grid", "41 x 41", "1681 evaluated checkpoints"),
        metricCard("Best merge worst acc", fmt(bestMethod.worst_acc), bestMethod.method),
        metricCard("Worst digit pair", `${{worstPair.left_digit}}/${{worstPair.right_digit}}`, `worst acc ${{fmt(worstPair.linear_worst_acc)}}`),
        metricCard("Best Qwen lambda", fmt(bestQwen.lambda, 2), `avg NLL ${{fmt(bestQwen.avg_nll)}}`),
        metricCard("Qwen multi best", bestQwenMulti.method, `avg NLL ${{fmt(bestQwenMulti.avg_nll)}}`),
      ].join("");

      renderTable("evidenceTable", [
        {{ artifact: "Merge landscape", result: "narrow shared basin; RegMean and layer-wise task arithmetic are included in the method overlay", path: "results/digits_merge" }},
        {{ artifact: "Single-digit experts", result: "weak global conflict/drop correlation; 3/9 is the clearest failure", path: "results/digit_pairwise_experts" }},
        {{ artifact: "Alignment barrier", result: "midpoint acc improves from 0.944 to 0.971 after hidden-unit matching", path: "results/alignment_barrier" }},
        {{ artifact: "ViT-style CIFAR100", result: "patch-transformer task-vector landscape with PCA geometry", path: "results/cifar100_vit_merge" }},
        {{ artifact: "Pretrained ViT transfer", result: "ImageNet-pretrained ViT-B/16 frozen-backbone head merge on CIFAR100 coarse groups", path: "results/pretrained_vit_transfer_merge" }},
        {{ artifact: "Qwen path", result: "instruction NLL best at lambda 0.75 on fixed prompt slice", path: "results/qwen_path_sweep" }},
        {{ artifact: "Qwen GSM8K slice", result: "cached GSM8K exact-match slice for base/intermediate/instruct", path: "results/qwen_gsm8k_slice" }},
        {{ artifact: "Qwen MMLU slice", result: "MMLU multiple-choice log-likelihood slice for base/intermediate/instruct", path: "results/qwen_mmlu_slice" }},
        {{ artifact: "Qwen HumanEval NLL", result: "HumanEval canonical-solution NLL slice for base/intermediate/instruct", path: "results/qwen_humaneval_nll_slice" }},
        {{ artifact: "Qwen safety/refusal", result: "BeaverTails safe-response and unsafe-refusal NLL slice", path: "results/qwen_safety_refusal_slice" }},
        {{ artifact: "Qwen multi-expert", result: "Qwen2.5 instruct and coder deltas evaluated in a two-expert merge plane", path: "results/qwen_multi_expert_merge" }},
      ], [
        {{ key: "artifact", label: "Artifact" }},
        {{ key: "result", label: "Result" }},
        {{ key: "path", label: "Path" }},
      ]);

      const methods = [...DATA.method_rows].sort((a, b) => Number(b.worst_acc) - Number(a.worst_acc));
      renderTable("methodTable", methods, [
        {{ key: "method", label: "Method" }},
        {{ key: "alpha", label: "alpha", format: v => fmt(v) }},
        {{ key: "beta", label: "beta", format: v => fmt(v) }},
        {{ key: "task_a_acc", label: "task A", format: v => fmt(v) }},
        {{ key: "task_b_acc", label: "task B", format: v => fmt(v) }},
        {{ key: "worst_acc", label: "worst", format: v => fmt(v) }},
        {{ key: "plane_residual", label: "residual", format: v => fmt(v) }},
      ]);

      document.getElementById("pairMetric").addEventListener("change", event => renderPairHeatmap(event.target.value));
      renderPairHeatmap("linear_worst_acc");
      const worstPairs = [...DATA.pairwise_rows].sort((a, b) => Number(a.linear_worst_acc) - Number(b.linear_worst_acc)).slice(0, 12);
      renderTable("worstPairTable", worstPairs, [
        {{ key: "left_digit", label: "left" }},
        {{ key: "right_digit", label: "right" }},
        {{ key: "linear_worst_acc", label: "worst acc", format: v => fmt(v) }},
        {{ key: "linear_drop_from_base", label: "drop", format: v => fmt(v) }},
        {{ key: "weighted_conflict", label: "weighted conflict", format: v => fmt(v) }},
      ]);

      const cifarMethods = [...DATA.cifar_method_rows].sort((a, b) => Number(b.worst_acc) - Number(a.worst_acc));
      const bestCifar = cifarMethods[0];
      const cifarBase = DATA.cifar_method_rows.find(r => r.method === "base");
      const cifarLinear = DATA.cifar_method_rows.find(r => r.method === "linear_average");
      document.getElementById("cifarMetrics").innerHTML = [
        metricCard("Best CIFAR method", bestCifar.method, `worst acc ${{fmt(bestCifar.worst_acc)}}`),
        metricCard("Base worst acc", fmt(cifarBase.worst_acc)),
        metricCard("Linear worst acc", fmt(cifarLinear.worst_acc), `drop ${{fmt(Number(cifarBase.worst_acc) - Number(cifarLinear.worst_acc))}}`),
        metricCard("Task-vector cosine", fmt(DATA.cifar_summary.global_task_vector_cosine)),
      ].join("");
      renderTable("cifarMethodTable", cifarMethods, [
        {{ key: "method", label: "Method" }},
        {{ key: "alpha", label: "alpha", format: v => fmt(v) }},
        {{ key: "beta", label: "beta", format: v => fmt(v) }},
        {{ key: "vehicle_acc", label: "vehicle", format: v => fmt(v) }},
        {{ key: "animal_acc", label: "animal", format: v => fmt(v) }},
        {{ key: "worst_acc", label: "worst", format: v => fmt(v) }},
        {{ key: "plane_residual", label: "residual", format: v => fmt(v) }},
      ]);
      renderLineChart("cifarChart", DATA.cifar_lambda_rows, "lambda", [
        {{ key: "vehicle_acc", label: "vehicle acc", color: "#2a9d8f" }},
        {{ key: "animal_acc", label: "animal acc", color: "#e76f51" }},
        {{ key: "worst_acc", label: "worst acc", color: "#6d597a" }},
      ], "accuracy");

      const vitMethods = [...DATA.vit_method_rows].sort((a, b) => Number(b.worst_acc) - Number(a.worst_acc));
      const bestVit = vitMethods[0];
      const vitBase = DATA.vit_method_rows.find(r => r.method === "base");
      const vitLinear = DATA.vit_method_rows.find(r => r.method === "linear_average");
      const preVitMethods = [...DATA.pretrained_vit_method_rows].sort((a, b) => Number(b.worst_acc) - Number(a.worst_acc));
      const bestPreVit = preVitMethods[0];
      const preVitBase = DATA.pretrained_vit_method_rows.find(r => r.method === "base");
      const preVitLinear = DATA.pretrained_vit_method_rows.find(r => r.method === "linear_average");
      document.getElementById("vitMetrics").innerHTML = [
        metricCard("Best ViT method", bestVit.method, `worst acc ${{fmt(bestVit.worst_acc)}}`),
        metricCard("Base worst acc", fmt(vitBase.worst_acc)),
        metricCard("Linear worst acc", fmt(vitLinear.worst_acc), `delta ${{fmt(Number(vitLinear.worst_acc) - Number(vitBase.worst_acc))}}`),
        metricCard("Task-vector cosine", fmt(DATA.vit_summary.global_task_vector_cosine)),
        metricCard("Pretrained ViT best", bestPreVit.method, `worst acc ${{fmt(bestPreVit.worst_acc)}}`),
        metricCard("Pretrained linear", fmt(preVitLinear.worst_acc), `base ${{fmt(preVitBase.worst_acc)}}`),
      ].join("");
      renderTable("vitMethodTable", vitMethods, [
        {{ key: "method", label: "Method" }},
        {{ key: "alpha", label: "alpha", format: v => fmt(v) }},
        {{ key: "beta", label: "beta", format: v => fmt(v) }},
        {{ key: "living_acc", label: "living", format: v => fmt(v) }},
        {{ key: "object_acc", label: "object", format: v => fmt(v) }},
        {{ key: "worst_acc", label: "worst", format: v => fmt(v) }},
        {{ key: "plane_residual", label: "residual", format: v => fmt(v) }},
      ]);
      renderLineChart("vitChart", DATA.vit_lambda_rows, "lambda", [
        {{ key: "living_acc", label: "living acc", color: "#2a9d8f" }},
        {{ key: "object_acc", label: "object acc", color: "#e76f51" }},
        {{ key: "worst_acc", label: "worst acc", color: "#6d597a" }},
      ], "accuracy");
      renderTable("pretrainedVitMethodTable", preVitMethods, [
        {{ key: "method", label: "Method" }},
        {{ key: "alpha", label: "alpha", format: v => fmt(v) }},
        {{ key: "beta", label: "beta", format: v => fmt(v) }},
        {{ key: "living_acc", label: "living", format: v => fmt(v) }},
        {{ key: "object_acc", label: "object", format: v => fmt(v) }},
        {{ key: "worst_acc", label: "worst", format: v => fmt(v) }},
        {{ key: "plane_residual", label: "residual", format: v => fmt(v) }},
      ]);
      renderLineChart("pretrainedVitChart", DATA.pretrained_vit_lambda_rows, "lambda", [
        {{ key: "living_acc", label: "living acc", color: "#2a9d8f" }},
        {{ key: "object_acc", label: "object acc", color: "#e76f51" }},
        {{ key: "worst_acc", label: "worst acc", color: "#6d597a" }},
      ], "accuracy");

      const align = DATA.alignment_summary;
      document.getElementById("alignmentMetrics").innerHTML = [
        metricCard("Model A acc", fmt(align.model_a_acc)),
        metricCard("Model B acc", fmt(align.model_b_acc)),
        metricCard("Midpoint before", fmt(align.midpoint_before_acc)),
        metricCard("Midpoint after", fmt(align.midpoint_after_acc), `barrier ${{fmt(align.barrier_after)}}`),
      ].join("");
      renderLineChart("alignmentChart", DATA.alignment_rows, "t", [
        {{ key: "before_loss", label: "before loss", color: "#e76f51" }},
        {{ key: "after_loss", label: "after loss", color: "#2a9d8f" }},
      ], "loss");

      const qBase = DATA.qwen_rows.find(r => Number(r.lambda) === 0);
      const qInstruct = DATA.qwen_rows.find(r => Number(r.lambda) === 1);
      const gsmBest = [...DATA.qwen_gsm8k_rows].sort((a, b) => Number(b.exact_match) - Number(a.exact_match) || Number(b.loose_exact_match) - Number(a.loose_exact_match))[0];
      const mmluBest = [...DATA.qwen_mmlu_rows].sort((a, b) => Number(b.accuracy) - Number(a.accuracy) || Number(a.avg_gold_nll) - Number(b.avg_gold_nll))[0];
      const humanEvalBest = [...DATA.qwen_humaneval_rows].sort((a, b) => Number(a.avg_solution_nll) - Number(b.avg_solution_nll))[0];
      const safetyBest = [...DATA.qwen_safety_rows].sort((a, b) => Number(a.avg_safety_nll) - Number(b.avg_safety_nll))[0];
      const qMultiMethods = [...DATA.qwen_multi_method_rows].sort((a, b) => Number(a.avg_nll) - Number(b.avg_nll));
      const qMultiBest = qMultiMethods[0];
      const qMultiLinear = DATA.qwen_multi_method_rows.find(r => r.method === "linear_average");
      const qMultiConflict = DATA.qwen_multi_conflict_rows[0];
      document.getElementById("qwenMetrics").innerHTML = [
        metricCard("Base instruction NLL", fmt(qBase.instruction_nll)),
        metricCard("Instruct instruction NLL", fmt(qInstruct.instruction_nll)),
        metricCard("Best avg lambda", fmt(bestQwen.lambda, 2), `avg NLL ${{fmt(bestQwen.avg_nll)}}`),
        metricCard("GSM8K strict best", fmt(gsmBest.exact_match), `lambda ${{fmt(gsmBest.lambda, 2)}}; loose ${{fmt(gsmBest.loose_exact_match)}}`),
        metricCard("MMLU best", fmt(mmluBest.accuracy), `lambda ${{fmt(mmluBest.lambda, 2)}}; ${{mmluBest.accuracy_count}}/${{mmluBest.examples}}`),
        metricCard("HumanEval NLL best", fmt(humanEvalBest.avg_solution_nll), `lambda ${{fmt(humanEvalBest.lambda, 2)}}`),
        metricCard("Safety NLL best", fmt(safetyBest.avg_safety_nll), `lambda ${{fmt(safetyBest.lambda, 2)}}`),
        metricCard("Multi-expert best", qMultiBest.method, `avg NLL ${{fmt(qMultiBest.avg_nll)}}`),
        metricCard("Linear average", fmt(qMultiLinear.avg_nll), `worst NLL ${{fmt(qMultiLinear.worst_nll)}}`),
        metricCard("Instruct/coder conflict", fmt(qMultiConflict.weighted_conflict), `cosine ${{fmt(qMultiConflict.cosine)}}`),
      ].join("");
      renderLineChart("qwenChart", DATA.qwen_rows, "lambda", [
        {{ key: "general_nll", label: "general", color: "#2a9d8f" }},
        {{ key: "instruction_nll", label: "instruction", color: "#e76f51" }},
        {{ key: "avg_nll", label: "average", color: "#264653" }},
        {{ key: "worst_nll", label: "worst", color: "#6d597a" }},
      ], "NLL");
      renderTable("qwenTable", DATA.qwen_rows, [
        {{ key: "lambda", label: "lambda", format: v => fmt(v, 2) }},
        {{ key: "general_nll", label: "general NLL", format: v => fmt(v) }},
        {{ key: "instruction_nll", label: "instruction NLL", format: v => fmt(v) }},
        {{ key: "avg_nll", label: "avg NLL", format: v => fmt(v) }},
        {{ key: "worst_nll", label: "worst NLL", format: v => fmt(v) }},
      ]);
      renderTable("qwenGsm8kTable", DATA.qwen_gsm8k_rows, [
        {{ key: "lambda", label: "lambda", format: v => fmt(v, 2) }},
        {{ key: "exact_match", label: "strict exact", format: v => fmt(v) }},
        {{ key: "loose_exact_match", label: "loose exact", format: v => fmt(v) }},
        {{ key: "exact_count", label: "strict count" }},
        {{ key: "examples", label: "examples" }},
      ]);
      renderTable("qwenMmluTable", DATA.qwen_mmlu_rows, [
        {{ key: "lambda", label: "lambda", format: v => fmt(v, 2) }},
        {{ key: "accuracy", label: "accuracy", format: v => fmt(v) }},
        {{ key: "accuracy_count", label: "correct" }},
        {{ key: "examples", label: "examples" }},
        {{ key: "avg_gold_nll", label: "gold NLL", format: v => fmt(v) }},
        {{ key: "avg_margin", label: "margin", format: v => fmt(v) }},
      ]);
      renderTable("qwenHumanEvalTable", DATA.qwen_humaneval_rows, [
        {{ key: "lambda", label: "lambda", format: v => fmt(v, 2) }},
        {{ key: "examples", label: "examples" }},
        {{ key: "solution_tokens", label: "tokens" }},
        {{ key: "avg_solution_nll", label: "solution NLL", format: v => fmt(v) }},
        {{ key: "mean_task_nll", label: "mean task NLL", format: v => fmt(v) }},
        {{ key: "median_task_nll", label: "median task NLL", format: v => fmt(v) }},
      ]);
      renderTable("qwenSafetyTable", DATA.qwen_safety_rows, [
        {{ key: "lambda", label: "lambda", format: v => fmt(v, 2) }},
        {{ key: "safe_examples", label: "safe" }},
        {{ key: "unsafe_examples", label: "unsafe" }},
        {{ key: "safe_response_nll", label: "safe NLL", format: v => fmt(v) }},
        {{ key: "unsafe_refusal_nll", label: "refusal NLL", format: v => fmt(v) }},
        {{ key: "avg_safety_nll", label: "avg NLL", format: v => fmt(v) }},
      ]);
      renderTable("qwenMultiTable", qMultiMethods, [
        {{ key: "method", label: "Method" }},
        {{ key: "alpha", label: "alpha", format: v => fmt(v, 2) }},
        {{ key: "beta", label: "beta", format: v => fmt(v, 2) }},
        {{ key: "general_nll", label: "general NLL", format: v => fmt(v) }},
        {{ key: "instruction_nll", label: "instruction NLL", format: v => fmt(v) }},
        {{ key: "code_nll", label: "code NLL", format: v => fmt(v) }},
        {{ key: "avg_nll", label: "avg NLL", format: v => fmt(v) }},
        {{ key: "worst_nll", label: "worst NLL", format: v => fmt(v) }},
      ]);
      renderTable("qwenConflictTable", DATA.qwen_multi_conflict_rows, [
        {{ key: "left", label: "Left" }},
        {{ key: "right", label: "Right" }},
        {{ key: "shared_tensors", label: "shared tensors" }},
        {{ key: "cosine", label: "cosine", format: v => fmt(v) }},
        {{ key: "sign_conflict", label: "sign conflict", format: v => fmt(v) }},
        {{ key: "weighted_conflict", label: "weighted conflict", format: v => fmt(v) }},
      ]);
    }}

    init();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a static dashboard for the model-merging study artifacts.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/dashboard"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = copy_assets(args.output_dir)
    data = load_data()
    (args.output_dir / "index.html").write_text(html_template(data, figures), encoding="utf-8")
    print(f"Wrote dashboard to {(args.output_dir / 'index.html').resolve()}")


if __name__ == "__main__":
    main()
