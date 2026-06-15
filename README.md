# C-RANS

C-RANS is a Chinese register and appropriateness benchmark for evaluating how well language models understand and revise sentences across communicative contexts. The release contains annotated Chinese learner sentences, expert revisions, task splits, model predictions, evaluation scripts, benchmark scores, and paper figures.

The benchmark focuses on both form and contextual appropriateness:

- **Task A: Grammar Rating** - predict a 1-5 grammar score.
- **Task B: Naturalness / Appropriateness Rating** - predict a 1-5 contextual appropriateness score.
- **Task C: Revision Generation** - produce a revised sentence that is more natural and appropriate for the given context.

## Repository Structure

```text
c-rans-release/
├── README.md
├── requirements.txt
├── data/
│   └── c_rans_release.json
├── docs/
│   └── annotation-manual-release.pdf
├── figures/
│   ├── Figure1_taskab.tiff
│   ├── Figure2_GLEU.tiff
│   ├── Figure3_Cosine.tiff
│   ├── Figure4_LogEAP.tiff
│   ├── Figure5_NCD.tiff
│   └── Figure6_heatmap.tiff
├── results/
│   ├── benchmark_scores.csv
│   ├── task_ab_model_predictions/
│   │   ├── model_taskab_prediction_fewshot.csv
│   │   └── model_taskab_prediction_zeroshot.csv
│   └── task_c_model_outputs/
│       ├── claude_sonnet4.5_hyp_outputs.jsonl
│       ├── deepseekv3_hyp_outputs.jsonl
│       ├── ernie4.5-300b_hyp_outputs.jsonl
│       ├── gemini3.5_hyp_outputs.jsonl
│       ├── glm5_hyp_outputs.jsonl
│       ├── gpt5_hyp_outputs.jsonl
│       └── qwen3_hyp_outputs.jsonl
├── scripts/
│   ├── C-RANS_R_visualization.R
│   ├── compute_ncd_from_taskc.py
│   ├── eval_bge_similarity.py
│   ├── eval_task_c.py
│   └── evaluate_ab.py
└── splits/
    ├── dev_ids.txt
    └── test_ids.txt
```

## Dataset

The main dataset is stored in `data/c_rans_release.json`. It contains **6,358** sentence-level examples with unique `sentence_id` values.

| Split | Examples |
| --- | ---: |
| Dev | 1,155 |
| Test | 5,203 |

The data covers four register settings:

| Register | Examples |
| --- | ---: |
| spoken formal | 1,953 |
| spoken informal | 1,495 |
| written formal | 1,474 |
| written informal | 1,436 |

## Data Format

Each item in `c_rans_release.json` is a JSON object with the following fields:

| Field | Description |
| --- | --- |
| `id` | Original item identifier. |
| `sentence_id` | Unique sentence-level identifier used for evaluation and splits. |
| `text_id` | Source text identifier. |
| `file_id` | Source file identifier. |
| `file_path` | Original corpus path or source path. |
| `data_type` | Data source type, such as `exam essay`, `free writing`, `oral exam`, or `oral practice`. |
| `register` | Communicative register: `written formal`, `written informal`, `spoken formal`, or `spoken informal`. |
| `topic` | Topic or prompt category. |
| `full_text` | Full source text containing the sentence. |
| `sentence` | Original sentence to be rated or revised. |
| `grammar_rating` | Human grammar rating from 1 to 5. A small number of examples may contain `None`. |
| `naturalness_rating` | Human naturalness / appropriateness rating from 1 to 5. A small number of examples may contain `None`. |
| `revision` | Expert revised sentence. |
| `comment` | Annotation comment or error category. |

The annotation manual is available at `docs/annotation-manual-release.pdf`.

## Results

The `results/` directory contains released model outputs and benchmark scores.

- `results/benchmark_scores.csv`: per-sentence benchmark metrics and model identifiers.
- `results/task_ab_model_predictions/`: model predictions for Task A and Task B under zero-shot and few-shot settings.
- `results/task_c_model_outputs/`: generated revisions for Task C in JSONL format.

Task C model output files use one JSON object per line:

```json
{"sentence_id": "text#1806_#984065_1", "model": "GPT-5", "hyp": "感谢贵方在来函中给予我方的高度评价。"}
```

## Installation

Create a Python environment and install the common dependencies:

```bash
pip install numpy pandas torch transformers tqdm
```

Some metrics require downloading Hugging Face models. Use `--device cpu` for CPU-only evaluation, or `--device cuda` when a compatible GPU is available.

The visualization script requires R packages used by `scripts/C-RANS_R_visualization.R`, including common tidyverse plotting packages.

## Evaluation

### Task A and Task B

Use `scripts/evaluate_ab.py` to compute Quadratic Weighted Kappa (QWK) for grammar and appropriateness ratings.

Prediction CSV files should contain sentence identifiers and rating predictions. If your file uses different column names, convert them to the columns expected by the script before running:

```text
sentence_id,task_a_score,task_b_score
```

The gold file uses the release field names `grammar_rating` and `naturalness_rating`. `evaluate_ab.py` can read `data/c_rans_release.json` directly.

Example evaluation command:

```bash
python scripts/evaluate_ab.py \
  --pred-csv results/task_ab_model_predictions/model_taskab_prediction_zeroshot.csv \
  --gold-file data/c_rans_release.json \
  --test-ids splits/test_ids.txt \
  --model-name GPT5 \
  --output-dir results/eval_task_ab \
  --save-details
```

### Task C

Use `scripts/eval_task_c.py` to score generated revisions with GLEU, language-model perplexity, optional mutual-information style scores, and Expert-Aligned Perplexity (EAP).

Example:

```bash
python scripts/eval_task_c.py \
  --hyps_jsonl results/task_c_model_outputs/gpt5_hyp_outputs.jsonl \
  --refs_jsonl data/refs_task_c.jsonl \
  --out_jsonl results/eval_task_c/gpt5_metrics.jsonl \
  --out_json results/eval_task_c/gpt5_summary.json \
  --compute_eap
```

For semantic similarity with BGE embeddings:

```bash
python scripts/eval_bge_similarity.py \
  --hyps_jsonl results/task_c_model_outputs/gpt5_hyp_outputs.jsonl \
  --refs_jsonl data/refs_task_c.jsonl \
  --out_jsonl results/eval_task_c/gpt5_bge_similarity.jsonl \
  --out_json results/eval_task_c/gpt5_bge_similarity_summary.json
```

For Normalized Compression Distance (NCD):

```bash
python scripts/compute_ncd_from_taskc.py \
  --taskc_jsonl results/task_c_model_outputs/gpt5_hyp_outputs.jsonl \
  --refs_jsonl data/refs_task_c.jsonl \
  --out_csv results/eval_task_c/gpt5_ncd.csv
```

If `data/refs_task_c.jsonl` is not included in your copy of the release, it can be reconstructed from `data/c_rans_release.json` by mapping each `sentence_id` to the corresponding expert `revision`.
