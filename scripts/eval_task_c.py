"""Task C / Task D scoring: GLEU, LM perplexity, optional MI, and Expert-Aligned Perplexity (EAP).

This script scores **existing** hypotheses with a fixed causal language model (``scoring_lm``).
It does **not** generate text; it computes token-level negative log-likelihood and aggregates
to mean cross-entropy (nats/token) and perplexity (PPL).

**Inputs**

* ``hyps_jsonl`` (required): one JSON object per line with ``sentence_id`` and ``hyp`` (the
  candidate string). Other keys (e.g. ``id``, ``uid``) are accepted as sentence id fallbacks.
* ``refs_jsonl`` (optional): expert references per ``sentence_id``. Loads ``refs`` list and/or
  ``ref_suggestion_*`` / ``ref_*`` string fields; de-duplicates while preserving order.
* ``meta_csv`` (optional): maps ``sentence_id`` → ``genre_merged`` (majority vote) for
  per-genre summary statistics.

**Unconditional LM metrics (always for each ``hyp``)**

* ``xent_nats_per_token``, ``ppl``, ``lm_num_pred_tokens``: full-sequence mean CE and exp(CE).

**With ``refs_jsonl``**

* **GLEU** (optional char/space/llama tokenization): max over refs of a lightweight sentence GLEU.
* ``--compute_ref_ppl``: PPL of each ref alone; reports ``ref_best_ppl`` and ``ppl_gap_vs_ref_best``.
* ``--compute_mi``: MI-style term ``-(xent_cond - xent_uncond)`` with prefix ``ref`` + ``mi_sep``
  + ``hyp`` (legacy conditional form; **not** the EAP meeting template).
* ``--compute_eap`` (**Task D EAP**): conditional scoring with the meeting-aligned prompt::

    专家参考：
    {expert_reference}

    句子：
    {candidate_sentence}

  Only **candidate** tokens (the ``hyp`` line) contribute to the reported CE/PPL; the prefix
  provides context. Multiple refs → ``eap_xent_best`` = min CE, ``eap_xent_mean`` = mean CE;
  ``delta_xent = eap_xent_best - xent_uncond``; ``eap_ppl = exp(eap_xent_best)``.

**Outputs**

Default paths (under the current working directory, typically ``vllm/``): ``result/eap/task_c_metrics.jsonl``
(per line) and ``result/eap/task_c_metrics_summary.json`` (aggregates). Parent directories are
created automatically.

**Environment**

``HF_ENDPOINT`` defaults to ``https://hf-mirror.com`` if unset (must be set before Hugging Face Hub use).

See ``python eval_task_c.py --help`` for CLI flags and examples.
"""

import argparse
import csv
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
# If the environment didn't set a Hub endpoint, default to the commonly used mirror.
# Must be set before any HuggingFace Hub I/O happens.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from transformers import AutoModelForCausalLM, AutoTokenizer


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


_RE_MULTI_WS = re.compile(r"\s+")
_RE_PREFIX_MARKER = re.compile(r"^\s*【改写句】\s*", flags=re.MULTILINE)

# Task D (EAP): meeting-aligned prefix before candidate_sentence (expert in prompt, score only candidate tokens).
EAP_PREFIX_HEAD = "专家参考：\n"
EAP_PREFIX_MID = "\n\n句子：\n"


def eap_prefix_before_candidate(expert_reference: str) -> str:
    """Return prefix such that full prompt is: prefix + candidate_sentence."""
    return f"{EAP_PREFIX_HEAD}{expert_reference}{EAP_PREFIX_MID}"


def normalize_hyp_text(s: str) -> str:
    s = s.strip()
    s = _RE_PREFIX_MARKER.sub("", s).strip()
    # keep as single "sentence-like" string for scoring
    s = s.replace("\u3000", " ")
    # remove all whitespace characters for strict comparability
    s = _RE_MULTI_WS.sub("", s).strip()
    return s


def tokenize_for_gleu(text: str, mode: str, tokenizer: Optional[Any] = None) -> List[str]:
    if mode == "space":
        return [t for t in text.strip().split() if t]
    if mode == "char":
        # remove whitespace; treat each char as a token
        return [c for c in re.sub(r"\s+", "", text) if c]
    if mode in ("llama", "internlm"):
        if tokenizer is None:
            raise ValueError(f"gleu_tokenize mode '{mode}' requires a tokenizer instance")
        enc = tokenizer(text, add_special_tokens=False)
        input_ids = getattr(enc, "input_ids", None)
        if input_ids is None:
            input_ids = enc["input_ids"]
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        # Some tokenizers may return a batch dimension; unwrap if single-item batch.
        if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
            if len(input_ids) != 1:
                raise ValueError("Unexpected batched input_ids for single-string tokenization")
            input_ids = input_ids[0]
        return [str(tid) for tid in (input_ids or [])]
    raise ValueError(f"Unknown gleu_tokenize mode: {mode}")


def ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    if n <= 0:
        return []
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def sentence_gleu_simple(hyp_tokens: List[str], ref_tokens: List[str], max_n: int = 4) -> float:
    # A lightweight sentence-level GLEU:
    # For each n, compute overlap and take min(precision_n, recall_n), then average over n.
    # Returns a score in [0, 1].
    if not hyp_tokens and not ref_tokens:
        return 1.0
    if not hyp_tokens or not ref_tokens:
        return 0.0
    scores: List[float] = []
    for n in range(1, max_n + 1):
        hyp_ng = Counter(ngrams(hyp_tokens, n))
        ref_ng = Counter(ngrams(ref_tokens, n))
        hyp_total = sum(hyp_ng.values())
        ref_total = sum(ref_ng.values())
        if hyp_total == 0 or ref_total == 0:
            scores.append(0.0)
            continue
        overlap = sum((hyp_ng & ref_ng).values())
        p = overlap / hyp_total
        r = overlap / ref_total
        scores.append(min(p, r))
    return float(sum(scores) / max_n)


@dataclass
class LmScore:
    mean_xent: float  # nats / token
    ppl: float
    num_pred_tokens: int


class FixedChineseLmScorer:
    def __init__(
        self,
        model_name: str,
        device: str,
        dtype: str,
        max_length: int,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_length = max_length

        torch_dtype = None
        if dtype == "auto":
            if device.startswith("cuda"):
                torch_dtype = torch.float16
        elif dtype == "fp16":
            torch_dtype = torch.float16
        elif dtype == "bf16":
            torch_dtype = torch.bfloat16
        elif dtype == "fp32":
            torch_dtype = torch.float32
        else:
            raise ValueError(f"Unknown dtype: {dtype}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            # Most GPT-like Chinese LMs have no pad token; use eos as pad for batching.
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # transformers>=4.57 deprecates `torch_dtype` in favor of `dtype`
        if torch_dtype is None:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch_dtype, trust_remote_code=True)
        self.model.to(device)
        self.model.eval()

    @torch.inference_mode()
    def score_texts_batch(self, texts: List[str], batch_size: int) -> List[LmScore]:
        out: List[LmScore] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            # predict token t given <t
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_mask = attention_mask[:, 1:].contiguous()

            vocab_size = shift_logits.size(-1)
            token_losses = F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
                reduction="none",
            ).view(shift_labels.size())

            token_losses = token_losses * shift_mask
            denom = shift_mask.sum(dim=1).clamp(min=1)
            mean_xent = (token_losses.sum(dim=1) / denom).detach().cpu().tolist()
            num_pred_tokens = denom.detach().cpu().tolist()

            for mx, nt in zip(mean_xent, num_pred_tokens):
                mx = float(mx)
                nt = int(nt)
                out.append(LmScore(mean_xent=mx, ppl=float(math.exp(mx)), num_pred_tokens=nt))
        return out

    @torch.inference_mode()
    def mean_xent_conditional(
        self,
        hyp_text: str,
        prefix_text: Optional[str] = None,
        sep: str = "\n",
    ) -> Tuple[float, int]:
        """
        Return (mean_xent, num_pred_tokens) for hyp under a fixed causal LM.

        If prefix_text is provided, computes xent over hyp tokens only in the sequence:
          prefix_text + sep + hyp_text
        i.e. estimates -1/T * log P(hyp | prefix).

        Notes:
        - Uses add_special_tokens=False.
        - If the concatenated sequence exceeds max_length, it truncates from the left,
          keeping the last max_length tokens (best-effort when prefix is long).
        """
        hyp_ids = self.tokenizer(hyp_text, add_special_tokens=False).input_ids
        if prefix_text is None:
            prefix_ids: List[int] = []
            sep_ids: List[int] = []
        else:
            prefix_ids = self.tokenizer(prefix_text, add_special_tokens=False).input_ids
            sep_ids = self.tokenizer(sep, add_special_tokens=False).input_ids

        full_ids = prefix_ids + sep_ids + hyp_ids
        hyp_start = len(prefix_ids) + len(sep_ids)

        if len(full_ids) > self.max_length:
            cut = len(full_ids) - self.max_length
            full_ids = full_ids[cut:]
            hyp_start = max(0, hyp_start - cut)

        input_ids = torch.tensor([full_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids, device=self.device)
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        vocab_size = shift_logits.size(-1)
        token_losses = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            reduction="none",
        ).view(shift_labels.size(0), shift_labels.size(1))

        # shift_labels position j corresponds to predicting token index (j+1) in input_ids.
        # We want to include only tokens whose index >= hyp_start.
        # That means include positions j where (j+1) >= hyp_start  =>  j >= hyp_start-1
        start_pos = max(0, hyp_start - 1)
        token_losses = token_losses[:, start_pos:]
        num_pred_tokens = token_losses.numel()
        if num_pred_tokens <= 0:
            return float("inf"), 0

        mean_xent = float(token_losses.mean().item())
        return mean_xent, int(num_pred_tokens)


def load_refs_by_id(refs_jsonl: str) -> Dict[str, List[str]]:
    refs_by_id: Dict[str, List[str]] = {}
    for row in iter_jsonl(refs_jsonl):
        sid = str(row.get("sentence_id") or row.get("id") or row.get("uid") or "").strip()
        if not sid:
            continue

        refs: List[str] = []
        if isinstance(row.get("refs"), list):
            refs.extend([str(x) for x in row["refs"] if str(x).strip()])

        # common field names
        for k, v in row.items():
            if not isinstance(k, str):
                continue
            if k.startswith("ref_suggestion") or k.startswith("ref_"):
                if isinstance(v, str) and v.strip():
                    refs.append(v.strip())

        # stable ordering + de-dup
        uniq: List[str] = []
        seen = set()
        for r in refs:
            r2 = r.strip()
            # keep the same scoring normalization as hyps
            r2 = normalize_hyp_text(r2)
            if not r2 or r2 in seen:
                continue
            seen.add(r2)
            uniq.append(r2)

        if uniq:
            refs_by_id[sid] = uniq
    return refs_by_id


def count_non_empty_lines(path: str) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def load_sentence_ids_txt(path: str) -> set[str]:
    ids: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            sid = line.strip()
            if sid:
                ids.add(sid)
    return ids


def _genre_acc_init() -> Dict[str, float]:
    return {
        "num_samples": 0.0,
        "sum_gleu": 0.0,
        "cnt_gleu": 0.0,
        "sum_xent": 0.0,
        "cnt_xent": 0.0,
        "sum_ppl": 0.0,
        "cnt_ppl": 0.0,
        "sum_ref_best_ppl": 0.0,
        "cnt_ref_best_ppl": 0.0,
        "sum_ppl_gap": 0.0,
        "cnt_ppl_gap": 0.0,
        "sum_mi": 0.0,
        "cnt_mi": 0.0,
        "sum_eap_xent_best": 0.0,
        "cnt_eap_xent_best": 0.0,
        "sum_eap_xent_mean": 0.0,
        "cnt_eap_xent_mean": 0.0,
        "sum_eap_ppl": 0.0,
        "cnt_eap_ppl": 0.0,
        "sum_eap_ppl_mean": 0.0,
        "cnt_eap_ppl_mean": 0.0,
        "sum_delta_xent": 0.0,
        "cnt_delta_xent": 0.0,
    }


def load_genre_by_sentence_id(
    meta_csv: str,
    sentence_id_col: str = "sentence_id",
    genre_col: str = "genre_merged",
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    Build a mapping: sentence_id -> genre_merged from a metadata CSV.

    Notes:
    - The metadata CSV can contain multiple rows per sentence_id (e.g. multiple annotations).
      We aggregate via majority vote over non-empty genre values.
    """
    by_sid: Dict[str, Counter] = {}
    genres_seen: Counter = Counter()
    num_rows = 0
    num_rows_missing_sid = 0
    num_rows_missing_genre = 0

    with open(meta_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"meta_csv has no header row: {meta_csv}")
        if sentence_id_col not in reader.fieldnames:
            raise KeyError(f"meta_csv missing column {sentence_id_col!r}: {meta_csv}")
        if genre_col not in reader.fieldnames:
            raise KeyError(f"meta_csv missing column {genre_col!r}: {meta_csv}")

        for row in reader:
            num_rows += 1
            sid = str(row.get(sentence_id_col, "")).strip()
            if not sid:
                num_rows_missing_sid += 1
                continue
            genre = str(row.get(genre_col, "")).strip()
            if not genre:
                num_rows_missing_genre += 1
                continue

            if sid not in by_sid:
                by_sid[sid] = Counter()
            by_sid[sid][genre] += 1
            genres_seen[genre] += 1

    mapping: Dict[str, str] = {}
    conflict_sids = 0
    for sid, c in by_sid.items():
        if not c:
            continue
        if len(c) > 1:
            conflict_sids += 1
        # deterministic tie-breaker: highest count, then lexicographic
        best = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        mapping[sid] = best

    stats: Dict[str, Any] = {
        "meta_csv": meta_csv,
        "meta_sentence_id_col": sentence_id_col,
        "meta_genre_col": genre_col,
        "meta_rows_total": num_rows,
        "meta_rows_missing_sentence_id": num_rows_missing_sid,
        "meta_rows_missing_genre": num_rows_missing_genre,
        "meta_unique_sentence_ids": len(by_sid),
        "meta_conflict_sentence_ids": conflict_sids,
        "meta_genres_seen": dict(genres_seen),
    }
    return mapping, stats


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Task C: GLEU + LM PPL (unconditional). Optional: ref PPL gap, MI (--compute_mi), "
            "Task D EAP (--compute_eap) with meeting prompt 专家参考/句子. Scoring only; no generation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Unconditional PPL only (default output: result/eap/*.json*)
  python eval_task_c.py --hyps_jsonl data/hyp_10.jsonl

  # GLEU + Task D EAP (needs refs aligned by sentence_id)
  python eval_task_c.py --hyps_jsonl data/hyp_10.jsonl --refs_jsonl data/refs_task_c.jsonl \\
    --compute_eap

  # Full optional LM metrics
  python eval_task_c.py --hyps_jsonl data/hyp_10.jsonl --refs_jsonl data/refs_task_c.jsonl \\
    --compute_eap --compute_mi --compute_ref_ppl --meta_csv path/to/meta.csv
""",
    )
    ap.add_argument("--hyps_jsonl", default="data/hyp_10.jsonl", help="JSONL with at least {sentence_id, hyp}.")
    ap.add_argument(
        "--refs_jsonl",
        default=None,
        help="Optional JSONL with {sentence_id, ref_suggestion_1/2/3 or refs:[...]}. Enables GLEU + MI.",
    )
    ap.add_argument(
        "--out_jsonl",
        default="result/eap/task_c_metrics.jsonl",
        help="Per-sample metrics JSONL (default under result/eap/).",
    )
    ap.add_argument(
        "--out_json",
        default="result/eap/task_c_metrics_summary.json",
        help="Dataset-level summary JSON (default under result/eap/).",
    )

    ap.add_argument("--normalize_hyp", action="store_true", help="Apply light normalization to hyp text.")
    ap.add_argument("--no_normalize_hyp", dest="normalize_hyp", action="store_false")
    ap.set_defaults(normalize_hyp=True)

    ap.add_argument(
        "--gleu_tokenize",
        "--gleu-tokenize",
        choices=["char", "space", "llama", "internlm"],
        default="char",
        help="Tokenization for sentence-level GLEU: char (default) | space | llama | internlm (use scoring LM tokenizer IDs).",
    )
    ap.add_argument("--gleu_max_n", type=int, default=4)

    ap.add_argument(
        "--scoring_lm",
        default="/root/autodl-tmp/huggingface_cache/internlm2_5-7b-base",
        help="Fixed LM for cross-entropy/PPL/MI/EAP scoring (Task D EAP defaults to local internlm2_5-7b-chat).",
    )
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0 ...")
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--lm_max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument(
        "--progress_total",
        type=int,
        default=None,
        help="Optional total samples for progress bar. If omitted, will count hyps_jsonl lines once.",
    )

    ap.add_argument("--compute_mi", action="store_true", help="Compute MI approximation via LM (requires refs).")
    ap.add_argument("--compute_ref_ppl", action="store_true", help="Also compute PPL(ref_best) if refs exist.")
    ap.add_argument(
        "--compute_eap",
        action="store_true",
        help="Task D: Expert-Aligned Perplexity — meeting prompt 专家参考/句子, score candidate tokens only (requires refs_jsonl).",
    )
    ap.add_argument("--mi_sep", default="\n", help="Separator inserted between ref and hyp for conditional scoring.")

    ap.add_argument(
        "--meta_csv",
        default=None,
        help="Optional metadata CSV (e.g. data_all_valid_senid_unique.csv) to attach genre_merged and compute by-genre summary.",
    )
    ap.add_argument("--meta_sentence_id_col", default="sentence_id", help="Column name for sentence_id in meta_csv.")
    ap.add_argument("--meta_genre_col", default="genre_merged", help="Column name for genre_merged in meta_csv.")
    ap.add_argument(
        "--sentence_ids_txt",
        default=None,
        help="Optional txt file (one sentence_id per line). If set, only score rows whose sentence_id is in this set.",
    )

    args = ap.parse_args()

    for _out_path in (args.out_jsonl, args.out_json):
        _parent = os.path.dirname(_out_path)
        if _parent:
            os.makedirs(_parent, exist_ok=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    genre_by_id: Optional[Dict[str, str]] = None
    meta_stats: Optional[Dict[str, Any]] = None
    if args.meta_csv:
        genre_by_id, meta_stats = load_genre_by_sentence_id(
            args.meta_csv,
            sentence_id_col=args.meta_sentence_id_col,
            genre_col=args.meta_genre_col,
        )

    refs_by_id: Optional[Dict[str, List[str]]] = None
    if args.refs_jsonl:
        refs_by_id = load_refs_by_id(args.refs_jsonl)
    allowed_sentence_ids: Optional[set[str]] = None
    if args.sentence_ids_txt:
        allowed_sentence_ids = load_sentence_ids_txt(args.sentence_ids_txt)

    scorer = FixedChineseLmScorer(
        model_name=args.scoring_lm,
        device=device,
        dtype=args.dtype,
        max_length=args.lm_max_length,
    )
    gleu_tokenizer = scorer.tokenizer if args.gleu_tokenize in ("llama", "internlm") else None

    # Optional: prepare GLEU + optional MI and ref_ppl
    can_use_refs = refs_by_id is not None and len(refs_by_id) > 0
    do_gleu = can_use_refs
    do_mi = bool(args.compute_mi and can_use_refs)
    do_ref_ppl = bool(args.compute_ref_ppl and can_use_refs)
    do_eap = bool(args.compute_eap and can_use_refs)

    total = args.progress_total
    if total is None:
        total = count_non_empty_lines(args.hyps_jsonl)

    # Stream: read hyps -> score LM in batches -> compute ref-based metrics -> write per-sample JSONL incrementally.
    num_samples = 0
    sum_gleu = 0.0
    cnt_gleu = 0
    sum_xent = 0.0
    cnt_xent = 0
    sum_ppl = 0.0
    cnt_ppl = 0
    sum_ref_best_ppl = 0.0
    cnt_ref_best_ppl = 0
    sum_ppl_gap = 0.0
    cnt_ppl_gap = 0
    sum_mi = 0.0
    cnt_mi = 0
    sum_eap_xent_best = 0.0
    cnt_eap_xent_best = 0
    sum_eap_xent_mean = 0.0
    cnt_eap_xent_mean = 0
    sum_eap_ppl = 0.0
    cnt_eap_ppl = 0
    sum_eap_ppl_mean = 0.0
    cnt_eap_ppl_mean = 0
    sum_delta_xent = 0.0
    cnt_delta_xent = 0

    per_genre_acc: Dict[str, Dict[str, float]] = {}
    meta_unknown_genre_samples = 0
    meta_mapped_genre_samples = 0

    buf_ids: List[str] = []
    buf_hyps: List[str] = []

    pbar = tqdm(total=total, desc="Scoring hyps", unit="sample")
    with open(args.out_jsonl, "w", encoding="utf-8") as out_f:
        def flush_buffer() -> None:
            nonlocal num_samples, sum_gleu, cnt_gleu, sum_xent, cnt_xent, sum_ppl, cnt_ppl
            nonlocal sum_ref_best_ppl, cnt_ref_best_ppl, sum_ppl_gap, cnt_ppl_gap, sum_mi, cnt_mi
            nonlocal sum_eap_xent_best, cnt_eap_xent_best, sum_eap_xent_mean, cnt_eap_xent_mean
            nonlocal sum_eap_ppl, cnt_eap_ppl, sum_eap_ppl_mean, cnt_eap_ppl_mean, sum_delta_xent, cnt_delta_xent
            nonlocal meta_unknown_genre_samples, meta_mapped_genre_samples

            if not buf_hyps:
                return

            hyp_scores = scorer.score_texts_batch(buf_hyps, batch_size=len(buf_hyps))
            for sid, hyp, hs in zip(buf_ids, buf_hyps, hyp_scores):
                genre = None
                gacc: Optional[Dict[str, float]] = None
                if genre_by_id is not None:
                    genre = genre_by_id.get(sid) or "UNKNOWN"
                    if genre == "UNKNOWN":
                        meta_unknown_genre_samples += 1
                    else:
                        meta_mapped_genre_samples += 1
                    gacc = per_genre_acc.setdefault(genre, _genre_acc_init())
                    gacc["num_samples"] += 1.0

                row_out: Dict[str, Any] = {
                    "sentence_id": sid,
                    "hyp": hyp,
                    "xent_nats_per_token": hs.mean_xent,
                    "ppl": hs.ppl,
                    "lm_num_pred_tokens": hs.num_pred_tokens,
                }
                if genre is not None:
                    row_out["genre_merged"] = genre

                num_samples += 1
                if math.isfinite(hs.mean_xent):
                    sum_xent += hs.mean_xent
                    cnt_xent += 1
                    if gacc is not None:
                        gacc["sum_xent"] += float(hs.mean_xent)
                        gacc["cnt_xent"] += 1.0
                if math.isfinite(hs.ppl):
                    sum_ppl += hs.ppl
                    cnt_ppl += 1
                    if gacc is not None:
                        gacc["sum_ppl"] += float(hs.ppl)
                        gacc["cnt_ppl"] += 1.0

                refs = refs_by_id.get(sid) if refs_by_id else None
                if refs:
                    if do_gleu:
                        hyp_toks = tokenize_for_gleu(hyp, args.gleu_tokenize, tokenizer=gleu_tokenizer)
                        best_g = 0.0
                        for ref in refs:
                            ref_toks = tokenize_for_gleu(ref, args.gleu_tokenize, tokenizer=gleu_tokenizer)
                            g = sentence_gleu_simple(hyp_toks, ref_toks, max_n=args.gleu_max_n)
                            if g > best_g:
                                best_g = g
                        row_out["gleu"] = float(best_g)
                        sum_gleu += float(best_g)
                        cnt_gleu += 1
                        if gacc is not None:
                            gacc["sum_gleu"] += float(best_g)
                            gacc["cnt_gleu"] += 1.0

                    if do_ref_ppl:
                        ref_scores = scorer.score_texts_batch(refs, batch_size=min(args.batch_size, len(refs)))
                        best_ref = min(ref_scores, key=lambda s: s.mean_xent)
                        row_out["ref_best_ppl"] = float(best_ref.ppl)
                        row_out["ppl_gap_vs_ref_best"] = float(hs.ppl - best_ref.ppl)
                        if math.isfinite(best_ref.ppl):
                            sum_ref_best_ppl += float(best_ref.ppl)
                            cnt_ref_best_ppl += 1
                            if gacc is not None:
                                gacc["sum_ref_best_ppl"] += float(best_ref.ppl)
                                gacc["cnt_ref_best_ppl"] += 1.0
                        if math.isfinite(hs.ppl) and math.isfinite(best_ref.ppl):
                            sum_ppl_gap += float(hs.ppl - best_ref.ppl)
                            cnt_ppl_gap += 1
                            if gacc is not None:
                                gacc["sum_ppl_gap"] += float(hs.ppl - best_ref.ppl)
                                gacc["cnt_ppl_gap"] += 1.0

                    if do_mi:
                        xent_uncond_mi = hs.mean_xent
                        best_mi = -float("inf")
                        for ref in refs:
                            xent_cond, nt = scorer.mean_xent_conditional(hyp, prefix_text=ref, sep=args.mi_sep)
                            if nt <= 0 or not math.isfinite(xent_cond) or not math.isfinite(xent_uncond_mi):
                                continue
                            mi = -(xent_cond - xent_uncond_mi)
                            if mi > best_mi:
                                best_mi = mi
                        if best_mi != -float("inf"):
                            row_out["mi_nats_per_token"] = float(best_mi)
                            sum_mi += float(best_mi)
                            cnt_mi += 1
                            if gacc is not None:
                                gacc["sum_mi"] += float(best_mi)
                                gacc["cnt_mi"] += 1.0

                    if do_eap:
                        xent_uncond = hs.mean_xent
                        cond_xents: List[float] = []
                        for ref in refs:
                            pref = eap_prefix_before_candidate(ref)
                            xent_cond, nt = scorer.mean_xent_conditional(hyp, prefix_text=pref, sep="")
                            if nt > 0 and math.isfinite(xent_cond):
                                cond_xents.append(float(xent_cond))
                        if cond_xents:
                            eap_best = min(cond_xents)
                            eap_mean = float(sum(cond_xents) / len(cond_xents))
                            d_xent = float(eap_best - xent_uncond)
                            eppl = float(math.exp(eap_best))
                            eppl_m = float(math.exp(eap_mean))
                            row_out["xent_uncond"] = float(xent_uncond)
                            row_out["eap_xent_best"] = float(eap_best)
                            row_out["eap_xent_mean"] = float(eap_mean)
                            row_out["delta_xent"] = d_xent
                            row_out["eap_ppl"] = eppl
                            row_out["eap_ppl_mean"] = eppl_m
                            row_out["eap_num_expert_refs_scored"] = int(len(cond_xents))

                            if math.isfinite(eap_best):
                                sum_eap_xent_best += eap_best
                                cnt_eap_xent_best += 1
                                if gacc is not None:
                                    gacc["sum_eap_xent_best"] += float(eap_best)
                                    gacc["cnt_eap_xent_best"] += 1.0
                            if math.isfinite(eap_mean):
                                sum_eap_xent_mean += eap_mean
                                cnt_eap_xent_mean += 1
                                if gacc is not None:
                                    gacc["sum_eap_xent_mean"] += float(eap_mean)
                                    gacc["cnt_eap_xent_mean"] += 1.0
                            if math.isfinite(eppl):
                                sum_eap_ppl += eppl
                                cnt_eap_ppl += 1
                                if gacc is not None:
                                    gacc["sum_eap_ppl"] += float(eppl)
                                    gacc["cnt_eap_ppl"] += 1.0
                            if math.isfinite(eppl_m):
                                sum_eap_ppl_mean += eppl_m
                                cnt_eap_ppl_mean += 1
                                if gacc is not None:
                                    gacc["sum_eap_ppl_mean"] += float(eppl_m)
                                    gacc["cnt_eap_ppl_mean"] += 1.0
                            if math.isfinite(d_xent):
                                sum_delta_xent += d_xent
                                cnt_delta_xent += 1
                                if gacc is not None:
                                    gacc["sum_delta_xent"] += float(d_xent)
                                    gacc["cnt_delta_xent"] += 1.0

                out_f.write(json.dumps(row_out, ensure_ascii=False) + "\n")

            buf_ids.clear()
            buf_hyps.clear()

        for row in iter_jsonl(args.hyps_jsonl):
            sid = str(row.get("sentence_id") or row.get("id") or row.get("uid") or "").strip()
            if not sid:
                continue
            if allowed_sentence_ids is not None and sid not in allowed_sentence_ids:
                continue
            hyp = row.get("hyp")
            if hyp is None:
                continue
            hyp = str(hyp)
            if args.normalize_hyp:
                hyp = normalize_hyp_text(hyp)

            buf_ids.append(sid)
            buf_hyps.append(hyp)
            pbar.update(1)

            if len(buf_hyps) >= args.batch_size:
                flush_buffer()

        flush_buffer()

    pbar.close()

    def safe_mean(sum_v: float, cnt_v: int) -> Optional[float]:
        if cnt_v <= 0:
            return None
        return float(sum_v / cnt_v)

    summary: Dict[str, Any] = {
        "num_samples": num_samples,
        "hyps_jsonl": args.hyps_jsonl,
        "sentence_ids_txt": args.sentence_ids_txt,
        "num_allowed_sentence_ids": (len(allowed_sentence_ids) if allowed_sentence_ids is not None else None),
        "refs_jsonl": args.refs_jsonl,
        "scoring_lm": args.scoring_lm,
        "device": device,
        "dtype": args.dtype,
        "lm_max_length": args.lm_max_length,
        "gleu_tokenize": args.gleu_tokenize if do_gleu else None,
        "gleu_max_n": args.gleu_max_n if do_gleu else None,
        "mean_gleu": safe_mean(sum_gleu, cnt_gleu),
        "mean_xent_nats_per_token": safe_mean(sum_xent, cnt_xent),
        "mean_ppl": safe_mean(sum_ppl, cnt_ppl),
        "mean_ref_best_ppl": safe_mean(sum_ref_best_ppl, cnt_ref_best_ppl),
        "mean_ppl_gap_vs_ref_best": safe_mean(sum_ppl_gap, cnt_ppl_gap),
        "mean_mi_nats_per_token": safe_mean(sum_mi, cnt_mi),
        "compute_eap": bool(args.compute_eap),
        "eap_prompt_prefix_head": EAP_PREFIX_HEAD,
        "eap_prompt_mid_before_candidate": EAP_PREFIX_MID,
        "mean_eap_xent_best": safe_mean(sum_eap_xent_best, cnt_eap_xent_best),
        "mean_eap_xent_mean": safe_mean(sum_eap_xent_mean, cnt_eap_xent_mean),
        "mean_eap_ppl": safe_mean(sum_eap_ppl, cnt_eap_ppl),
        "mean_eap_ppl_mean": safe_mean(sum_eap_ppl_mean, cnt_eap_ppl_mean),
        "mean_delta_xent": safe_mean(sum_delta_xent, cnt_delta_xent),
        "task_d_eap_acceptance": {
            "ordering_hypothesis": "expert < LLM_rewrite < student in EAP_PPL (lower is better) under same sentence_id and expert_reference.",
            "how_to_verify": (
                "Run eval_task_c.py separately on hyps_jsonl for student, each LLM, and expert-as-candidate; "
                "compare mean_eap_ppl (or per-genre). Single hyps_jsonl run does not prove the three-way ordering."
            ),
            "stability_vs_uncond": (
                "Compare dispersion of eap_ppl vs ppl within genre_merged (e.g. IQR/std); plan expects EAP often more stable."
            ),
        },
    }

    if meta_stats is not None:
        by_genre: Dict[str, Any] = {}
        by_genre_num_samples_sum = 0
        for genre, acc in sorted(per_genre_acc.items(), key=lambda kv: kv[0]):
            n = int(acc["num_samples"])
            by_genre_num_samples_sum += n
            by_genre[genre] = {
                "num_samples": n,
                "mean_gleu": safe_mean(acc["sum_gleu"], int(acc["cnt_gleu"])),
                "mean_xent_nats_per_token": safe_mean(acc["sum_xent"], int(acc["cnt_xent"])),
                "mean_ppl": safe_mean(acc["sum_ppl"], int(acc["cnt_ppl"])),
                "mean_ref_best_ppl": safe_mean(acc["sum_ref_best_ppl"], int(acc["cnt_ref_best_ppl"])),
                "mean_ppl_gap_vs_ref_best": safe_mean(acc["sum_ppl_gap"], int(acc["cnt_ppl_gap"])),
                "mean_mi_nats_per_token": safe_mean(acc["sum_mi"], int(acc["cnt_mi"])),
                "mean_eap_xent_best": safe_mean(acc["sum_eap_xent_best"], int(acc["cnt_eap_xent_best"])),
                "mean_eap_xent_mean": safe_mean(acc["sum_eap_xent_mean"], int(acc["cnt_eap_xent_mean"])),
                "mean_eap_ppl": safe_mean(acc["sum_eap_ppl"], int(acc["cnt_eap_ppl"])),
                "mean_eap_ppl_mean": safe_mean(acc["sum_eap_ppl_mean"], int(acc["cnt_eap_ppl_mean"])),
                "mean_delta_xent": safe_mean(acc["sum_delta_xent"], int(acc["cnt_delta_xent"])),
            }

        summary.update(meta_stats)
        summary["meta_unknown_genre_samples"] = int(meta_unknown_genre_samples)
        summary["meta_mapped_genre_samples"] = int(meta_mapped_genre_samples)
        summary["by_genre_num_samples_sum"] = int(by_genre_num_samples_sum)
        summary["by_genre"] = by_genre

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

