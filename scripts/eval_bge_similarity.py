import argparse
import csv
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Use HF mirror if not explicitly configured (keep consistent with eval_task_c.py)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from transformers import AutoModel, AutoTokenizer


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


def load_refs_by_id(refs_jsonl: str) -> Dict[str, List[str]]:
    refs_by_id: Dict[str, List[str]] = {}
    for row in iter_jsonl(refs_jsonl):
        sid = str(row.get("sentence_id") or row.get("id") or row.get("uid") or "").strip()
        if not sid:
            continue

        refs: List[str] = []
        if isinstance(row.get("refs"), list):
            refs.extend([str(x) for x in row["refs"] if str(x).strip()])

        for k, v in row.items():
            if not isinstance(k, str):
                continue
            if k.startswith("ref_suggestion") or k.startswith("ref_"):
                if isinstance(v, str) and v.strip():
                    refs.append(v.strip())

        uniq: List[str] = []
        seen = set()
        for r in refs:
            r2 = r.strip()
            if not r2 or r2 in seen:
                continue
            seen.add(r2)
            uniq.append(r2)

        if uniq:
            refs_by_id[sid] = uniq
    return refs_by_id


def load_genre_by_sentence_id(
    meta_csv: str,
    sentence_id_col: str = "sentence_id",
    genre_col: str = "genre_merged",
) -> Tuple[Dict[str, str], Dict[str, Any]]:
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


def normalize_hyp_text(s: str) -> str:
    s = s.strip()
    s = s.replace("\u3000", " ")
    s = " ".join(s.split())
    return s


def count_non_empty_lines(path: str) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


@dataclass
class BgeConfig:
    model_name: str
    device: str
    dtype: str
    max_length: int


class BgeSentenceEncoder:
    def __init__(self, cfg: BgeConfig) -> None:
        self.model_name = cfg.model_name
        self.device = cfg.device
        self.max_length = cfg.max_length

        torch_dtype = None
        if cfg.dtype == "auto":
            if cfg.device.startswith("cuda"):
                torch_dtype = torch.float16
        elif cfg.dtype == "fp16":
            torch_dtype = torch.float16
        elif cfg.dtype == "bf16":
            torch_dtype = torch.bfloat16
        elif cfg.dtype == "fp32":
            torch_dtype = torch.float32
        else:
            raise ValueError(f"Unknown dtype: {cfg.dtype}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(self.model_name, torch_dtype=torch_dtype)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, texts: List[str], batch_size: int) -> torch.Tensor:
        all_embeddings: List[torch.Tensor] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state

            mask = attention_mask.unsqueeze(-1)
            masked = last_hidden_state * mask
            summed = masked.sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            embeddings = summed / counts

            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.detach().cpu())

        if not all_embeddings:
            return torch.empty(0, 0)
        return torch.cat(all_embeddings, dim=0)


def _genre_acc_init() -> Dict[str, float]:
    return {
        "num_samples": 0.0,
        "sum_bge_sim": 0.0,
        "cnt_bge_sim": 0.0,
    }


def safe_mean(sum_v: float, cnt_v: int) -> Optional[float]:
    if cnt_v <= 0:
        return None
    return float(sum_v / cnt_v)


def compute_bge_similarity(
    hyps_jsonl: str,
    refs_jsonl: str,
    meta_csv: Optional[str],
    meta_sentence_id_col: str,
    meta_genre_col: str,
    encoder: BgeSentenceEncoder,
    batch_size: int,
    normalize_hyp: bool,
    progress_total: Optional[int] = None,
    sim_thresholds: Optional[List[float]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if sim_thresholds is None:
        sim_thresholds = [0.6, 0.7, 0.8]

    genre_by_id: Optional[Dict[str, str]] = None
    meta_stats: Optional[Dict[str, Any]] = None
    if meta_csv:
        genre_by_id, meta_stats = load_genre_by_sentence_id(
            meta_csv,
            sentence_id_col=meta_sentence_id_col,
            genre_col=meta_genre_col,
        )

    refs_by_id = load_refs_by_id(refs_jsonl) if refs_jsonl else {}

    all_ref_texts: List[str] = []
    ref_sid_index: List[str] = []
    for sid, refs in refs_by_id.items():
        for r in refs:
            all_ref_texts.append(r)
            ref_sid_index.append(sid)

    ref_embs_by_sid: Dict[str, List[torch.Tensor]] = {}
    if all_ref_texts:
        ref_embs = encoder.encode(all_ref_texts, batch_size=batch_size)
        for sid, emb in zip(ref_sid_index, ref_embs):
            ref_embs_by_sid.setdefault(sid, []).append(emb)

    for sid, emb_list in list(ref_embs_by_sid.items()):
        if not emb_list:
            continue
        ref_embs_by_sid[sid] = torch.stack(emb_list, dim=0)

    if progress_total is None:
        progress_total = count_non_empty_lines(hyps_jsonl)

    per_sample: List[Dict[str, Any]] = []
    all_sims: List[float] = []

    per_genre_acc: Dict[str, Dict[str, float]] = {}
    meta_unknown_genre_samples = 0
    meta_mapped_genre_samples = 0

    threshold_counts: Dict[float, int] = {t: 0 for t in sim_thresholds}

    pbar = tqdm(total=progress_total, desc="Computing BGE similarity", unit="sample")

    buf_ids: List[str] = []
    buf_hyps: List[str] = []

    def flush_buffer() -> None:
        nonlocal meta_unknown_genre_samples, meta_mapped_genre_samples
        if not buf_hyps:
            return

        hyp_embs = encoder.encode(buf_hyps, batch_size=batch_size)
        for idx, (sid, hyp) in enumerate(zip(buf_ids, buf_hyps)):
            row_out: Dict[str, Any] = {
                "sentence_id": sid,
                "hyp": hyp,
            }

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

            hyp_emb = hyp_embs[idx : idx + 1]
            ref_embs = ref_embs_by_sid.get(sid)
            best_sim: Optional[float] = None
            if ref_embs is not None and ref_embs.numel() > 0:
                sims = torch.matmul(hyp_emb, ref_embs.T)
                best_sim = float(sims.max().item())
                row_out["bge_sim"] = best_sim
                all_sims.append(best_sim)

                for t in sim_thresholds:
                    if best_sim >= t:
                        threshold_counts[t] += 1

                if gacc is not None:
                    gacc["sum_bge_sim"] += best_sim
                    gacc["cnt_bge_sim"] += 1.0

            if genre is not None:
                row_out["genre_merged"] = genre

            per_sample.append(row_out)

        buf_ids.clear()
        buf_hyps.clear()

    for row in iter_jsonl(hyps_jsonl):
        sid = str(row.get("sentence_id") or row.get("id") or row.get("uid") or "").strip()
        if not sid:
            continue
        hyp = row.get("hyp")
        if hyp is None:
            continue
        hyp = str(hyp)
        if normalize_hyp:
            hyp = normalize_hyp_text(hyp)

        buf_ids.append(sid)
        buf_hyps.append(hyp)
        pbar.update(1)

        if len(buf_hyps) >= batch_size:
            flush_buffer()

    flush_buffer()
    pbar.close()

    num_samples = len(per_sample)
    sims_sorted = sorted(all_sims)

    def percentile(values: List[float], p: float) -> Optional[float]:
        if not values:
            return None
        if p <= 0:
            return float(values[0])
        if p >= 100:
            return float(values[-1])
        k = (len(values) - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return float(values[int(k)])
        d0 = values[f] * (c - k)
        d1 = values[c] * (k - f)
        return float(d0 + d1)

    summary: Dict[str, Any] = {
        "num_samples": num_samples,
        "hyps_jsonl": hyps_jsonl,
        "refs_jsonl": refs_jsonl,
        "bge_model": encoder.model_name,
        "device": encoder.device,
        "batch_size": batch_size,
        "mean_bge_sim": safe_mean(sum(all_sims), len(all_sims)),
        "median_bge_sim": percentile(sims_sorted, 50.0),
        "bge_sim_p10": percentile(sims_sorted, 10.0),
        "bge_sim_p25": percentile(sims_sorted, 25.0),
        "bge_sim_p75": percentile(sims_sorted, 75.0),
        "bge_sim_p90": percentile(sims_sorted, 90.0),
        "bge_sim_threshold_stats": {
            str(t): (threshold_counts[t] / num_samples if num_samples > 0 else None) for t in sim_thresholds
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
                "mean_bge_sim": safe_mean(acc["sum_bge_sim"], int(acc["cnt_bge_sim"])),
            }

        summary.update(meta_stats)
        summary["meta_unknown_genre_samples"] = int(meta_unknown_genre_samples)
        summary["meta_mapped_genre_samples"] = int(meta_mapped_genre_samples)
        summary["by_genre_num_samples_sum"] = int(by_genre_num_samples_sum)
        summary["by_genre"] = by_genre

    return per_sample, summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Task C BGE sentence similarity metrics: semantic similarity between hyp and refs."
    )
    ap.add_argument("--hyps_jsonl", required=True, help="JSONL with at least {sentence_id, hyp}.")
    ap.add_argument(
        "--refs_jsonl",
        required=True,
        help="JSONL with {sentence_id, ref_suggestion_1/2/3 or refs:[...]}.",
    )
    ap.add_argument(
        "--out_jsonl",
        default="result/bge/task_c_bge_similarity.jsonl",
        help="Per-sample BGE similarity JSONL.",
    )
    ap.add_argument(
        "--out_json",
        default="result/bge/task_c_bge_similarity_summary.json",
        help="Dataset-level BGE similarity summary JSON.",
    )

    ap.add_argument(
        "--bge_model",
        default="BAAI/bge-large-zh-v1.5",
        help="BGE Chinese model name or path, e.g., BAAI/bge-large-zh-v1.5.",
    )
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0 ...")
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--max_length", type=int, default=256, help="Max sequence length for BGE encoder.")
    ap.add_argument("--batch_size", type=int, default=64, help="Batch size for encoding.")

    ap.add_argument("--normalize_hyp", action="store_true", help="Apply light normalization to hyp text.")
    ap.add_argument("--no_normalize_hyp", dest="normalize_hyp", action="store_false")
    ap.set_defaults(normalize_hyp=True)

    ap.add_argument(
        "--meta_csv",
        default=None,
        help="Optional metadata CSV (e.g. data_all_valid_senid_unique.csv) to attach genre_merged and compute by-genre summary.",
    )
    ap.add_argument("--meta_sentence_id_col", default="sentence_id", help="Column name for sentence_id in meta_csv.")
    ap.add_argument("--meta_genre_col", default="genre_merged", help="Column name for genre_merged in meta_csv.")

    ap.add_argument(
        "--progress_total",
        type=int,
        default=None,
        help="Optional total samples for progress bar. If omitted, will count hyps_jsonl lines once.",
    )
    ap.add_argument(
        "--sim_thresholds",
        default="0.6,0.7,0.8",
        help="Comma-separated thresholds for bge_sim_threshold_stats, e.g. '0.6,0.7,0.8'.",
    )

    args = ap.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    thresholds: List[float] = []
    for part in str(args.sim_thresholds).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            thresholds.append(float(part))
        except ValueError:
            continue
    if not thresholds:
        thresholds = [0.6, 0.7, 0.8]

    cfg = BgeConfig(
        model_name=args.bge_model,
        device=device,
        dtype=args.dtype,
        max_length=args.max_length,
    )
    encoder = BgeSentenceEncoder(cfg)

    total = args.progress_total
    if total is None:
        total = count_non_empty_lines(args.hyps_jsonl)

    per_sample, summary = compute_bge_similarity(
        hyps_jsonl=args.hyps_jsonl,
        refs_jsonl=args.refs_jsonl,
        meta_csv=args.meta_csv,
        meta_sentence_id_col=args.meta_sentence_id_col,
        meta_genre_col=args.meta_genre_col,
        encoder=encoder,
        batch_size=args.batch_size,
        normalize_hyp=args.normalize_hyp,
        progress_total=total,
        sim_thresholds=thresholds,
    )

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    write_jsonl(args.out_jsonl, per_sample)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

