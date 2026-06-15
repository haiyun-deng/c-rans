#!/usr/bin/env python3
"""
Compute Normalized Compression Distance (NCD) between each model sentence ("hyp")
and expert references ("refs") using standard compressors (gzip or lzma).

Inputs:
  - --taskc_jsonl: a *_taskc_with_genre.jsonl file containing at least:
      { "sentence_id": str, "hyp": str, "genre_merged": str? }
    (If genre_merged is missing, it will be left blank.)
  - --refs_jsonl: refs_task_c.jsonl containing:
      { "sentence_id": str, "refs": [str, ...] }

Outputs:
  - --out_csv: per-sentence CSV with NCD and length stats.

NCD definition:
  NCD(x, y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))
where C(s) is compressed byte length and xy is concatenation with a delimiter.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import lzma
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _utf8_bytes(s: str) -> bytes:
    # Keep exact bytes; do not normalize to avoid mixing signals.
    return (s or "").encode("utf-8", errors="replace")


def _gzip_len(data: bytes, *, level: int = 9) -> int:
    # mtime=0 for determinism across runs.
    return len(gzip.compress(data, compresslevel=level, mtime=0))


def _lzma_len(data: bytes, *, preset: int = 6) -> int:
    return len(lzma.compress(data, preset=preset, format=lzma.FORMAT_XZ))


def _compress_len(data: bytes, *, compressor: str, gzip_level: int, lzma_preset: int) -> int:
    if compressor == "gzip":
        return _gzip_len(data, level=gzip_level)
    if compressor == "lzma":
        return _lzma_len(data, preset=lzma_preset)
    raise ValueError(f"Unknown compressor: {compressor}")


def _ncd(c_x: int, c_y: int, c_xy: int) -> Optional[float]:
    if c_x <= 0 or c_y <= 0:
        return None
    den = max(c_x, c_y)
    num = c_xy - min(c_x, c_y)
    if den <= 0:
        return None
    v = num / den
    if not math.isfinite(v):
        return None
    # Theoretical range is [0, 1+epsilon]; clamp to [0, 1] for reporting stability.
    if v < 0:
        v = 0.0
    if v > 1:
        v = 1.0
    return float(v)


def _char_len_for_binning(text: str) -> int:
    # "字数" here: count Unicode codepoints excluding whitespace.
    return sum(1 for ch in (text or "") if not ch.isspace())


@dataclass
class RefPickResult:
    ref_text: str
    c_ref: int
    c_pair: int
    ncd: Optional[float]


def _pick_ref(
    hyp: str,
    refs: List[str],
    *,
    compressor: str,
    gzip_level: int,
    lzma_preset: int,
    strategy: str,
) -> Optional[RefPickResult]:
    if not refs:
        return None
    hyp_b = _utf8_bytes(hyp)
    c_h = _compress_len(hyp_b, compressor=compressor, gzip_level=gzip_level, lzma_preset=lzma_preset)

    delim = b"\n<<<NCD_DELIM>>>\n"
    best: Optional[RefPickResult] = None
    sum_ncd = 0.0
    n_ncd = 0
    sum_c_ref = 0
    sum_c_pair = 0

    for r in refs:
        r_b = _utf8_bytes(r)
        c_r = _compress_len(r_b, compressor=compressor, gzip_level=gzip_level, lzma_preset=lzma_preset)
        c_xy = _compress_len(hyp_b + delim + r_b, compressor=compressor, gzip_level=gzip_level, lzma_preset=lzma_preset)
        ncd = _ncd(c_h, c_r, c_xy)

        if ncd is not None:
            sum_ncd += ncd
            n_ncd += 1
        sum_c_ref += c_r
        sum_c_pair += c_xy

        if strategy == "first":
            return RefPickResult(ref_text=r, c_ref=c_r, c_pair=c_xy, ncd=ncd)
        if strategy == "min_ncd":
            if best is None:
                best = RefPickResult(ref_text=r, c_ref=c_r, c_pair=c_xy, ncd=ncd)
            else:
                # Prefer defined ncd; then smaller; break ties by smaller c_pair.
                if best.ncd is None and ncd is not None:
                    best = RefPickResult(ref_text=r, c_ref=c_r, c_pair=c_xy, ncd=ncd)
                elif best.ncd is not None and ncd is not None:
                    if ncd < best.ncd - 1e-12 or (abs(ncd - best.ncd) <= 1e-12 and c_xy < best.c_pair):
                        best = RefPickResult(ref_text=r, c_ref=c_r, c_pair=c_xy, ncd=ncd)
            continue

    if strategy == "min_ncd":
        return best
    if strategy == "mean_ncd":
        # "mean_ncd" uses the first ref text (for display) but averages NCD across refs.
        # For C(ref)/C(pair) we report mean over refs for interpretability.
        ref0 = refs[0]
        mean_ncd = (sum_ncd / n_ncd) if n_ncd > 0 else None
        mean_c_ref = int(round(sum_c_ref / len(refs)))
        mean_c_pair = int(round(sum_c_pair / len(refs)))
        return RefPickResult(ref_text=ref0, c_ref=mean_c_ref, c_pair=mean_c_pair, ncd=mean_ncd)

    raise ValueError(f"Unknown --ref_strategy: {strategy}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute pairwise NCD for Task C outputs (hyp vs expert refs).")
    ap.add_argument("--taskc_jsonl", type=Path, required=True, help="Path to *_taskc_with_genre.jsonl")
    ap.add_argument("--refs_jsonl", type=Path, default=Path("vllm/data/refs_task_c.jsonl"), help="Path to refs_task_c.jsonl")
    ap.add_argument(
        "--out_csv",
        type=Path,
        default=None,
        help="Output CSV path (default: vllm/result/ncd/<model>_ncd_<compressor>.csv).",
    )
    ap.add_argument("--compressor", choices=["gzip", "lzma"], default="gzip", help="Compressor for NCD.")
    ap.add_argument("--gzip_level", type=int, default=9, help="gzip compress level (1-9).")
    ap.add_argument("--lzma_preset", type=int, default=6, help="lzma preset (0-9).")
    ap.add_argument(
        "--ref_strategy",
        choices=["first", "min_ncd", "mean_ncd"],
        default="min_ncd",
        help="How to pick/aggregate multi-reference refs for each sentence.",
    )
    args = ap.parse_args()

    if args.compressor == "gzip" and not (1 <= args.gzip_level <= 9):
        raise SystemExit(f"[ERR] --gzip_level must be 1..9, got {args.gzip_level}")
    if args.compressor == "lzma" and not (0 <= args.lzma_preset <= 9):
        raise SystemExit(f"[ERR] --lzma_preset must be 0..9, got {args.lzma_preset}")

    # Load refs map.
    refs_map: Dict[str, List[str]] = {}
    for obj in _iter_jsonl(args.refs_jsonl):
        sid = str(obj.get("sentence_id") or "").strip()
        if not sid:
            continue
        refs = obj.get("refs")
        if isinstance(refs, list):
            refs_map[sid] = [str(x) for x in refs if str(x).strip()]

    in_path = args.taskc_jsonl
    out_csv = args.out_csv
    if out_csv is None:
        # Default: vllm/result/ncd/<model>_ncd_<compressor>.csv
        base_dir = Path(__file__).resolve().parent
        out_dir = base_dir / "result" / "ncd"
        out_dir.mkdir(parents=True, exist_ok=True)

        name = in_path.name
        if name.endswith("_taskc_with_genre.jsonl"):
            model = name[: -len("_taskc_with_genre.jsonl")]
        elif name.endswith("_taskc.jsonl"):
            model = name[: -len("_taskc.jsonl")]
        else:
            model = in_path.stem
        out_csv = out_dir / f"{model}_ncd_{args.compressor}.csv"
    else:
        out_csv = Path(out_csv)
        if out_csv.parent:
            out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sentence_id",
        "genre_merged",
        "len_chars_hyp",
        "len_bytes_hyp",
        "len_chars_ref",
        "len_bytes_ref",
        "compressor",
        "c_hyp",
        "c_ref",
        "c_pair",
        "ncd",
        "hyp",
        "ref",
        "ref_strategy",
        "num_refs",
    ]

    n_rows = 0
    n_missing_ref = 0
    n_defined = 0

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for obj in _iter_jsonl(in_path):
            n_rows += 1
            sid = str(obj.get("sentence_id") or "").strip()
            hyp = str(obj.get("hyp") or "")
            genre = str(obj.get("genre_merged") or "").strip()

            refs = refs_map.get(sid) or []
            pick = _pick_ref(
                hyp,
                refs,
                compressor=args.compressor,
                gzip_level=args.gzip_level,
                lzma_preset=args.lzma_preset,
                strategy=args.ref_strategy,
            )
            if pick is None:
                n_missing_ref += 1
                ref_text = ""
                c_ref = ""
                c_pair = ""
                ncd = ""
                num_refs = 0
            else:
                ref_text = pick.ref_text
                num_refs = len(refs)
                hyp_b = _utf8_bytes(hyp)
                ref_b = _utf8_bytes(ref_text)
                c_h = _compress_len(hyp_b, compressor=args.compressor, gzip_level=args.gzip_level, lzma_preset=args.lzma_preset)
                c_ref = pick.c_ref
                c_pair = pick.c_pair
                ncd_val = pick.ncd
                if ncd_val is not None:
                    n_defined += 1
                    ncd = f"{ncd_val:.6f}"
                else:
                    ncd = ""

            hyp_b = _utf8_bytes(hyp)
            ref_b = _utf8_bytes(ref_text)
            c_h = _compress_len(hyp_b, compressor=args.compressor, gzip_level=args.gzip_level, lzma_preset=args.lzma_preset)
            c_ref_int = int(c_ref) if isinstance(c_ref, int) else (int(c_ref) if str(c_ref).isdigit() else 0)
            c_pair_int = int(c_pair) if isinstance(c_pair, int) else (int(c_pair) if str(c_pair).isdigit() else 0)

            row = {
                "sentence_id": sid,
                "genre_merged": genre,
                "len_chars_hyp": _char_len_for_binning(hyp),
                "len_bytes_hyp": len(hyp_b),
                "len_chars_ref": _char_len_for_binning(ref_text),
                "len_bytes_ref": len(ref_b),
                "compressor": args.compressor,
                "c_hyp": c_h,
                "c_ref": c_ref_int if pick is not None else "",
                "c_pair": c_pair_int if pick is not None else "",
                "ncd": ncd,
                "hyp": hyp,
                "ref": ref_text,
                "ref_strategy": args.ref_strategy,
                "num_refs": num_refs,
            }
            w.writerow(row)

    print(f"[OK] Wrote: {out_csv}")
    print(f"[OK] Rows: {n_rows}, NCD_defined: {n_defined}, missing_ref: {n_missing_ref}")


if __name__ == "__main__":
    main()

