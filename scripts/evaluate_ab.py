"""
Evaluation script for Task A (Grammar) and Task B (Appropriateness) of
the Chinese Appropriateness Benchmark.

This script evaluates model predictions against human annotations using
Quadratic Weighted Kappa (QWK) metric.

Usage:
    python scripts/evaluate_ab.py \
        --pred-csv results/task_ab_model_predictions/model_taskab_prediction_zeroshot.csv \
        --gold-file data/c_rans_release.json \
        --test-ids splits/test_ids.txt \
        --model-name GPT5


"""

import argparse
import json
import os
import sys
from typing import List, Tuple, Set
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------
# QWK Implementation
# ---------------------------------------------------------------

def quadratic_weighted_kappa(
    y_true: List[int], y_pred: List[int], min_rating: int = 1, max_rating: int = 5
) -> float:
    """计算Quadratic Weighted Kappa (QWK)指标。
    
    Args:
        y_true: 真实标签列表（1-5的整数）
        y_pred: 预测标签列表（1-5的整数）
        min_rating: 最小评分值，默认1
        max_rating: 最大评分值，默认5
    
    Returns:
        QWK分数，范围通常在-1到1之间，1表示完全一致
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    assert y_true.shape == y_pred.shape

    n_ratings = max_rating - min_rating + 1

    # 混淆矩阵 O
    O = np.zeros((n_ratings, n_ratings), dtype=float)
    for t, p in zip(y_true, y_pred):
        if min_rating <= t <= max_rating and min_rating <= p <= max_rating:
            O[t - min_rating, p - min_rating] += 1.0

    # 行列边际分布
    act_hist = np.bincount(y_true - min_rating, minlength=n_ratings).astype(float)
    pred_hist = np.bincount(y_pred - min_rating, minlength=n_ratings).astype(float)

    # 期望矩阵 E
    E = np.outer(act_hist, pred_hist)
    if E.sum() == 0:
        return 0.0
    E = E / E.sum() * O.sum()

    # 权重矩阵 W
    W = np.zeros((n_ratings, n_ratings), dtype=float)
    for i in range(n_ratings):
        for j in range(n_ratings):
            W[i, j] = ((i - j) ** 2) / ((n_ratings - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum()
    if den == 0:
        return 0.0
    return 1.0 - num / den


# ---------------------------------------------------------------
# Data Loading Functions
# ---------------------------------------------------------------

GOLD_RATING_COLS = ["grammar_rating", "naturalness_rating"]
LEGACY_GOLD_RATING_COLS = {
    "grammar_score": "grammar_rating",
    "appropriateness_score": "naturalness_rating",
}
PRED_RATING_COLS = ["task_a_score", "task_b_score"]
RELEASE_PRED_RATING_COLS = {
    "Task A": "task_a_score",
    "Task B": "task_b_score",
}


def load_test_ids(test_ids_path: str) -> Set[str]:
    """从文件加载测试集sentence_id列表。
    
    Args:
        test_ids_path: 测试集ID文件路径，每行一个sentence_id
    
    Returns:
        sentence_id的集合
    """
    if not os.path.exists(test_ids_path):
        raise FileNotFoundError(f"Test IDs file not found: {test_ids_path}")
    
    with open(test_ids_path, "r", encoding="utf-8") as f:
        test_ids = {line.strip() for line in f if line.strip()}
    
    return test_ids


def _read_gold_file(gold_path: str) -> pd.DataFrame:
    """Read release gold annotations from JSON or CSV."""
    suffix = Path(gold_path).suffix.lower()
    if suffix == ".json":
        with open(gold_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Gold JSON must contain a list of objects: {gold_path}")
        return pd.DataFrame(data)
    return pd.read_csv(gold_path)


def load_gold_data(gold_path: str, test_ids_path: str) -> pd.DataFrame:
    """加载人类标注数据并过滤测试集。
    
    Args:
        gold_path: 标注数据JSON或CSV文件路径
        test_ids_path: 测试集ID文件路径
    
    Returns:
        包含sentence_id, sentence, grammar_rating, naturalness_rating的DataFrame
    """
    if not os.path.exists(gold_path):
        raise FileNotFoundError(f"Gold file not found: {gold_path}")
    
    # 加载测试集ID
    test_ids = load_test_ids(test_ids_path)
    
    # 读取发布数据文件。优先使用release字段名；旧字段名仅做兼容转换。
    df = _read_gold_file(gold_path)
    df = df.rename(columns={k: v for k, v in LEGACY_GOLD_RATING_COLS.items() if k in df.columns})
    
    # 检查必需的列
    required_cols = ["sentence_id", "sentence", *GOLD_RATING_COLS]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Gold file missing required columns: {missing_cols}")
    
    # 过滤测试集
    df_test = df[df["sentence_id"].isin(test_ids)].copy()
    
    # 只保留需要的列
    df_test = df_test[required_cols].copy()
    
    # 验证分数有效性
    for col in GOLD_RATING_COLS:
        # 尝试转换为整数
        df_test[col] = pd.to_numeric(df_test[col], errors="coerce")
        # 过滤掉无效值（NaN或不在1-5范围内）
        invalid_mask = (
            df_test[col].isna() | 
            (df_test[col] < 1) | 
            (df_test[col] > 5)
        )
        if invalid_mask.any():
            print(f"Warning: Found {invalid_mask.sum()} invalid {col} values, filtering them out.")
            df_test = df_test[~invalid_mask].copy()
    
    # 转换为整数
    df_test["grammar_rating"] = df_test["grammar_rating"].astype(int)
    df_test["naturalness_rating"] = df_test["naturalness_rating"].astype(int)
    
    return df_test.reset_index(drop=True)


def load_pred_data(csv_path: str, model_name: str = None) -> pd.DataFrame:
    """加载模型预测数据。
    
    Args:
        csv_path: 模型预测CSV文件路径
        model_name: 可选模型名；用于过滤包含多个模型的release预测文件
    
    Returns:
        包含sentence_id, task_a_score, task_b_score的DataFrame
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Prediction CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    df = df.rename(columns={k: v for k, v in RELEASE_PRED_RATING_COLS.items() if k in df.columns})
    
    if model_name is not None:
        if "model_name" not in df.columns:
            raise ValueError("--model-name was provided, but prediction CSV has no 'model_name' column.")
        df = df[df["model_name"].astype(str) == model_name].copy()
        if len(df) == 0:
            raise ValueError(f"No prediction rows found for model_name={model_name!r}.")
    
    # 检查必需的列
    required_cols = ["sentence_id", *PRED_RATING_COLS]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Prediction CSV missing required columns: {missing_cols}")
    
    # 只保留需要的列
    df = df[required_cols].copy()
    
    # 验证分数有效性
    for col in ["task_a_score", "task_b_score"]:
        # 尝试转换为整数
        df[col] = pd.to_numeric(df[col], errors="coerce")
        # 过滤掉无效值（NaN或不在1-5范围内）
        invalid_mask = (
            df[col].isna() | 
            (df[col] < 1) | 
            (df[col] > 5)
        )
        if invalid_mask.any():
            print(f"Warning: Found {invalid_mask.sum()} invalid {col} values, filtering them out.")
            df = df[~invalid_mask].copy()
    
    # 转换为整数
    df["task_a_score"] = df["task_a_score"].astype(int)
    df["task_b_score"] = df["task_b_score"].astype(int)
    
    return df.reset_index(drop=True)


# ---------------------------------------------------------------
# Model Name Extraction
# ---------------------------------------------------------------

def extract_model_name(pred_csv_path: str) -> str:
    """从预测文件路径中提取模型名称。
    
    路径格式：outputs/experiment_ex02/{timestamp}/predictions/{model_name}/submission_ab.csv
    
    Args:
        pred_csv_path: 预测CSV文件的路径
    
    Returns:
        模型名称（例如：openai_gpt-5）
    """
    path = Path(pred_csv_path)
    
    # 查找predictions目录
    parts = path.parts
    try:
        predictions_idx = parts.index("predictions")
        if predictions_idx + 1 < len(parts):
            model_name = parts[predictions_idx + 1]
            return model_name
    except ValueError:
        pass
    
    # 如果找不到predictions/{model_name}/...结构，就使用文件名。
    return path.stem or "unknown_model"


# ---------------------------------------------------------------
# Data Alignment
# ---------------------------------------------------------------

def align_data(
    gold_df: pd.DataFrame, pred_df: pd.DataFrame
) -> Tuple[pd.DataFrame, List[int], List[int], List[int], List[int]]:
    """对齐gold和pred数据，返回对齐后的DataFrame和分数列表。
    
    Args:
        gold_df: 包含sentence_id, sentence, grammar_rating, naturalness_rating的DataFrame
        pred_df: 包含sentence_id, task_a_score, task_b_score的DataFrame
    
    Returns:
        Tuple包含：
        - 对齐后的DataFrame（包含sentence_id, sentence, gold和pred分数）
        - gold_grammar_scores: 人类标注的语法分数列表
        - pred_grammar_scores: 模型预测的语法分数列表
        - gold_appropriateness_scores: 人类标注的得体性分数列表
        - pred_appropriateness_scores: 模型预测的得体性分数列表
    """
    # 通过sentence_id合并
    merged = pd.merge(
        gold_df,
        pred_df,
        on="sentence_id",
        how="inner",
        suffixes=("_gold", "_pred"),
    )
    
    if len(merged) == 0:
        raise ValueError(
            "No matching sentence_ids found between gold and prediction data. "
            "Please check that the sentence_ids match."
        )
    
    # 重命名列以便清晰
    result_df = pd.DataFrame({
        "sentence_id": merged["sentence_id"],
        "sentence": merged["sentence"],
        "gold_grammar_rating": merged["grammar_rating"],
        "pred_grammar_rating": merged["task_a_score"],
        "gold_naturalness_rating": merged["naturalness_rating"],
        "pred_naturalness_rating": merged["task_b_score"],
    })
    
    # 提取分数列表
    gold_grammar_scores = result_df["gold_grammar_rating"].tolist()
    pred_grammar_scores = result_df["pred_grammar_rating"].tolist()
    gold_appropriateness_scores = result_df["gold_naturalness_rating"].tolist()
    pred_appropriateness_scores = result_df["pred_naturalness_rating"].tolist()
    
    return (
        result_df,
        gold_grammar_scores,
        pred_grammar_scores,
        gold_appropriateness_scores,
        pred_appropriateness_scores,
    )


# ---------------------------------------------------------------
# Result Saving
# ---------------------------------------------------------------

def save_detailed_results(df_aligned: pd.DataFrame, output_path: str) -> None:
    """保存详细评估结果到CSV文件。
    
    Args:
        df_aligned: 对齐后的DataFrame，包含sentence_id, sentence和所有分数
        output_path: 输出文件路径
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 保存CSV
    df_aligned.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Detailed results saved to: {output_path}")


def save_evaluation_summary(
    output_dir: str,
    model_name: str,
    valid_samples: int,
    task_a_qwk: float,
    task_b_qwk: float,
) -> None:
    """保存当次评估的统计摘要（含有效样本数），便于汇总多模型结果。
    
    Args:
        output_dir: 输出目录
        model_name: 模型名称
        valid_samples: 有效样本数
        task_a_qwk: Task A QWK
        task_b_qwk: Task B QWK
    """
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "evaluation_summary.csv")
    summary_df = pd.DataFrame(
        [
            {
                "model": model_name,
                "valid_samples": valid_samples,
                "task_a_qwk": round(task_a_qwk, 4),
                "task_b_qwk": round(task_b_qwk, 4),
            }
        ]
    )
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"Summary (with valid samples) saved to: {summary_path}")


# ---------------------------------------------------------------
# Main Function
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model predictions for Task A (Grammar) and Task B (Appropriateness) using QWK metric."
    )
    
    parser.add_argument(
        "--pred-csv",
        type=str,
        required=True,
        help="Path to model prediction CSV file (must contain sentence_id, task_a_score/task_b_score or Task A/Task B)",
    )
    
    parser.add_argument(
        "--gold-file",
        "--gold-csv",
        dest="gold_file",
        type=str,
        default=None,
        help="Path to gold annotation JSON/CSV file (default: data/c_rans_release.json)",
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Optional model_name filter for prediction CSV files that contain multiple models.",
    )
    
    parser.add_argument(
        "--test-ids",
        type=str,
        default=None,
        help="Path to test IDs file (default: splits/test_ids.txt)",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save evaluation results (default: same directory as pred-csv)",
    )
    
    parser.add_argument(
        "--save-details",
        action="store_true",
        help="Save detailed results CSV file with sentence_id, sentence, and all scores",
    )
    
    args = parser.parse_args()
    
    # 设置默认路径
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RELEASE_ROOT = os.path.dirname(BASE_DIR)
    
    if args.gold_file is None:
        args.gold_file = os.path.join(RELEASE_ROOT, "data/c_rans_release.json")
    
    if args.test_ids is None:
        # 尝试多个可能的路径
        test_ids_candidates = [
            os.path.join(RELEASE_ROOT, "splits/test_ids.txt"),
            os.path.join(RELEASE_ROOT, "experiment/splits/test_ids.txt"),
            os.path.join(RELEASE_ROOT, "benchmark/splits/test_ids.txt"),
        ]
        args.test_ids = None
        for candidate in test_ids_candidates:
            if os.path.exists(candidate):
                args.test_ids = candidate
                break
        if args.test_ids is None:
            args.test_ids = test_ids_candidates[0]  # 使用第一个作为默认值
    
    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(args.pred_csv))
    
    # 提取模型名称
    model_name = args.model_name or extract_model_name(args.pred_csv)
    
    print("=" * 80)
    print("Evaluation Configuration")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Model filter: {args.model_name or '(none)'}")
    print(f"Gold file: {args.gold_file}")
    print(f"Test IDs: {args.test_ids}")
    print(f"Prediction CSV: {args.pred_csv}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 80)
    print()
    
    # 加载数据
    print("Loading data...")
    try:
        gold_df = load_gold_data(args.gold_file, args.test_ids)
        print(f"Loaded {len(gold_df)} gold samples from test set")
    except Exception as e:
        print(f"Error loading gold data: {e}", file=sys.stderr)
        sys.exit(1)
    
    try:
        pred_df = load_pred_data(args.pred_csv, model_name=args.model_name)
        print(f"Loaded {len(pred_df)} prediction samples")
    except Exception as e:
        print(f"Error loading prediction data: {e}", file=sys.stderr)
        sys.exit(1)
    
    # 对齐数据
    print("\nAligning data...")
    try:
        (
            aligned_df,
            gold_grammar_scores,
            pred_grammar_scores,
            gold_appropriateness_scores,
            pred_appropriateness_scores,
        ) = align_data(gold_df, pred_df)
        print(f"Successfully aligned {len(aligned_df)} samples")
    except Exception as e:
        print(f"Error aligning data: {e}", file=sys.stderr)
        sys.exit(1)
    
    if len(aligned_df) == 0:
        print("Error: No samples to evaluate after alignment.", file=sys.stderr)
        sys.exit(1)
    
    # 计算QWK
    print("\nComputing QWK scores...")
    try:
        qwk_task_a = quadratic_weighted_kappa(
            gold_grammar_scores, pred_grammar_scores, min_rating=1, max_rating=5
        )
        qwk_task_b = quadratic_weighted_kappa(
            gold_appropriateness_scores,
            pred_appropriateness_scores,
            min_rating=1,
            max_rating=5,
        )
    except Exception as e:
        print(f"Error computing QWK: {e}", file=sys.stderr)
        sys.exit(1)
    
    valid_samples = len(aligned_df)

    # 打印结果（含有效样本数）
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Valid Samples: {valid_samples}")
    print(f"Task A (Grammar) QWK: {qwk_task_a:.4f}")
    print(f"Task B (Appropriateness) QWK: {qwk_task_b:.4f}")
    print("=" * 80)

    # 每次评估都保存统计摘要（含有效样本数）
    save_evaluation_summary(
        args.output_dir,
        model_name,
        valid_samples,
        qwk_task_a,
        qwk_task_b,
    )

    # 保存详细结果
    if args.save_details:
        output_path = os.path.join(
            args.output_dir, f"evaluation_details_{model_name}.csv"
        )
        save_detailed_results(aligned_df, output_path)

    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
