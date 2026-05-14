import asyncio
import csv
import glob
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set

import hydra
from omegaconf import DictConfig, OmegaConf

from src.core.resources import ResourceManager
from src.models.remote_model import RemoteLLMClient
from src.utils.helpers import read_jsonl, save_json
from src.utils.logger import setup_logger

# Silence noisy per-request transport logs during evaluation.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = setup_logger(__name__)

KS = [1, 5, 10, 20]
LLM_ACC_PROMPT_TEMPLATE = """
You are a professional evaluator. Your task is to determine whether the predicted answer is correct based on the question and the actual answer.
The evaluation criteria should be reasonable and not overly strict. As long as the meaning is correct, or if the relevant statement has been mentioned, it is acceptable.

Question: {question}
Gold Answer: {gold_answer}
Predicted Answer: {answer}

Return only "1" (correct) or "0" (incorrect):
""".strip()

def recall_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    if not ground_truth_ids:
        return 0.0
    retrieved_k = set(retrieved_ids[:k])
    return len(retrieved_k.intersection(ground_truth_ids)) / len(ground_truth_ids)

def precision_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    retrieved_k = retrieved_ids[:k]
    if not retrieved_k:
        return 0.0
    rel_hits = sum(1 for doc_id in retrieved_k if doc_id in ground_truth_ids)
    return rel_hits / len(retrieved_k)

def f1_from_precision_recall(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def r_precision_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    """
    Truncated R-Precision at k:
    precision over the first min(R, k) results, where R = number of relevant docs.
    This keeps the metric comparable across fixed cutoffs and @all.
    """
    R = len(ground_truth_ids)
    if R == 0:
        return 0.0
    cutoff = min(R, k)
    if cutoff <= 0:
        return 0.0
    retrieved_k = retrieved_ids[:cutoff]
    if not retrieved_k:
        return 0.0
    rel_hits = sum(1 for doc_id in retrieved_k if doc_id in ground_truth_ids)
    return rel_hits / cutoff


def resolve_dataset_file(dataset_dir: str, preferred_name: str, fallback_glob: Optional[str] = None) -> str:
    preferred_path = os.path.join(dataset_dir, preferred_name)
    if os.path.exists(preferred_path):
        return preferred_path

    if fallback_glob:
        matches = sorted(glob.glob(os.path.join(dataset_dir, fallback_glob)))
        if matches:
            return matches[0]

    return preferred_path

def load_relevance(path: str) -> Dict[str, Set[str]]:
    """Load relevance judgments: query_id -> set of relevant doc_ids"""
    relevance = {}
    data = read_jsonl(path)
    for item in data:
        qid = item.get("query-id")
        docid = item.get("corpus-id")
        if qid is not None and docid is not None:
            qid = str(qid)
            if qid not in relevance:
                relevance[qid] = set()
            relevance[qid].add(str(docid))
    return relevance

def load_queries(path: str) -> Dict[str, str]:
    """Load queries: query_id -> query_text"""
    queries = {}
    data = read_jsonl(path)
    for item in data:
        qid = item.get("id")
        text = item.get("text") or item.get("query") or item.get("question") or ""
        if qid is not None:
            queries[str(qid)] = text
    return queries


def load_gold_answers(path: str) -> Dict[str, str]:
    """Load gold final answers: query_id -> gold_answer"""
    gold_answers = {}
    data = read_jsonl(path)
    for item in data:
        qid = item.get("id")
        answer = item.get("final_answer") or item.get("answer") or ""
        if qid is not None:
            gold_answers[str(qid)] = answer
    return gold_answers

def load_results(path: str) -> Dict[str, List[str]]:
    """Load system results: query_id -> list of retrieved doc_ids"""
    results = {}
    if not os.path.exists(path):
        print(f"Warning: Results file not found at {path}")
        return results
        
    data = read_jsonl(path)
    for item in data:
        qid = item.get("id")
        # Support both "doc_ids" (list) or infer from other fields if needed
        doc_ids = item.get("doc_ids", [])
        if qid is not None:
            # Ensure doc_ids are strings
            results[str(qid)] = [str(d) for d in doc_ids]
    return results


def load_result_items(path: str) -> Dict[str, Dict[str, Any]]:
    """Load raw result rows keyed by query id."""
    results = {}
    if not os.path.exists(path):
        return results

    for item in read_jsonl(path):
        qid = item.get("id")
        if qid is not None:
            results[str(qid)] = item
    return results


def parse_binary_judgment(raw_output: str) -> Optional[int]:
    """Parse the judge model output into 0/1."""
    text = (raw_output or "").strip()
    if text in {"0", "1"}:
        return int(text)

    match = re.search(r"\b([01])\b", text)
    if match:
        return int(match.group(1))

    collapsed = "".join(ch for ch in text if ch in {"0", "1"})
    if len(collapsed) == 1:
        return int(collapsed)

    return None


async def evaluate_llm_acc(
    cfg: DictConfig,
    dataset_dir: str,
    results_path: str,
    output_dir: str,
) -> Optional[Dict[str, Any]]:
    queries_path = resolve_dataset_file(dataset_dir, "queries.jsonl", "queries*.jsonl")
    gold_answers_path = resolve_dataset_file(dataset_dir, "answers.jsonl", "final_answer*.jsonl")

    if not os.path.exists(gold_answers_path):
        logger.warning("Gold answers file not found at %s; skipping llm-acc evaluation.", gold_answers_path)
        return None

    result_items = load_result_items(results_path)
    if not result_items:
        logger.warning("No final answers found in %s; skipping llm-acc evaluation.", results_path)
        return None

    query_map = load_queries(queries_path) if os.path.exists(queries_path) else {}
    gold_map = load_gold_answers(gold_answers_path)

    eval_qids = sorted(qid for qid in gold_map.keys() if qid in result_items)
    missing_predictions = sorted(qid for qid in gold_map.keys() if qid not in result_items)

    if not eval_qids:
        logger.warning("No overlapping ids between gold answers and predictions; skipping llm-acc evaluation.")
        return None

    ResourceManager.setup(OmegaConf.to_container(cfg.resources, resolve=True))
    judge_client = RemoteLLMClient(
        api_key=cfg.model.api_key,
        base_url=cfg.model.base_url,
        model=cfg.model.llm_name,
        api_mode=cfg.model.get("api_mode", "auto"),
    )

    llm_judge_concurrency = 8
    if "evaluation" in cfg and cfg.evaluation is not None:
        llm_judge_concurrency = int(cfg.evaluation.get("llm_judge_concurrency", llm_judge_concurrency))
    elif "search" in cfg and cfg.search is not None:
        llm_judge_concurrency = int(cfg.search.get("query_concurrency", llm_judge_concurrency))

    sem = asyncio.Semaphore(max(1, llm_judge_concurrency))

    async def judge_one(qid: str) -> Dict[str, Any]:
        result_item = result_items[qid]
        question = str(query_map.get(qid) or result_item.get("query") or result_item.get("text") or "")
        gold_answer = str(gold_map[qid])
        predicted_answer = str(
            result_item.get("answer")
            or result_item.get("final_answer")
            or result_item.get("result")
            or ""
        )
        prompt = LLM_ACC_PROMPT_TEMPLATE.format(
            question=question,
            gold_answer=gold_answer,
            answer=predicted_answer,
        )

        async with sem:
            raw_output = await judge_client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=8,
            )

        label = parse_binary_judgment(raw_output)
        return {
            "query_id": qid,
            "query_text": question,
            "gold_answer": gold_answer,
            "predicted_answer": predicted_answer,
            "judge_raw_output": raw_output,
            "llm_label": label,
        }

    rows = await asyncio.gather(*(judge_one(qid) for qid in eval_qids))

    valid_rows = [row for row in rows if row["llm_label"] in {0, 1}]
    invalid_rows = [row for row in rows if row["llm_label"] not in {0, 1}]
    llm_acc = (
        sum(int(row["llm_label"]) for row in valid_rows) / len(valid_rows)
        if valid_rows
        else 0.0
    )

    details_csv_path = os.path.join(output_dir, "llm_acc_details.csv")
    with open(details_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "query_id",
            "query_text",
            "llm_label",
            "judge_raw_output",
            "gold_answer",
            "predicted_answer",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "requested_gold_answers": len(gold_map),
        "evaluated_queries": len(rows),
        "valid_judgments": len(valid_rows),
        "invalid_judgments": len(invalid_rows),
        "missing_predictions": missing_predictions,
        "llm_acc": llm_acc,
        "prompt_template": LLM_ACC_PROMPT_TEMPLATE,
        "judge_model": cfg.model.llm_name,
        "judge_api_mode": cfg.model.get("api_mode", "auto"),
    }
    save_json(os.path.join(output_dir, "llm_acc_summary.json"), summary)

    return {
        "summary": summary,
        "rows": rows,
        "details_csv_path": details_csv_path,
    }

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    logger.info(f"Evaluating with configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Paths
    dataset_dir = cfg.dataset.path
    output_dir = cfg.output_dir
    
    relevance_path = os.path.join(dataset_dir, "relevance.jsonl")
    queries_path = resolve_dataset_file(dataset_dir, "queries.jsonl", "queries*.jsonl")
    results_path = os.path.join(output_dir, "final_results.jsonl")
    csv_output_path = os.path.join(output_dir, "evaluation_metrics.csv")

    logger.info(f"Loading data for dataset: {cfg.dataset.name}")
    rel_map = load_relevance(relevance_path)
    query_map = load_queries(queries_path)
    res_map = load_results(results_path)

    if not res_map:
        logger.warning("No retrieval results found for recall-style metrics.")

    logger.info(f"Found {len(rel_map)} queries in ground truth.")
    logger.info(f"Found {len(res_map)} queries in results.")

    metrics_rows = []
    totals = {
        "recall_all": 0.0,
        "r_precision_all": 0.0,
        "f1_all": 0.0,
    }
    for k in KS:
        totals[f"recall_at_{k}"] = 0.0
        totals[f"r_precision_at_{k}"] = 0.0
    count = 0

    # Evaluate intersection of queries found in both (or just results)
    # Usually we evaluate on all queries in ground truth, treating missing results as 0
    
    eval_qids = sorted(list(rel_map.keys()))
    
    for qid in eval_qids:
        if qid not in res_map:
            # If query was not processed, skip or count as 0? 
            # Usually count as 0, but here we might have partial runs.
            # Let's track it but maybe not penalize if we just ran a subset.
            continue

        gt_ids = rel_map[qid]
        retrieved_ids = res_map[qid]
        query_text = query_map.get(qid, "")

        all_k = max(len(retrieved_ids), len(gt_ids))
        recall_all = recall_at_k(retrieved_ids, gt_ids, k=all_k)
        precision_all = precision_at_k(retrieved_ids, gt_ids, k=len(retrieved_ids))
        f1_all = f1_from_precision_recall(precision_all, recall_all)
        r_precision_all = r_precision_at_k(retrieved_ids, gt_ids, k=all_k)

        row = {
            "query_id": qid,
            "query_text": query_text,
            "num_retrieved": len(retrieved_ids),
            "num_relevant": len(gt_ids),
        }

        for k in KS:
            row[f"recall_at_{k}"] = recall_at_k(retrieved_ids, gt_ids, k=k)
            row[f"r_precision_at_{k}"] = r_precision_at_k(retrieved_ids, gt_ids, k=k)

        row["recall_all"] = recall_all
        row["r_precision_all"] = r_precision_all
        row["f1_all"] = f1_all

        # Missing Docs
        retrieved_set = set(retrieved_ids)
        missing_ids = list(gt_ids - retrieved_set)
        row["missing_doc_ids"] = ";".join(missing_ids)
        metrics_rows.append(row)

        for k in KS:
            totals[f"recall_at_{k}"] += row[f"recall_at_{k}"]
            totals[f"r_precision_at_{k}"] += row[f"r_precision_at_{k}"]
        totals["recall_all"] += recall_all
        totals["r_precision_all"] += r_precision_all
        totals["f1_all"] += f1_all
        count += 1

    if count == 0:
        logger.warning("No matching queries found for retrieval metrics; skipping recall-style evaluation output.")
    else:
        avg_metrics = {name: value / count for name, value in totals.items()}

        print("-" * 40)
        print(f"Evaluation Results ({count} queries):")
        for k in KS:
            print(f"Average Recall@{k}:       {avg_metrics[f'recall_at_{k}']:.4f}")
        print(f"Average Recall@All:     {avg_metrics['recall_all']:.4f}")
        for k in KS:
            print(f"Average R-Precision@{k}: {avg_metrics[f'r_precision_at_{k}']:.4f}")
        print(f"Average R-Precision@All:{avg_metrics['r_precision_all']:.4f}")
        print(f"Average F1@All:         {avg_metrics['f1_all']:.4f}")
        print("-" * 40)
        print(f"Saving detailed metrics to {csv_output_path}")

        # Save CSV
        with open(csv_output_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "query_id",
                "query_text",
                "recall_at_1",
                "recall_at_5",
                "recall_at_10",
                "recall_at_20",
                "recall_all",
                "r_precision_at_1",
                "r_precision_at_5",
                "r_precision_at_10",
                "r_precision_at_20",
                "r_precision_all",
                "f1_all",
                "num_retrieved",
                "num_relevant",
                "missing_doc_ids",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics_rows)

    llm_acc_result = asyncio.run(
        evaluate_llm_acc(
            cfg=cfg,
            dataset_dir=dataset_dir,
            results_path=results_path,
            output_dir=output_dir,
        )
    )
    if llm_acc_result is not None:
        summary = llm_acc_result["summary"]
        print(f"Average LLM-Acc:       {summary['llm_acc']:.4f}")
        print(f"Valid LLM Judgments:   {summary['valid_judgments']}/{summary['evaluated_queries']}")
        print(f"LLM-Acc details saved to {llm_acc_result['details_csv_path']}")
        print(f"LLM-Acc summary saved to {os.path.join(output_dir, 'llm_acc_summary.json')}")

if __name__ == "__main__":
    main()
