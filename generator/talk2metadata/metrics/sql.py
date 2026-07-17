"""
SQL Evaluation Metrics Module.

This module implements comprehensive metrics for evaluating SQL generation performance,
including Exact Match, Execution Accuracy, component-wise matching, and efficiency scores.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set


@dataclass
class SQLMetricResult:
    """Result of a single SQL metric calculation."""

    metric_name: str
    score: float
    details: Dict[str, Any] = field(default_factory=dict)


class SQLEvaluator:
    """Comprehensive SQL Evaluator implementing various metrics."""

    def __init__(self, execute_sql_func: Optional[Callable[[str], Any]] = None):
        self.execute_sql_func = execute_sql_func

    def normalize_sql(self, sql: str) -> str:
        if not sql:
            return ""
        sql = sql.strip().lower()
        if sql.endswith(";"):
            sql = sql[:-1]
        sql = re.sub(r"\s+", " ", sql).strip()

        match = re.search(r" where (.*?)(?: group by | order by |$)", sql)
        if match:
            where_content = match.group(1)
            conditions = re.split(r"\s+and\s+", where_content)
            sorted_conditions = sorted([c.strip() for c in conditions])
            new_where_content = " and ".join(sorted_conditions)
            start, end = match.span(1)
            sql = sql[:start] + new_where_content + sql[end:]

        return sql

    def exact_match(self, predicted_sql: str, reference_sql: str) -> SQLMetricResult:
        norm_pred = self.normalize_sql(predicted_sql)
        norm_ref = self.normalize_sql(reference_sql)
        match = norm_pred == norm_ref

        return SQLMetricResult(
            metric_name="Exact Match",
            score=1.0 if match else 0.0,
            details={
                "normalized_predicted": norm_pred,
                "normalized_reference": norm_ref,
            },
        )

    def execution_accuracy(
        self, predicted_sql: str, reference_sql: str
    ) -> SQLMetricResult:
        if not self.execute_sql_func:
            return SQLMetricResult(
                metric_name="Execution Accuracy",
                score=0.0,
                details={"error": "No execution function provided"},
            )

        try:
            pred_result = self.execute_sql_func(predicted_sql)
        except Exception as e:
            return SQLMetricResult(
                metric_name="Execution Accuracy",
                score=0.0,
                details={"error": f"Predicted SQL execution failed: {str(e)}"},
            )

        try:
            ref_result = self.execute_sql_func(reference_sql)
        except Exception as e:
            return SQLMetricResult(
                metric_name="Execution Accuracy",
                score=0.0,
                details={"error": f"Reference SQL execution failed: {str(e)}"},
            )

        def normalize_row(row: Any) -> tuple:
            if isinstance(row, dict):
                return tuple(sorted((k, str(v)) for k, v in row.items()))
            if isinstance(row, (list, tuple)):
                return tuple(str(v) for v in row)
            return (str(row),)

        try:
            pred_set = set(normalize_row(r) for r in pred_result)
            ref_set = set(normalize_row(r) for r in ref_result)
        except Exception as e:
            return SQLMetricResult(
                metric_name="Execution Accuracy",
                score=0.0,
                details={"error": f"Result normalization failed: {str(e)}"},
            )

        intersection = pred_set.intersection(ref_set)
        precision = len(intersection) / len(pred_set) if pred_set else 0.0
        recall = len(intersection) / len(ref_set) if ref_set else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        is_correct = f1 == 1.0

        return SQLMetricResult(
            metric_name="Execution Accuracy",
            score=1.0 if is_correct else 0.0,
            details={
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "predicted_result_count": len(pred_result),
                "reference_result_count": len(ref_result),
                "intersection_count": len(intersection),
            },
        )

    def valid_efficiency_score(
        self, execution_accuracy: float, execution_time_ms: float
    ) -> SQLMetricResult:
        w = 2410.01066
        if execution_accuracy == 0:
            ves = 0.0
        else:
            ves = w / (w + execution_time_ms)

        return SQLMetricResult(
            metric_name="Valid Efficiency Score",
            score=ves,
            details={"execution_time_ms": execution_time_ms, "weight": w},
        )

    def _extract_components(self, sql: str) -> Dict[str, List[str]]:
        sql = self.normalize_sql(sql)
        components = {
            "SELECT": [],
            "FROM": [],
            "WHERE": [],
            "GROUP BY": [],
            "ORDER BY": [],
        }

        patterns = {
            "SELECT": r"select\s+(.*?)\s+(from|$)",
            "FROM": r"from\s+(.*?)\s+(where|group by|order by|$)",
            "WHERE": r"where\s+(.*?)\s+(group by|order by|$)",
            "GROUP BY": r"group by\s+(.*?)\s+(order by|$)",
            "ORDER BY": r"order by\s+(.*?)$",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, sql)
            if not match:
                continue
            content = match.group(1).strip()
            if key in {"SELECT", "GROUP BY", "ORDER BY"}:
                components[key] = [x.strip() for x in content.split(",") if x.strip()]
            elif key == "FROM":
                items = re.split(r"\s+(?:join|,)\s+", content)
                components[key] = [x.strip() for x in items if x.strip()]
            elif key == "WHERE":
                items = re.split(r"\s+(?:and|or)\s+", content)
                components[key] = [x.strip() for x in items if x.strip()]

        return components

    def component_matching(
        self, predicted_sql: str, reference_sql: str
    ) -> SQLMetricResult:
        pred_comps = self._extract_components(predicted_sql)
        ref_comps = self._extract_components(reference_sql)

        f1_scores: List[float] = []
        details: Dict[str, float] = {}

        for key in pred_comps:
            pred_set = set(pred_comps[key])
            ref_set = set(ref_comps[key])

            if not pred_set and not ref_set:
                f1 = 1.0
            else:
                intersection = len(pred_set.intersection(ref_set))
                precision = intersection / len(pred_set) if pred_set else 0.0
                recall = intersection / len(ref_set) if ref_set else 0.0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )

            f1_scores.append(f1)
            details[key] = f1

        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

        return SQLMetricResult(
            metric_name="Component Matching F1",
            score=avg_f1,
            details=details,
        )

    def _tokenize(self, sql: str) -> Set[str]:
        sql = self.normalize_sql(sql)
        tokens = re.split(r"[^a-z0-9_]+", sql)
        return set(filter(None, tokens))

    def soft_f1(self, predicted_sql: str, reference_sql: str) -> SQLMetricResult:
        pred_tokens = self._tokenize(predicted_sql)
        ref_tokens = self._tokenize(reference_sql)

        if not pred_tokens and not ref_tokens:
            return SQLMetricResult("Soft F1", 1.0)

        intersection = len(pred_tokens.intersection(ref_tokens))
        precision = intersection / len(pred_tokens) if pred_tokens else 0.0
        recall = intersection / len(ref_tokens) if ref_tokens else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return SQLMetricResult(
            metric_name="Soft F1",
            score=f1,
            details={"precision": precision, "recall": recall},
        )

    def _simple_tree_edit_distance(self, t1: Dict, t2: Dict) -> float:
        distance = 0
        keys = set(t1.keys()) | set(t2.keys())
        for key in keys:
            s1 = set(t1.get(key, []))
            s2 = set(t2.get(key, []))
            distance += len(s1.symmetric_difference(s2))
        return float(distance)

    def static_metrics(self, predicted_sql: str, reference_sql: str) -> SQLMetricResult:
        pred_comps = self._extract_components(predicted_sql)
        ref_comps = self._extract_components(reference_sql)

        tsed = self._simple_tree_edit_distance(pred_comps, ref_comps)

        pred_keys = set(k for k, v in pred_comps.items() if v)
        ref_keys = set(k for k, v in ref_comps.items() if v)

        if not pred_keys and not ref_keys:
            sqam = 1.0
        else:
            intersection = len(pred_keys.intersection(ref_keys))
            union = len(pred_keys.union(ref_keys))
            sqam = intersection / union if union > 0 else 0.0

        return SQLMetricResult(
            metric_name="Static Metrics",
            score=0.0,
            details={"TSED": tsed, "SQAM": sqam},
        )

    def evaluate_all(
        self,
        predicted_sql: str,
        reference_sql: str,
        execution_time_ms: float = 0.0,
        enabled_metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        if enabled_metrics is None:
            enabled_metrics = [
                "exact_match",
                "execution_accuracy",
                "valid_efficiency_score",
                "component_matching",
                "soft_f1",
                "static_metrics",
            ]

        if "exact_match" in enabled_metrics:
            em = self.exact_match(predicted_sql, reference_sql)
            results["exact_match"] = em.score

        ex_score = 0.0
        if "execution_accuracy" in enabled_metrics:
            ex = self.execution_accuracy(predicted_sql, reference_sql)
            results["execution_accuracy"] = ex.score
            ex_score = ex.score

        if "valid_efficiency_score" in enabled_metrics:
            ves = self.valid_efficiency_score(ex_score, execution_time_ms)
            results["valid_efficiency_score"] = ves.score

        if "component_matching" in enabled_metrics:
            cm = self.component_matching(predicted_sql, reference_sql)
            results["component_matching_f1"] = cm.score
            results["component_matching_details"] = cm.details

        if "soft_f1" in enabled_metrics:
            sf1 = self.soft_f1(predicted_sql, reference_sql)
            results["soft_f1"] = sf1.score

        if "static_metrics" in enabled_metrics:
            static = self.static_metrics(predicted_sql, reference_sql)
            results["tsed"] = static.details["TSED"]
            results["sqam"] = static.details["SQAM"]

        return results
