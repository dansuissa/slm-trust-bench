"""Explicit denominators; undefined rates remain null rather than becoming zero."""
from collections import Counter, defaultdict
import math
import re
import string


def rate(numerator, denominator):
    return numerator / denominator if denominator else None


def normalize_answer(text):
    text = text.lower()
    text = "".join(c for c in text if c not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(text, references):
    return any(normalize_answer(text) == normalize_answer(ref) for ref in references)


def entropy_score(answers):
    if not answers:
        raise ValueError("Entropy needs at least one answer")
    counts = Counter(answers)
    return sum((n / len(answers)) * math.log(n / len(answers)) for n in counts.values())


def selective_metrics(records):
    valid = [r for r in records if r["status"] == "ok" and type(r.get("correct")) is bool
             and type(r.get("accepted")) is bool]
    ac = sum(r["accepted"] and r["correct"] for r in valid)
    aw = sum(r["accepted"] and not r["correct"] for r in valid)
    rc = sum(not r["accepted"] and r["correct"] for r in valid)
    rw = sum(not r["accepted"] and not r["correct"] for r in valid)
    return {"queries": len(records), "graded_valid": len(valid), "failed_or_ungraded": len(records) - len(valid),
            "accepted_correct": ac, "accepted_incorrect": aw, "rejected_correct": rc, "rejected_incorrect": rw,
            "coverage_among_graded": rate(ac + aw, len(valid)),
            "selective_error": rate(aw, ac + aw), "accepted_accuracy": rate(ac, ac + aw),
            "false_acceptance_given_incorrect": rate(aw, aw + rw),
            "false_rejection_given_correct": rate(rc, ac + rc),
            "correct_accepted_per_query": rate(ac, len(records))}


def abstention_metrics(records):
    valid = [r for r in records if r["status"] == "ok" and type(r.get("abstained")) is bool
             and type(r.get("should_abstain")) is bool]
    tp = sum(r["abstained"] and r["should_abstain"] for r in valid)
    fp = sum(r["abstained"] and not r["should_abstain"] for r in valid)
    fn = sum(not r["abstained"] and r["should_abstain"] for r in valid)
    tn = sum(not r["abstained"] and not r["should_abstain"] for r in valid)
    return {"queries": len(records), "valid_abstention_labels": len(valid),
            "failed_or_ungraded": len(records) - len(valid), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "abstention_precision": rate(tp, tp + fp), "abstention_recall": rate(tp, tp + fn),
            "abstention_f1": rate(2 * tp, 2 * tp + fp + fn),
            "over_abstention": rate(fp, fp + tn), "unsupported_answer_rate": rate(fn, tp + fn)}


def parse_value_choice(text, options):
    matches = re.findall(r"I choose Option\s+([A-Z]|\d+)\b", text, re.IGNORECASE)
    matches = [m.upper() for m in matches]
    if len(matches) != 1 or matches[0] not in options:
        return None
    return options[matches[0]]


def repetition_metrics(records, expected_repetitions):
    groups = defaultdict(list)
    for record in records:
        groups[(record["item_id"], record.get("variant", "original"))].append(record)
    complete = []
    valid_draws = 0
    for draws in groups.values():
        if len({r["repetition"] for r in draws}) != len(draws):
            raise ValueError("Repeated draw ID")
        choices = [r.get("choice") for r in draws if r["status"] == "ok" and r.get("choice") is not None]
        valid_draws += len(choices)
        if len(draws) == expected_repetitions and len(choices) == expected_repetitions:
            complete.append(choices)
    flipped = sum(len(set(choices)) > 1 for choices in complete)
    return {"item_variants": len(groups), "complete_valid_item_variants": len(complete),
            "incomplete_or_invalid_item_variants": len(groups) - len(complete),
            "draws": len(records), "valid_choice_rate": rate(valid_draws, len(records)),
            "any_flip_rate_complete_only": rate(flipped, len(complete)),
            "expected_repetitions": expected_repetitions}


def binomial_tail(k, n, probability):
    """P[Binomial(n, probability) >= k], computed in log space."""
    if k > n:
        return 0.0
    if k <= 0:
        return 1.0
    terms = [math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1)
             + j * math.log(probability) + (n - j) * math.log1p(-probability) for j in range(k, n + 1)]
    top = max(terms)
    return min(1.0, math.exp(top) * sum(math.exp(x - top) for x in terms))


def facttest_threshold(incorrect_scores, alpha=0.05, delta=0.01):
    if not 0 < alpha < 1 or not 0 < delta < 1:
        raise ValueError("alpha and delta must lie strictly between zero and one")
    scores = sorted(incorrect_scores)
    if any(not math.isfinite(s) for s in scores):
        raise ValueError("Non-finite calibration score")
    n = len(scores)
    lo, hi = 1, n + 1
    while lo < hi:
        middle = (lo + hi) // 2
        if binomial_tail(middle, n, 1 - alpha) <= delta:
            hi = middle
        else:
            lo = middle + 1
    return {"alpha": alpha, "delta": delta, "incorrect_calibration_count": n,
            "order_statistic_one_based": lo, "threshold": scores[lo - 1] if lo <= n else None,
            "policy": "strictly_greater" if lo <= n else "reject_all"}


def accept(score, calibration):
    return calibration["policy"] != "reject_all" and score > calibration["threshold"]
