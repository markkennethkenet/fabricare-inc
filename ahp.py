"""Analytic Hierarchy Process (Saaty, 1980) for the Fabric Health Score weights.

Experts compare the four damage types two at a time on Saaty's 1-9 scale: which one leaves a garment in worse
condition, and by how much. Each expert's answers form a pairwise comparison matrix A, where a_ij says how much
worse damage i is than damage j and a_ji = 1 / a_ij. The weights are the principal eigenvector of A, scaled to sum
to 1. The consistency ratio CR = ((lambda_max - n) / (n - 1)) / RI checks that the answers agree with each other;
Saaty accepts CR <= 0.10. The group weights come from the element-wise geometric mean of the accepted experts'
matrices (aggregation of individual judgments, Aczel and Saaty, 1983).

Usage:
    python ahp.py responses.csv                  # print each expert's weights and CR, and the group result
    python ahp.py responses.csv --write-config   # also store the group weights in assessment_config.json

responses.csv has one row per expert: an "expert" column and one column per pair, e.g. "hole_vs_stain".
A value of 3 means the first damage is moderately worse; 1/3 means the second one is; 1 means equal.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

CRITERIA = ("hole", "stain", "pilling", "needlemark")
PAIRS = tuple((first, second) for i, first in enumerate(CRITERIA) for second in CRITERIA[i + 1:])
# Saaty's random consistency index for a matrix of n criteria.
RANDOM_INDEX = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}
MAX_CONSISTENCY_RATIO = 0.10
CONFIG_PATH = Path(__file__).parent / "assessment_config.json"


def column(pair):
    return f"{pair[0]}_vs_{pair[1]}"


def parse_judgment(text):
    """'3' -> 3.0 (the first damage is worse), '1/3' -> 0.333 (the second one is), on Saaty's 1/9 to 9 scale."""
    try:
        value = float(Fraction(str(text).strip()))
    except (ValueError, ZeroDivisionError):
        raise ValueError(f"Not a Saaty scale value: {text!r}") from None
    if not 1 / 9 - 1e-9 <= value <= 9 + 1e-9:
        raise ValueError(f"Saaty scale values run from 1/9 to 9, not {text!r}")
    return value


def comparison_matrix(judgments):
    """The reciprocal matrix from {pair: value}: a_ij = value, a_ji = 1 / value, a_ii = 1."""
    index = {name: i for i, name in enumerate(CRITERIA)}
    matrix = [[1.0] * len(CRITERIA) for _ in CRITERIA]
    for (first, second), value in judgments.items():
        matrix[index[first]][index[second]] = value
        matrix[index[second]][index[first]] = 1 / value
    return matrix


def priorities(matrix, tolerance=1e-12, max_iterations=10_000):
    """Principal eigenvector by power iteration, scaled to sum to 1, and its eigenvalue lambda_max."""
    n = len(matrix)
    weights = [1 / n] * n
    for _ in range(max_iterations):
        product = [sum(matrix[i][j] * weights[j] for j in range(n)) for i in range(n)]
        total = sum(product)
        updated = [value / total for value in product]
        done = max(abs(a - b) for a, b in zip(updated, weights)) < tolerance
        weights = updated
        if done:
            break
    lambda_max = sum(sum(matrix[i][j] * weights[j] for j in range(n)) / weights[i] for i in range(n)) / n
    return weights, lambda_max


def consistency_ratio(matrix):
    n = len(matrix)
    _, lambda_max = priorities(matrix)
    index = (lambda_max - n) / (n - 1)
    return index / RANDOM_INDEX[n] if RANDOM_INDEX[n] else 0.0


def group_matrix(matrices):
    """Element-wise geometric mean of the experts' matrices; it stays reciprocal."""
    count = len(matrices)
    size = len(matrices[0])
    return [[math.prod(m[i][j] for m in matrices) ** (1 / count) for j in range(size)] for i in range(size)]


def read_responses(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("The responses file has no rows.")
    missing = [column(pair) for pair in PAIRS if column(pair) not in rows[0]]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    return [(row.get("expert") or f"Expert {number}",
             {pair: parse_judgment(row[column(pair)]) for pair in PAIRS})
            for number, row in enumerate(rows, start=1)]


def analyse(responses):
    """Per-expert weights and CR, and the group weights from the experts whose CR is acceptable."""
    experts = []
    for name, judgments in responses:
        matrix = comparison_matrix(judgments)
        weights, _ = priorities(matrix)
        ratio = consistency_ratio(matrix)
        experts.append({"expert": name, "matrix": matrix, "weights": dict(zip(CRITERIA, weights)),
                        "consistency_ratio": ratio, "accepted": ratio <= MAX_CONSISTENCY_RATIO})
    accepted = [expert["matrix"] for expert in experts if expert["accepted"]]
    if not accepted:
        raise ValueError("No expert has a consistency ratio of 0.10 or less; ask them to review their answers.")
    combined = group_matrix(accepted)
    weights, _ = priorities(combined)
    return {"experts": experts, "group_weights": dict(zip(CRITERIA, weights)),
            "group_consistency_ratio": consistency_ratio(combined), "accepted_count": len(accepted)}


def write_config(result, path=CONFIG_PATH):
    config = json.loads(path.read_text(encoding="utf-8"))
    config["health_weights"] = {
        "source": (f"Expert AHP survey: {result['accepted_count']} expert(s) with CR <= 0.10, "
                   f"group CR {result['group_consistency_ratio']:.3f}"),
        "weights": {name: round(value, 4) for name, value in result["group_weights"].items()},
    }
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def report(result):
    header = f"{'Expert':<24}" + "".join(f"{name:>12}" for name in CRITERIA) + f"{'CR':>8}  Used"
    lines = [header, "-" * len(header)]
    for expert in result["experts"]:
        lines.append(f"{expert['expert'][:23]:<24}"
                     + "".join(f"{expert['weights'][name]:>12.3f}" for name in CRITERIA)
                     + f"{expert['consistency_ratio']:>8.3f}  {'yes' if expert['accepted'] else 'no (CR > 0.10)'}")
    lines.append("-" * len(header))
    lines.append(f"{'Group':<24}" + "".join(f"{result['group_weights'][name]:>12.3f}" for name in CRITERIA)
                 + f"{result['group_consistency_ratio']:>8.3f}")
    lines.append("")
    lines.append("Points each damage type removes from the Fabric Health Score (100 x weight):")
    lines.extend(f"  {name:<11} {100 * result['group_weights'][name]:5.1f}" for name in CRITERIA)
    return "\n".join(lines)


def main(argv):
    if not argv or argv[0] in {"-h", "--help"}:
        print(__doc__)
        return 0 if argv else 2
    result = analyse(read_responses(argv[0]))
    print(report(result))
    if "--write-config" in argv[1:]:
        write_config(result)
        print(f"\nSaved the group weights to {CONFIG_PATH.name}. Restart the API server to use them.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
