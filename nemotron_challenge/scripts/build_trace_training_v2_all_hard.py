"""Build a hard-family-heavy trace-training CSV.

The output keeps every cipher, bit-manipulation, and equation row after public
sanity IDs are excluded. Easy families are sampled as deterministic rehearsal.

Trace policy:
- cipher: strict v2 cited-map trace when every target character is supported;
  boxed-only otherwise
- bit manipulation: verified execution trace when the DSL verifier has a rule;
  boxed-only otherwise
- equation: boxed-only until a real verifier exists
- unit/gravity: compact procedural rehearsal traces
- numeral: boxed-only rehearsal
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from build_trace_training_data import (
    PUBLIC_SANITY_IDS,
    build_boxed_target,
    infer_family,
    normalize_answer,
    read_verified_bit_rules,
)
from build_trace_training_v2_2500 import (
    build_bit_execution_trace,
    build_cipher_v2_trace,
    build_gravity_compact_trace,
    build_unit_short_trace,
    select_rows,
)


HARD_FAMILIES = {"cipher", "bit_manipulation", "equation_symbolic"}
EASY_FAMILIES = {"gravity", "unit_conversion", "numeral"}


def read_train_rows(train_csv: Path) -> list[dict[str, str]]:
    with train_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            if row["id"] in PUBLIC_SANITY_IDS:
                continue
            rows.append(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "gold_answer": normalize_answer(row["gold_answer"]),
                    "family": infer_family(row["question"]),
                }
            )
        return rows


def build_trace(row: dict[str, str], verified_bits: dict[str, dict[str, str]]) -> tuple[str, str]:
    family = row["family"]
    question = row["question"]
    answer = row["gold_answer"]
    row_id = row["id"]

    if family == "cipher":
        trace = build_cipher_v2_trace(question, answer)
        if trace is not None:
            return trace, "cipher_strict_cited_trace"
        return build_boxed_target(answer), "cipher_boxed_only_unsupported"

    if family == "bit_manipulation":
        if row_id in verified_bits:
            trace = build_bit_execution_trace(question, answer, verified_bits[row_id])
            if trace is not None:
                return trace, "bit_verified_execution_trace"
        return build_boxed_target(answer), "bit_boxed_only_unverified"

    if family == "equation_symbolic":
        return build_boxed_target(answer), "equation_boxed_only"

    if family == "unit_conversion":
        trace = build_unit_short_trace(question, answer)
        if trace is not None:
            return trace, "unit_short_trace"
        return build_boxed_target(answer), "unit_boxed_only_fallback"

    if family == "gravity":
        trace = build_gravity_compact_trace(question, answer)
        if trace is not None:
            return trace, "gravity_compact_trace"
        return build_boxed_target(answer), "gravity_boxed_only_fallback"

    if family == "numeral":
        return build_boxed_target(answer), "numeral_boxed_only"

    return build_boxed_target(answer), f"{family}_boxed_only"


def build_rows(
    train_csv: Path,
    bit_audit_csv: Path,
    easy_fraction: float,
) -> tuple[list[dict[str, str]], Counter[str], Counter[str]]:
    if easy_fraction < 0 or easy_fraction > 1:
        raise ValueError("--easy-fraction must be between 0 and 1")

    rows = read_train_rows(train_csv)
    verified_bits = read_verified_bit_rules(bit_audit_csv)
    by_family: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_family.setdefault(row["family"], []).append(row)

    selected: list[dict[str, str]] = []
    for family in sorted(HARD_FAMILIES):
        selected.extend(by_family.get(family, []))
    for family in sorted(EASY_FAMILIES):
        family_rows = by_family.get(family, [])
        count = round(len(family_rows) * easy_fraction)
        selected.extend(select_rows(family_rows, count, f"{family}_p{easy_fraction}"))

    selected = select_rows(selected, len(selected), "all_hard_curriculum_order")
    output_rows: list[dict[str, str]] = []
    family_counts: Counter[str] = Counter()
    trace_counts: Counter[str] = Counter()

    for row in selected:
        trace, trace_kind = build_trace(row, verified_bits)
        family_counts[row["family"]] += 1
        trace_counts[trace_kind] += 1
        output_rows.append(
            {
                "id": row["id"],
                "question": row["question"],
                "trace": trace,
                "gold_answer": row["gold_answer"],
            }
        )

    return output_rows, family_counts, trace_counts


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "question", "trace", "gold_answer"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=Path("data/input/official/train.csv"))
    parser.add_argument("--bit-audit-csv", type=Path, default=Path("data/input/verifier/bit_candidate_trace_audit.csv"))
    parser.add_argument("--easy-fraction", type=float, default=0.25)
    parser.add_argument("--output-csv", type=Path, default=Path("data/input/traces/trace_training_v2_all_hard_p25.csv"))
    args = parser.parse_args()

    rows, family_counts, trace_counts = build_rows(args.train_csv, args.bit_audit_csv, args.easy_fraction)
    write_csv(args.output_csv, rows)
    print(f"wrote {args.output_csv} ({len(rows)} rows)")
    print("families:", dict(sorted(family_counts.items())))
    print("trace kinds:", dict(sorted(trace_counts.items())))


if __name__ == "__main__":
    main()
