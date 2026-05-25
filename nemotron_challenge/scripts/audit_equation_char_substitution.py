"""Audit equation rows for a conservative same-length character substitution rule."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


EQUATION_MARKER = "secret set of transformation rules is applied to equations"


def parse_equation_prompt(prompt: str) -> tuple[list[tuple[str, str]], str]:
    examples: list[tuple[str, str]] = []
    target = ""
    for raw_line in prompt.splitlines():
        line = raw_line.strip()
        if line.startswith("Now, determine the result for:"):
            target = line.split(":", 1)[1].strip()
            continue
        if " = " in line:
            left, right = line.split(" = ", 1)
            examples.append((left.strip(), right.strip()))

    if not examples:
        raise ValueError("missing equation examples")
    if not target:
        raise ValueError("missing equation target")
    return examples, target


def infer_char_map(pairs: list[tuple[str, str]]) -> tuple[dict[str, str], list[str]]:
    char_map: dict[str, str] = {}
    conflicts: list[str] = []
    for source, target in pairs:
        if len(source) != len(target):
            conflicts.append(f"length mismatch: {source!r} -> {target!r}")
            continue
        for source_char, target_char in zip(source, target, strict=True):
            previous = char_map.get(source_char)
            if previous is not None and previous != target_char:
                conflicts.append(f"{source_char!r}: {previous!r} vs {target_char!r}")
            else:
                char_map[source_char] = target_char
    return char_map, conflicts


def audit_row(row: dict[str, str]) -> dict[str, str]:
    row_id = row["id"]
    question = row["question"]
    gold_answer = row["gold_answer"].strip()

    if EQUATION_MARKER not in question.lower():
        raise ValueError(f"row {row_id} is not equation_symbolic")

    try:
        examples, target = parse_equation_prompt(question)
    except ValueError as exc:
        return {
            "id": row_id,
            "gold_answer": gold_answer,
            "target": "",
            "status": f"parse_error:{exc}",
            "examples_total": "0",
            "examples_same_length": "0",
            "mapping_size": "0",
            "prediction": "",
        }

    same_length_count = sum(1 for source, result in examples if len(source) == len(result))
    if same_length_count != len(examples):
        return {
            "id": row_id,
            "gold_answer": gold_answer,
            "target": target,
            "status": "length_changing_examples",
            "examples_total": str(len(examples)),
            "examples_same_length": str(same_length_count),
            "mapping_size": "0",
            "prediction": "",
        }

    if len(target) != len(gold_answer):
        return {
            "id": row_id,
            "gold_answer": gold_answer,
            "target": target,
            "status": "target_length_mismatch",
            "examples_total": str(len(examples)),
            "examples_same_length": str(same_length_count),
            "mapping_size": "0",
            "prediction": "",
        }

    char_map, conflicts = infer_char_map(examples)
    if conflicts:
        return {
            "id": row_id,
            "gold_answer": gold_answer,
            "target": target,
            "status": "same_length_conflict",
            "examples_total": str(len(examples)),
            "examples_same_length": str(same_length_count),
            "mapping_size": str(len(char_map)),
            "prediction": "",
        }

    if any(char not in char_map for char in target):
        return {
            "id": row_id,
            "gold_answer": gold_answer,
            "target": target,
            "status": "target_unmapped_chars",
            "examples_total": str(len(examples)),
            "examples_same_length": str(same_length_count),
            "mapping_size": str(len(char_map)),
            "prediction": "",
        }

    prediction = "".join(char_map[char] for char in target)
    status = "verified_char_substitution" if prediction == gold_answer else "same_length_wrong_target"
    return {
        "id": row_id,
        "gold_answer": gold_answer,
        "target": target,
        "status": status,
        "examples_total": str(len(examples)),
        "examples_same_length": str(same_length_count),
        "mapping_size": str(len(char_map)),
        "prediction": prediction,
    }


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "gold_answer",
        "target",
        "status",
        "examples_total",
        "examples_same_length",
        "mapping_size",
        "prediction",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=Path("data/input/train.csv"))
    parser.add_argument(
        "--audit-csv",
        type=Path,
        required=True,
        help="Output CSV path for this optional diagnostic audit.",
    )
    args = parser.parse_args()

    equation_rows = [
        row
        for row in read_rows(args.train_csv)
        if EQUATION_MARKER in row["question"].lower()
    ]
    audit_rows = [audit_row(row) for row in equation_rows]
    write_rows(args.audit_csv, audit_rows)

    status_counts: dict[str, int] = {}
    for row in audit_rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1

    print(f"audited {len(audit_rows)} equation rows")
    print(f"wrote {args.audit_csv}")
    print("status counts:", dict(sorted(status_counts.items())))


if __name__ == "__main__":
    main()
