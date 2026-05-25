"""Build a cipher-only trace CSV with position-level evidence.

This is a focused follow-up to the v2 cited-map traces. The target is not more
generic explanation; it is mechanical alignment:

- include only cipher rows whose target answer is fully supported by examples
- cite an aligned example word and character position for every target character
- assemble each plaintext word explicitly before the final boxed answer
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path

from build_trace_training_data import (
    PUBLIC_SANITY_IDS,
    boxed_answer,
    collect_char_map,
    infer_family,
    normalize_answer,
    parse_cipher_prompt,
    validate_word_alignment,
)


def stable_key(row_id: str, salt: str) -> str:
    return hashlib.sha1(f"{salt}:{row_id}".encode("utf-8")).hexdigest()


def aligned_example_positions(
    examples: list[tuple[str, str]],
) -> tuple[dict[str, dict[str, str | int]], list[str]]:
    """Return first position-level evidence for each encrypted character."""
    evidence: dict[str, dict[str, str | int]] = {}
    conflicts: list[str] = []
    char_map: dict[str, str] = {}

    for cipher_text, plain_text in examples:
        cipher_words = cipher_text.split()
        plain_words = plain_text.split()
        if len(cipher_words) != len(plain_words):
            conflicts.append(f"word-count mismatch: {cipher_text!r} -> {plain_text!r}")
            continue
        for cipher_word, plain_word in zip(cipher_words, plain_words, strict=True):
            if len(cipher_word) != len(plain_word):
                conflicts.append(f"word-length mismatch: {cipher_word!r} -> {plain_word!r}")
                continue
            for index, (cipher_char, plain_char) in enumerate(zip(cipher_word, plain_word, strict=True), start=1):
                previous = char_map.get(cipher_char)
                if previous is not None and previous != plain_char:
                    conflicts.append(f"{cipher_char!r}: {previous!r} vs {plain_char!r}")
                    continue
                char_map[cipher_char] = plain_char
                evidence.setdefault(
                    cipher_char,
                    {
                        "plain_char": plain_char,
                        "source_cipher_word": cipher_word,
                        "source_plain_word": plain_word,
                        "source_position": index,
                    },
                )
    return evidence, conflicts


def build_cipher_v3_position_trace(prompt: str, answer: str) -> str | None:
    examples, target_cipher = parse_cipher_prompt(prompt)
    target_plain = normalize_answer(answer)
    target_pairs, reject_reason = validate_word_alignment(target_cipher, target_plain)
    if reject_reason:
        return None

    example_map, _example_sources, example_conflicts = collect_char_map(examples)
    if example_conflicts:
        return None
    target_map, _target_sources, target_conflicts = collect_char_map(target_pairs)
    if target_conflicts:
        return None

    evidence, evidence_conflicts = aligned_example_positions(examples)
    if evidence_conflicts:
        return None

    for cipher_char, plain_char in target_map.items():
        if cipher_char not in example_map:
            return None
        if example_map[cipher_char] != plain_char:
            return None
        if cipher_char not in evidence:
            return None

    lines = [
        "Thinking:",
        "Category: cipher.",
        "Use aligned example word pairs. Copy one target character at a time.",
        "No phrase-context guessing: every target character must cite a source position.",
        f"Target cipher phrase: {target_cipher}.",
        "",
    ]

    for word_index, (cipher_word, plain_word) in enumerate(target_pairs, start=1):
        lines.append(f"Word {word_index}: {cipher_word}")
        assembled_chars: list[str] = []
        for char_index, cipher_char in enumerate(cipher_word, start=1):
            plain_char = target_map[cipher_char]
            source = evidence[cipher_char]
            assembled_chars.append(plain_char)
            lines.append(
                f"- pos {char_index}: {cipher_char}->{plain_char}; "
                f"source {source['source_cipher_word']}->{source['source_plain_word']} "
                f"pos {source['source_position']}."
            )
        lines.append(f"Assemble word {word_index}: {' '.join(assembled_chars)} => {plain_word}.")
        lines.append("")

    lines.extend(
        [
            "Check: every target character above has source-position evidence.",
            "",
            "Final answer:",
            boxed_answer(target_plain),
        ]
    )
    return "\n".join(lines)


def read_cipher_rows(train_csv: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with train_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["id"] in PUBLIC_SANITY_IDS:
                continue
            if infer_family(row["question"]) != "cipher":
                continue
            rows.append(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "gold_answer": normalize_answer(row["gold_answer"]),
                }
            )
    return rows


def build_rows(train_csv: Path) -> tuple[list[dict[str, str]], Counter[str]]:
    output_rows: list[dict[str, str]] = []
    summary: Counter[str] = Counter()

    for row in read_cipher_rows(train_csv):
        summary["cipher_candidates"] += 1
        trace = build_cipher_v3_position_trace(row["question"], row["gold_answer"])
        if trace is None:
            summary["cipher_rejected_unsupported"] += 1
            continue
        summary["cipher_v3_position_trace"] += 1
        output_rows.append(
            {
                "id": row["id"],
                "question": row["question"],
                "trace": trace,
                "gold_answer": row["gold_answer"],
            }
        )

    output_rows = sorted(output_rows, key=lambda row: stable_key(row["id"], "cipher_v3_position"))
    return output_rows, summary


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "question", "trace", "gold_answer"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=Path("data/input/official/train.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/input/traces/trace_training_v3_cipher_only.csv"))
    args = parser.parse_args()

    rows, summary = build_rows(args.train_csv)
    write_csv(args.output_csv, rows)
    print(f"wrote {args.output_csv} ({len(rows)} rows)")
    for key in sorted(summary):
        print(f"{key}: {summary[key]}")


if __name__ == "__main__":
    main()
