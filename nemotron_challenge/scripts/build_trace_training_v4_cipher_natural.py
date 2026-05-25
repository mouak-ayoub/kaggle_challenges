"""Build model-native natural-alignment cipher traces.

This is the v4 replacement for the long v3 position-citation traces. The goal is
to preserve the base model's useful pre-training behavior as observed in the
before-training probe:

- infer a cipher-to-plain mapping from aligned example words
- explicitly list cipher/plain word alignment before character mapping
- check repeated mappings for consistency
- apply the resulting mapping to the target phrase
- reach the boxed final answer quickly

The default source is the augmented 2,500-row cipher CSV from v3, but this
builder ignores the old trace column and regenerates the target trace.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from collections import Counter
from pathlib import Path

from build_trace_training_data import (
    boxed_answer,
    collect_char_map,
    infer_family,
    normalize_answer,
    parse_cipher_prompt,
    validate_word_alignment,
)


BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]*)\}")


def stable_key(row_id: str, salt: str) -> str:
    return hashlib.sha1(f"{salt}:{row_id}".encode("utf-8")).hexdigest()


def ordered_unique(values: str) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def compact_source(source: str) -> str:
    return source.replace(" -> ", "->")


def numbered_words(words: list[str]) -> str:
    return ", ".join(f"{index}:{word}" for index, word in enumerate(words, start=1))


def sentence_mapping_lines(
    cipher_text: str,
    plain_text: str,
    known_map: dict[str, str],
    update_known_map: bool,
) -> tuple[list[str], dict[str, str]]:
    """Format the probe-style word-by-word mapping for one example sentence."""
    cipher_words = cipher_text.split()
    plain_words = plain_text.split()
    lines: list[str] = [
        f'Cipher: "{cipher_text}" -> plaintext: "{plain_text}"',
        "",
        f"Let's align words. Cipher words: {numbered_words(cipher_words)}",
        f"Plain words: {numbered_words(plain_words)}",
        "",
        "So mapping per word? Let's map letters.",
        "",
    ]

    next_map = dict(known_map)
    for word_index, (cipher_word, plain_word) in enumerate(zip(cipher_words, plain_words, strict=True), start=1):
        parts: list[str] = []
        for cipher_char, plain_char in zip(cipher_word, plain_word, strict=True):
            previous = next_map.get(cipher_char)
            suffix = ""
            if previous == plain_char:
                suffix = " (consistent)"
            elif previous is not None and previous != plain_char:
                suffix = f" (conflict: expected {previous})"
            parts.append(f"{cipher_char}->{plain_char}{suffix}")
            if update_known_map:
                next_map[cipher_char] = plain_char
        lines.append(f'Word{word_index} cipher "{cipher_word}" -> "{plain_word}". So {", ".join(parts)}.')
    return lines, next_map


def target_application_lines(
    target_pairs: list[tuple[str, str]],
    target_map: dict[str, str],
    example_sources: dict[str, str],
) -> list[str]:
    lines = [
        "Now apply the mapping to the target text.",
        "",
    ]
    for word_index, (cipher_word, plain_word) in enumerate(target_pairs, start=1):
        parts: list[str] = []
        assembled_chars: list[str] = []
        for cipher_char in cipher_word:
            plain_char = target_map[cipher_char]
            assembled_chars.append(plain_char)
            parts.append(f"{cipher_char}->{plain_char}")
        source_parts = [
            f"{cipher_char}->{target_map[cipher_char]} from {compact_source(example_sources[cipher_char])}"
            for cipher_char in ordered_unique(cipher_word)
        ]
        lines.append(f'Target word{word_index} cipher "{cipher_word}". So {", ".join(parts)}.')
        lines.append(f"Sources: {'; '.join(source_parts)}.")
        lines.append(f"{cipher_word} -> {' '.join(assembled_chars)} -> {plain_word}.")
        lines.append("")
    return lines


def build_cipher_v4_natural_trace(prompt: str, answer: str) -> str | None:
    examples, target_cipher = parse_cipher_prompt(prompt)
    target_plain = normalize_answer(answer)
    target_pairs, reject_reason = validate_word_alignment(target_cipher, target_plain)
    if reject_reason:
        return None

    example_map, example_sources, example_conflicts = collect_char_map(examples)
    if example_conflicts:
        return None

    target_map, _target_sources, target_conflicts = collect_char_map(target_pairs)
    if target_conflicts:
        return None

    for cipher_char, plain_char in target_map.items():
        if example_map.get(cipher_char) != plain_char:
            return None

    lines = [
        "We need to find mapping from cipher to plaintext.",
        "Given examples:",
        "",
    ]

    known_map: dict[str, str] = {}
    example_lines, known_map = sentence_mapping_lines(
        examples[0][0],
        examples[0][1],
        known_map,
        update_known_map=True,
    )
    lines.extend(example_lines)
    lines.append("")

    remaining = len(examples) - 1
    if remaining > 0:
        lines.append(f"The other {remaining} example(s) give more aligned word mappings and consistency checks.")
        lines.append("")

    lines.extend(
        [
            f'Target cipher text: "{target_cipher}"',
            "",
            *target_application_lines(target_pairs, target_map, example_sources),
        ]
    )

    lines.extend(
        [
            f"So the plaintext phrase is: {target_plain}.",
            "",
            "Final answer:",
            boxed_answer(target_plain),
        ]
    )
    return "\n".join(lines)


def read_source_rows(source_csv: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with source_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            question = row.get("question", "")
            answer = row.get("gold_answer", "")
            if not question or not answer:
                continue
            if infer_family(question) != "cipher":
                continue
            rows.append(
                {
                    "id": row["id"],
                    "question": question,
                    "gold_answer": normalize_answer(answer),
                }
            )
    return rows


def build_rows(source_csv: Path) -> tuple[list[dict[str, str]], Counter[str]]:
    output_rows: list[dict[str, str]] = []
    summary: Counter[str] = Counter()

    for row in read_source_rows(source_csv):
        summary["cipher_candidates"] += 1
        trace = build_cipher_v4_natural_trace(row["question"], row["gold_answer"])
        if trace is None:
            summary["cipher_rejected_unsupported"] += 1
            continue
        summary["cipher_v4_natural_trace"] += 1
        output_rows.append(
            {
                "id": row["id"],
                "question": row["question"],
                "trace": trace,
                "gold_answer": row["gold_answer"],
            }
        )

    output_rows = sorted(output_rows, key=lambda row: stable_key(row["id"], "cipher_v4_natural"))
    return output_rows, summary


def audit_rows(rows: list[dict[str, str]]) -> Counter[str | int]:
    summary: Counter[str | int] = Counter()
    lengths: list[int] = []
    for row in rows:
        trace = row["trace"]
        lengths.append(len(trace))
        boxed = BOXED_PATTERN.findall(trace)
        if len(boxed) != 1:
            summary["bad_box_count"] += 1
        elif normalize_answer(boxed[0]) != normalize_answer(row["gold_answer"]):
            summary["boxed_gold_mismatch"] += 1
        if any(ord(char) > 127 for char in trace):
            summary["non_ascii_trace"] += 1
        if "source position" in trace or "pos " in trace:
            summary["position_language_rows"] += 1

    if lengths:
        summary["trace_chars_min"] = min(lengths)
        summary["trace_chars_avg"] = round(sum(lengths) / len(lengths))
        summary["trace_chars_max"] = max(lengths)
    return summary


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "question", "trace", "gold_answer"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, default=Path("data/input/traces/trace_cipher_v3_synth2500.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/input/traces/trace_cipher_v4_natural2500.csv"))
    args = parser.parse_args()

    rows, summary = build_rows(args.source_csv)
    write_csv(args.output_csv, rows)
    audit = audit_rows(rows)

    print(f"wrote {args.output_csv} ({len(rows)} rows)")
    for key in sorted(summary):
        print(f"{key}: {summary[key]}")
    for key in sorted(audit, key=str):
        print(f"{key}: {audit[key]}")
    if rows:
        print("\nfirst trace preview:\n")
        print(rows[0]["trace"][:1600])


if __name__ == "__main__":
    main()
