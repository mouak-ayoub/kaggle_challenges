"""Build trace-enriched SFT input rows for the Nemotron challenge.

The input schema is intentionally small:

- data/input/official/train.csv: id, question, gold_answer
- data/input/official/test.csv: id, question
- data/input/traces/trace_training.csv: id, question, trace, gold_answer
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path


CIPHER_MARKER = "secret encryption rules are used on text"
CIPHER_COMPACT_TEMPLATE_BUCKETS = 7
NUMBER_PATTERN = r"[-+]?\d+(?:\.\d+)?"
BIT_EXAMPLE_PATTERN = re.compile(r"([01]{8})\s*->\s*([01]{8})")
BIT_QUERY_PATTERN = re.compile(r"Now, determine the output for:\s*([01]{8})")
PUBLIC_SANITY_IDS = {"00066667", "000b53cf", "00189f6a"}


def normalize_answer(text: str) -> str:
    return " ".join(str(text).strip().split())


def boxed_answer(answer: str) -> str:
    return f"\\boxed{{{normalize_answer(answer)}}}"


def infer_family(prompt: str) -> str:
    lower = prompt.lower()
    if CIPHER_MARKER in lower:
        return "cipher"
    if "secret bit manipulation rule" in lower:
        return "bit_manipulation"
    if "secret unit conversion" in lower:
        return "unit_conversion"
    if "gravitational constant" in lower:
        return "gravity"
    if "secret set of transformation rules is applied to equations" in lower:
        return "equation_symbolic"
    if "numbers are secretly converted into a different numeral system" in lower:
        return "numeral"
    return "unknown"


def build_boxed_target(answer: str) -> str:
    return f"Final answer:\n{boxed_answer(answer)}"


def parse_cipher_prompt(prompt: str) -> tuple[list[tuple[str, str]], str]:
    lines = [line.strip() for line in prompt.splitlines() if line.strip()]
    examples: list[tuple[str, str]] = []
    target = ""

    for line in lines:
        if line.startswith("Now, decrypt the following text:"):
            target = line.split(":", 1)[1].strip()
            continue
        if " -> " in line:
            left, right = line.split(" -> ", 1)
            examples.append((left.strip(), right.strip()))

    if not target:
        raise ValueError("missing target decrypt text")
    if not examples:
        raise ValueError("missing cipher examples")
    return examples, target


def parse_gravity_prompt(prompt: str) -> tuple[list[tuple[float, float, str, str]], float, str]:
    examples = [
        (float(time), float(distance), time, distance)
        for time, distance in re.findall(
            rf"For\s+t\s*=\s*({NUMBER_PATTERN})s,\s*distance\s*=\s*({NUMBER_PATTERN})\s*m",
            prompt,
        )
    ]
    target_match = re.search(
        rf"Now, determine the falling distance for t\s*=\s*({NUMBER_PATTERN})s",
        prompt,
    )
    if not examples:
        raise ValueError("missing gravity examples")
    if target_match is None:
        raise ValueError("missing gravity target")
    target_time_text = target_match.group(1)
    return examples, float(target_time_text), target_time_text


def parse_unit_conversion_prompt(prompt: str) -> tuple[list[tuple[float, float, str, str]], float, str]:
    examples = [
        (float(source), float(converted), source, converted)
        for source, converted in re.findall(
            rf"({NUMBER_PATTERN})\s*m\s+becomes\s+({NUMBER_PATTERN})",
            prompt,
        )
    ]
    target_match = re.search(
        rf"Now, convert the following measurement:\s*({NUMBER_PATTERN})\s*m",
        prompt,
    )
    if not examples:
        raise ValueError("missing unit conversion examples")
    if target_match is None:
        raise ValueError("missing unit conversion target")
    target_value_text = target_match.group(1)
    return examples, float(target_value_text), target_value_text


def parse_numeral_prompt(prompt: str) -> tuple[int, str]:
    target_match = re.search(
        rf"Now, write the number\s+({NUMBER_PATTERN})\s+in the Wonderland numeral system\.",
        prompt,
    )
    if target_match is None:
        raise ValueError("missing numeral target")
    target_text = target_match.group(1)
    target_value = float(target_text)
    if not target_value.is_integer():
        raise ValueError("non-integer numeral target")
    return int(target_value), target_text


def parse_bit_prompt(prompt: str) -> tuple[list[tuple[str, str]], str]:
    examples = BIT_EXAMPLE_PATTERN.findall(prompt)
    query_match = BIT_QUERY_PATTERN.search(prompt)
    if not examples:
        raise ValueError("missing bit examples")
    if query_match is None:
        raise ValueError("missing bit query")
    return examples, query_match.group(1)


def validate_word_alignment(cipher_text: str, plain_text: str) -> tuple[list[tuple[str, str]], str]:
    cipher_words = cipher_text.split()
    plain_words = plain_text.split()
    if len(cipher_words) != len(plain_words):
        return [], "target_word_count_mismatch"

    pairs = list(zip(cipher_words, plain_words, strict=True))
    for cipher_word, plain_word in pairs:
        if len(cipher_word) != len(plain_word):
            return [], "target_word_length_mismatch"
    return pairs, ""


def collect_char_map(pairs: list[tuple[str, str]]) -> tuple[dict[str, str], dict[str, str], list[str]]:
    char_map: dict[str, str] = {}
    source_map: dict[str, str] = {}
    conflicts: list[str] = []
    for cipher_text, plain_text in pairs:
        cipher_words = cipher_text.split()
        plain_words = plain_text.split()
        if len(cipher_words) != len(plain_words):
            conflicts.append(f"word-count mismatch: {cipher_text!r} -> {plain_text!r}")
            continue
        for cipher_word, plain_word in zip(cipher_words, plain_words, strict=True):
            if len(cipher_word) != len(plain_word):
                conflicts.append(f"word-length mismatch: {cipher_word!r} -> {plain_word!r}")
                continue
            for cipher_char, plain_char in zip(cipher_word, plain_word, strict=True):
                previous = char_map.get(cipher_char)
                if previous is not None and previous != plain_char:
                    conflicts.append(f"{cipher_char!r}: {previous!r} vs {plain_char!r}")
                else:
                    char_map[cipher_char] = plain_char
                    source_map.setdefault(cipher_char, f"{cipher_word} -> {plain_word}")
    return char_map, source_map, conflicts


def ordered_unique(values: list[str] | str) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def join_human_list(values: list[str]) -> str:
    if len(values) <= 1:
        return "".join(values)
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return f"{', '.join(values[:-1])}, and {values[-1]}"


def select_cipher_template(row_id: str) -> str:
    try:
        bucket = int(row_id, 16) % 10
    except ValueError:
        bucket = sum(ord(char) for char in row_id) % 10
    if bucket < CIPHER_COMPACT_TEMPLATE_BUCKETS:
        return "compact"
    return "source_rich"


def select_unit_conversion_template(row_id: str) -> str:
    try:
        bucket = int(row_id, 16) % 4
    except ValueError:
        bucket = sum(ord(char) for char in row_id) % 4
    if bucket < 2:
        return "short"
    if bucket == 2:
        return "full"
    return "boxed_only"


def select_numeral_template(row_id: str) -> str:
    try:
        bucket = int(row_id, 16) % 100
    except ValueError:
        bucket = sum(ord(char) for char in row_id) % 100
    if bucket < 10:
        return "short"
    return "boxed_only"


def cipher_known_chars_in_target_order(
    target_pairs: list[tuple[str, str]],
    example_map: dict[str, str],
) -> list[str]:
    chars: list[str] = []
    for cipher_word, _plain_word in target_pairs:
        chars.extend(char for char in cipher_word if char in example_map)
    return ordered_unique(chars)


def format_mapping_list(chars: list[str], char_map: dict[str, str]) -> str:
    if not chars:
        return "none"
    return ", ".join(f"{char}->{char_map[char]}" for char in chars)


def partial_plain_word(cipher_word: str, target_map: dict[str, str], example_map: dict[str, str]) -> str:
    return "".join(target_map[char] if char in example_map else "?" for char in cipher_word)


def build_cipher_apply_lines(
    target_pairs: list[tuple[str, str]],
    target_map: dict[str, str],
    example_map: dict[str, str],
) -> list[str]:
    lines: list[str] = []
    for cipher_word, plain_word in target_pairs:
        missing_chars = ordered_unique([char for char in cipher_word if char not in example_map])
        if not missing_chars:
            spaced_plain = " ".join(plain_word)
            lines.append(f"{cipher_word} -> {spaced_plain} -> {plain_word}.")
            continue

        partial_word = partial_plain_word(cipher_word, target_map, example_map)
        spaced_partial = " ".join(partial_word)
        lines.append(f"{cipher_word} -> {spaced_partial} from known mappings.")
        lines.append(
            f"The examples do not determine {join_human_list(missing_chars)} directly; "
            f"phrase context completes {partial_word} as {plain_word}."
        )
    return lines


def build_cipher_compact_trace(
    target_pairs: list[tuple[str, str]],
    target_plain: str,
    target_map: dict[str, str],
    example_map: dict[str, str],
) -> str:
    useful_chars = cipher_known_chars_in_target_order(target_pairs, example_map)
    lines = [
        "Thinking:",
        "Category: cipher. Use same-length encrypted/plain word pairs.",
        "",
        "Needed mappings from aligned examples:",
        f"{format_mapping_list(useful_chars, target_map)}.",
        "",
        "Apply to target:",
        *build_cipher_apply_lines(target_pairs, target_map, example_map),
        "",
        "Final answer:",
        boxed_answer(target_plain),
    ]
    return "\n".join(lines)


def build_cipher_source_rich_trace(
    target_pairs: list[tuple[str, str]],
    target_plain: str,
    target_map: dict[str, str],
    example_map: dict[str, str],
    example_sources: dict[str, str],
) -> str:
    useful_chars = cipher_known_chars_in_target_order(target_pairs, example_map)
    lines = [
        "Thinking:",
        "Category: cipher. Align encrypted words with plaintext words.",
        "",
        "Needed mappings with sources:",
    ]
    if useful_chars:
        for char in useful_chars:
            lines.append(f"{char}->{target_map[char]} from {example_sources[char]}.")
    else:
        lines.append("No target characters are directly mapped by the examples.")

    lines.extend(
        [
            "",
            "Apply known mappings:",
            *build_cipher_apply_lines(target_pairs, target_map, example_map),
            "",
            "Final answer:",
            boxed_answer(target_plain),
        ]
    )
    return "\n".join(lines)


def build_cipher_trace(row_id: str, prompt: str, answer: str) -> tuple[str, str] | None:
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

    conflicts_with_examples = [
        cipher_char
        for cipher_char, plain_char in target_map.items()
        if cipher_char in example_map and example_map[cipher_char] != plain_char
    ]
    if conflicts_with_examples:
        return None

    template = select_cipher_template(row_id)
    if template == "compact":
        trace = build_cipher_compact_trace(target_pairs, target_plain, target_map, example_map)
    else:
        trace = build_cipher_source_rich_trace(
            target_pairs,
            target_plain,
            target_map,
            example_map,
            example_sources,
        )
    return trace, template


def build_gravity_trace(prompt: str, answer: str) -> str | None:
    examples, target_time, target_time_text = parse_gravity_prompt(prompt)
    answer_text = normalize_answer(answer)
    try:
        target_distance = float(answer_text)
    except ValueError:
        return None

    estimates = [(time_text, distance_text, 2 * distance / (time * time)) for time, distance, time_text, distance_text in examples]
    if not estimates:
        return None

    compatible_g = 2 * target_distance / (target_time * target_time)
    displayed_examples = estimates[:3]

    lines = [
        "Thinking:",
        "Category: gravity. Infer the hidden gravitational constant.",
        "",
        "Use d = 0.5*g*t^2, so g = 2*d/t^2.",
        "",
        "Estimate g from examples:",
    ]
    for time_text, distance_text, g_estimate in displayed_examples:
        lines.append(f"t={time_text}, d={distance_text} gives g about {g_estimate:.3f}.")

    lines.extend(
        [
            "",
            f"The distances are rounded, so use a common value near g about {compatible_g:.4f}.",
            "",
            "Apply to target:",
            (
                f"d = 0.5 * {compatible_g:.4f} * {target_time_text}^2 "
                f"is about {answer_text} after rounding."
            ),
            "",
            "Final answer:",
            boxed_answer(answer_text),
        ]
    )
    return "\n".join(lines)


def build_unit_conversion_trace(row_id: str, prompt: str, answer: str) -> tuple[str, str] | None:
    template = select_unit_conversion_template(row_id)
    answer_text = normalize_answer(answer)
    if template == "boxed_only":
        return build_boxed_target(answer_text), template

    examples, target_value, target_value_text = parse_unit_conversion_prompt(prompt)
    try:
        target_answer = float(answer_text)
    except ValueError:
        return None

    compatible_k = target_answer / target_value
    displayed_examples = examples[:3]

    if template == "short":
        lines = [
            "Thinking:",
            "Category: unit conversion. The examples show output = k*input.",
            f"The rounded examples give a common ratio near k = {compatible_k:.4f}.",
            f"For {target_value_text}, {target_value_text}*k is about {answer_text} after rounding.",
            "",
            "Final answer:",
            boxed_answer(answer_text),
        ]
        return "\n".join(lines), template

    lines = [
        "Thinking:",
        "Category: unit conversion. Infer the hidden proportional factor.",
        "",
        "Use output = k*input, so k = output/input.",
        "",
        "Estimate k from examples:",
    ]
    for source_value, converted_value, source_text, converted_text in displayed_examples:
        k_estimate = converted_value / source_value
        lines.append(f"{source_text} -> {converted_text} gives k about {k_estimate:.4f}.")

    lines.extend(
        [
            "",
            f"The outputs are rounded, so use a common value near k = {compatible_k:.4f}.",
            "",
            "Apply to target:",
            f"{target_value_text} * {compatible_k:.4f} is about {answer_text} after rounding.",
            "",
            "Final answer:",
            boxed_answer(answer_text),
        ]
    )
    return "\n".join(lines), template


def int_to_roman(value: int) -> str:
    if value <= 0:
        raise ValueError("Roman numerals require positive integers")
    symbols = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ]
    remaining = value
    parts: list[str] = []
    for amount, symbol in symbols:
        count, remaining = divmod(remaining, amount)
        if count:
            parts.append(symbol * count)
    return "".join(parts)


def roman_place_parts(value: int) -> list[tuple[int, str]]:
    parts: list[tuple[int, str]] = []
    place = 1
    remaining = value
    while remaining:
        digit = remaining % 10
        if digit:
            part_value = digit * place
            parts.append((part_value, int_to_roman(part_value)))
        remaining //= 10
        place *= 10
    return list(reversed(parts))


def build_numeral_trace(row_id: str, prompt: str, answer: str) -> tuple[str, str] | None:
    template = select_numeral_template(row_id)
    answer_text = normalize_answer(answer)
    if template == "boxed_only":
        return build_boxed_target(answer_text), template

    target_value, target_text = parse_numeral_prompt(prompt)
    try:
        expected_answer = int_to_roman(target_value)
    except ValueError:
        return build_boxed_target(answer_text), "boxed_only"
    if expected_answer != answer_text:
        return build_boxed_target(answer_text), "boxed_only"

    parts = roman_place_parts(target_value)
    decomposition = " + ".join(str(part_value) for part_value, _roman in parts)
    part_mappings = " and ".join(f"{part_value} -> {roman}" for part_value, roman in parts)

    lines = [
        "Thinking:",
        "Category: numeral. Use the same Roman-style symbols as the examples.",
        f"Decompose {target_text} as {decomposition}.",
        f"{part_mappings}, so {target_text} -> {answer_text}.",
        "",
        "Final answer:",
        boxed_answer(answer_text),
    ]
    return "\n".join(lines), template


def read_verified_bit_rules(audit_csv: Path | None) -> dict[str, dict[str, str]]:
    if audit_csv is None or not audit_csv.exists():
        return {}
    verified: dict[str, dict[str, str]] = {}
    with audit_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("status") != "valid_rule_and_correct_answer":
                continue
            verified[row["id"]] = row
    return verified


def build_bit_trace(prompt: str, answer: str, verified_rule: dict[str, str]) -> str | None:
    examples, query_bits = parse_bit_prompt(prompt)
    answer_text = normalize_answer(answer)
    if verified_rule["query_bits"] != query_bits:
        return None
    if verified_rule["query_prediction"] != answer_text:
        return None

    displayed_checks = examples[:3]
    lines = [
        "Thinking:",
        "Category: bit manipulation. Use a verified 8-bit DSL rule.",
        "",
        "A rule matching all examples is:",
        f"y = {verified_rule['rule']}.",
        "",
        "Checks:",
    ]
    for input_bits, output_bits in displayed_checks:
        lines.append(f"{input_bits} -> {output_bits}.")

    lines.extend(
        [
            f"This rule matches all {verified_rule['examples_total']} provided examples.",
            "",
            "Apply to target:",
            f"x = {query_bits}",
            f"y = {answer_text}",
            "",
            "Final answer:",
            boxed_answer(answer_text),
        ]
    )
    return "\n".join(lines)


def build_rows(
    train_csv: Path,
    bit_audit_csv: Path | None = None,
) -> tuple[list[dict[str, str]], Counter[str], Counter[str]]:
    trace_rows: list[dict[str, str]] = []
    family_counts: Counter[str] = Counter()
    trace_counts: Counter[str] = Counter()
    verified_bit_rules = read_verified_bit_rules(bit_audit_csv)

    with train_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row_id = raw["id"]
            if row_id in PUBLIC_SANITY_IDS:
                trace_counts["skipped_public_sanity_id"] += 1
                continue
            question = raw["question"]
            gold_answer = normalize_answer(raw["gold_answer"])
            family = infer_family(question)
            trace = build_boxed_target(gold_answer)
            trace_kind = "boxed_answer_only"

            if family == "cipher":
                try:
                    cipher_result = build_cipher_trace(row_id, question, gold_answer)
                except ValueError as exc:
                    trace_kind = f"cipher_rejected:{exc}"
                else:
                    if cipher_result is not None:
                        trace, template = cipher_result
                        trace_kind = f"cipher_{template}_trace_boxed"
                    else:
                        trace_kind = "cipher_rejected"
            elif family == "gravity":
                try:
                    gravity_trace = build_gravity_trace(question, gold_answer)
                except ValueError as exc:
                    trace_kind = f"gravity_rejected:{exc}"
                else:
                    if gravity_trace is not None:
                        trace = gravity_trace
                        trace_kind = "gravity_formula_trace_boxed"
                    else:
                        trace_kind = "gravity_rejected"
            elif family == "unit_conversion":
                try:
                    unit_result = build_unit_conversion_trace(row_id, question, gold_answer)
                except ValueError as exc:
                    trace_kind = f"unit_conversion_rejected:{exc}"
                else:
                    if unit_result is not None:
                        trace, template = unit_result
                        trace_kind = f"unit_conversion_{template}_trace_boxed"
                    else:
                        trace_kind = "unit_conversion_rejected"
            elif family == "numeral":
                try:
                    numeral_result = build_numeral_trace(row_id, question, gold_answer)
                except ValueError as exc:
                    trace_kind = f"numeral_rejected:{exc}"
                else:
                    if numeral_result is not None:
                        trace, template = numeral_result
                        trace_kind = f"numeral_{template}_trace_boxed"
                    else:
                        trace_kind = "numeral_rejected"
            elif family == "bit_manipulation" and row_id in verified_bit_rules:
                try:
                    bit_trace = build_bit_trace(question, gold_answer, verified_bit_rules[row_id])
                except ValueError as exc:
                    trace_kind = f"bit_manipulation_rejected:{exc}"
                else:
                    if bit_trace is not None:
                        trace = bit_trace
                        trace_kind = "bit_manipulation_verified_dsl_trace_boxed"
                    else:
                        trace_kind = "bit_manipulation_rejected"

            family_counts[family] += 1
            trace_counts[trace_kind] += 1
            trace_rows.append(
                {
                    "id": row_id,
                    "question": question,
                    "trace": trace,
                    "gold_answer": gold_answer,
                }
            )

    return trace_rows, family_counts, trace_counts


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=Path("data/input/official/train.csv"))
    parser.add_argument("--trace-csv", type=Path, default=Path("data/input/traces/trace_training.csv"))
    parser.add_argument(
        "--bit-audit-csv",
        type=Path,
        default=Path("data/input/verifier/bit_candidate_trace_audit.csv"),
        help="Optional audit CSV with verified bit-manipulation DSL rules.",
    )
    args = parser.parse_args()

    trace_rows, family_counts, trace_counts = build_rows(args.train_csv, args.bit_audit_csv)
    write_csv(args.trace_csv, trace_rows, fieldnames=["id", "question", "trace", "gold_answer"])

    print(f"wrote {args.trace_csv} ({len(trace_rows)} rows)")
    print("families:", dict(sorted(family_counts.items())))
    print("trace kinds:", dict(sorted(trace_counts.items())))


if __name__ == "__main__":
    main()
