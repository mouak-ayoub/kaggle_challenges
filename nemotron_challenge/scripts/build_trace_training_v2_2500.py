"""Build the exp06 fix-tracing SFT input file.

This is a deliberately small curriculum, not a replacement for the canonical
full trace CSV. It focuses on the families that exp05 did not learn:

- strict cipher traces with cited mappings only
- bit-manipulation traces that execute the verified DSL rule
- boxed or compact rehearsal rows for the other families
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from build_trace_training_data import (
    PUBLIC_SANITY_IDS,
    BIT_QUERY_PATTERN,
    boxed_answer,
    build_boxed_target,
    build_gravity_trace,
    collect_char_map,
    infer_family,
    normalize_answer,
    parse_cipher_prompt,
    parse_gravity_prompt,
    parse_unit_conversion_prompt,
    read_verified_bit_rules,
    validate_word_alignment,
)


TARGET_COUNTS = {
    "cipher": 600,
    "bit_manipulation": 900,
    "equation_symbolic": 250,
    "unit_conversion": 250,
    "gravity": 250,
    "numeral": 250,
}

TOKEN_PATTERN = re.compile(r"[A-Z]+|[01]{8}|\d+|x|[(),]")


def stable_key(row_id: str, salt: str) -> str:
    return hashlib.sha1(f"{salt}:{row_id}".encode("utf-8")).hexdigest()


def ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def build_cipher_v2_trace(prompt: str, answer: str) -> str | None:
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
        if cipher_char not in example_map:
            return None
        if example_map[cipher_char] != plain_char:
            return None

    lines = [
        "Thinking:",
        "Category: cipher.",
        "Use aligned example words. Every target mapping must cite a source pair.",
        "",
    ]
    for cipher_word, plain_word in target_pairs:
        citations: list[str] = []
        for cipher_char in ordered_unique(cipher_word):
            plain_char = target_map[cipher_char]
            citations.append(f"{cipher_char}->{plain_char} from {example_sources[cipher_char]}")
        lines.append(f"{cipher_word}: {'; '.join(citations)} => {plain_word}.")

    lines.extend(
        [
            "",
            "Verified: all target characters are supported by cited example pairs.",
            "",
            "Final answer:",
            boxed_answer(target_plain),
        ]
    )
    return "\n".join(lines)


def bits(value: int) -> str:
    return format(value & 0xFF, "08b")


@dataclass(frozen=True)
class TokenStream:
    tokens: list[str]
    index: int = 0

    def current(self) -> str | None:
        if self.index >= len(self.tokens):
            return None
        return self.tokens[self.index]

    def advance(self) -> "TokenStream":
        return TokenStream(self.tokens, self.index + 1)


Ast = tuple


def parse_rule(rule: str) -> Ast:
    tokens = TOKEN_PATTERN.findall(rule)
    if "".join(tokens).replace(",", ",") != re.sub(r"\s+", "", rule):
        raise ValueError(f"unsupported rule syntax: {rule}")
    node, stream = parse_or(TokenStream(tokens))
    if stream.current() is not None:
        raise ValueError(f"trailing tokens in rule: {rule}")
    return node


def parse_or(stream: TokenStream) -> tuple[Ast, TokenStream]:
    node, stream = parse_xor(stream)
    while stream.current() == "OR":
        stream = stream.advance()
        right, stream = parse_xor(stream)
        node = ("bin", "OR", node, right)
    return node, stream


def parse_xor(stream: TokenStream) -> tuple[Ast, TokenStream]:
    node, stream = parse_and(stream)
    while stream.current() == "XOR":
        stream = stream.advance()
        right, stream = parse_and(stream)
        node = ("bin", "XOR", node, right)
    return node, stream


def parse_and(stream: TokenStream) -> tuple[Ast, TokenStream]:
    node, stream = parse_primary(stream)
    while stream.current() == "AND":
        stream = stream.advance()
        right, stream = parse_primary(stream)
        node = ("bin", "AND", node, right)
    return node, stream


def parse_primary(stream: TokenStream) -> tuple[Ast, TokenStream]:
    token = stream.current()
    if token is None:
        raise ValueError("unexpected end of rule")
    if token == "(":
        node, stream = parse_or(stream.advance())
        if stream.current() != ")":
            raise ValueError("missing closing parenthesis")
        return node, stream.advance()
    if token == "x":
        return ("x",), stream.advance()
    if re.fullmatch(r"[01]{8}", token):
        return ("const", int(token, 2), token), stream.advance()
    if token in {"SHL", "SHR", "ROTL", "ROTR"}:
        stream = stream.advance()
        if stream.current() != "(":
            raise ValueError(f"missing open parenthesis after {token}")
        inner, stream = parse_or(stream.advance())
        if stream.current() != ",":
            raise ValueError(f"missing comma after {token} input")
        stream = stream.advance()
        amount = stream.current()
        if amount is None or not amount.isdigit():
            raise ValueError(f"missing shift amount for {token}")
        stream = stream.advance()
        if stream.current() != ")":
            raise ValueError(f"missing close parenthesis after {token}")
        return ("func", token, inner, int(amount)), stream.advance()
    if token == "NOT":
        stream = stream.advance()
        if stream.current() != "(":
            raise ValueError("missing open parenthesis after NOT")
        inner, stream = parse_or(stream.advance())
        if stream.current() != ")":
            raise ValueError("missing close parenthesis after NOT")
        return ("not", inner), stream.advance()
    raise ValueError(f"unexpected token: {token}")


def eval_node(node: Ast, x_value: int) -> int:
    kind = node[0]
    if kind == "x":
        return x_value & 0xFF
    if kind == "const":
        return int(node[1]) & 0xFF
    if kind == "not":
        return (~eval_node(node[1], x_value)) & 0xFF
    if kind == "func":
        op, inner, amount = node[1], node[2], int(node[3]) % 8
        value = eval_node(inner, x_value)
        if op == "SHL":
            return (value << amount) & 0xFF
        if op == "SHR":
            return (value >> amount) & 0xFF
        if op == "ROTL":
            return ((value << amount) | (value >> (8 - amount))) & 0xFF
        if op == "ROTR":
            return ((value >> amount) | (value << (8 - amount))) & 0xFF
        raise ValueError(f"unsupported function: {op}")
    if kind == "bin":
        op, left, right = node[1], node[2], node[3]
        left_value = eval_node(left, x_value)
        right_value = eval_node(right, x_value)
        if op == "XOR":
            return (left_value ^ right_value) & 0xFF
        if op == "AND":
            return (left_value & right_value) & 0xFF
        if op == "OR":
            return (left_value | right_value) & 0xFF
        raise ValueError(f"unsupported binary op: {op}")
    raise ValueError(f"unsupported node: {node}")


def format_node(node: Ast) -> str:
    kind = node[0]
    if kind == "x":
        return "x"
    if kind == "const":
        return str(node[2])
    if kind == "not":
        return f"NOT({format_node(node[1])})"
    if kind == "func":
        return f"{node[1]}({format_node(node[2])},{node[3]})"
    if kind == "bin":
        return f"({format_node(node[2])} {node[1]} {format_node(node[3])})"
    raise ValueError(f"unsupported node: {node}")


def explain_node(node: Ast, x_value: int, seen: set[str]) -> list[str]:
    kind = node[0]
    if kind in {"x", "const"}:
        return []
    lines: list[str] = []
    if kind == "not":
        lines.extend(explain_node(node[1], x_value, seen))
    elif kind == "func":
        lines.extend(explain_node(node[2], x_value, seen))
    elif kind == "bin":
        lines.extend(explain_node(node[2], x_value, seen))
        lines.extend(explain_node(node[3], x_value, seen))

    expression = format_node(node)
    if expression not in seen:
        seen.add(expression)
        lines.append(f"{expression} = {bits(eval_node(node, x_value))}.")
    return lines


def build_bit_execution_trace(prompt: str, answer: str, verified_rule: dict[str, str]) -> str | None:
    query_match = BIT_QUERY_PATTERN.search(prompt)
    if query_match is None:
        return None
    query_bits = query_match.group(1)
    answer_text = normalize_answer(answer)
    if verified_rule["query_bits"] != query_bits:
        return None
    if verified_rule["query_prediction"] != answer_text:
        return None

    x_value = int(query_bits, 2)
    node = parse_rule(verified_rule["rule"])
    if bits(eval_node(node, x_value)) != answer_text:
        return None

    execution_lines = explain_node(node, x_value, set())
    lines = [
        "Thinking:",
        "Category: bit manipulation.",
        "Width: 8 bits. SHL keeps the lowest 8 bits. SHR pads with 0 on the left.",
        f"Rule: y = {verified_rule['rule']}.",
        f"Target x = {query_bits}.",
        *execution_lines,
        "",
        "Final answer:",
        boxed_answer(answer_text),
    ]
    return "\n".join(lines)


def build_unit_short_trace(prompt: str, answer: str) -> str | None:
    examples, target_value, target_value_text = parse_unit_conversion_prompt(prompt)
    answer_text = normalize_answer(answer)
    try:
        target_answer = float(answer_text)
    except ValueError:
        return None
    if not examples:
        return None
    compatible_k = target_answer / target_value
    lines = [
        "Thinking:",
        "Category: unit conversion.",
        f"Use output = k*input with k about {compatible_k:.4f}.",
        f"{target_value_text} * {compatible_k:.4f} is about {answer_text} after rounding.",
        "",
        "Final answer:",
        boxed_answer(answer_text),
    ]
    return "\n".join(lines)


def build_gravity_compact_trace(prompt: str, answer: str) -> str | None:
    examples, target_time, target_time_text = parse_gravity_prompt(prompt)
    answer_text = normalize_answer(answer)
    try:
        target_distance = float(answer_text)
    except ValueError:
        return None
    if not examples:
        return None
    compatible_g = 2 * target_distance / (target_time * target_time)
    lines = [
        "Thinking:",
        "Category: gravity.",
        "Use d = 0.5*g*t^2 and infer g from the rounded examples.",
        f"For the target, use g about {compatible_g:.4f}.",
        f"0.5 * {compatible_g:.4f} * {target_time_text}^2 is about {answer_text} after rounding.",
        "",
        "Final answer:",
        boxed_answer(answer_text),
    ]
    return "\n".join(lines)


def read_train_rows(train_csv: Path) -> list[dict[str, str]]:
    with train_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            {
                "id": row["id"],
                "question": row["question"],
                "gold_answer": normalize_answer(row["gold_answer"]),
            }
            for row in csv.DictReader(handle)
            if row["id"] not in PUBLIC_SANITY_IDS
        ]


def select_rows(rows: list[dict[str, str]], count: int, salt: str) -> list[dict[str, str]]:
    selected = sorted(rows, key=lambda row: stable_key(row["id"], salt))
    return selected[:count]


def build_trace_rows(train_csv: Path, bit_audit_csv: Path) -> tuple[list[dict[str, str]], dict[str, int]]:
    raw_rows = read_train_rows(train_csv)
    verified_bits = read_verified_bit_rules(bit_audit_csv)

    candidates: dict[str, list[dict[str, str]]] = {family: [] for family in TARGET_COUNTS}
    rejected: dict[str, int] = {}

    for row in raw_rows:
        row_id = row["id"]
        family = infer_family(row["question"])
        if family not in candidates:
            continue
        trace: str | None = None
        if family == "cipher":
            trace = build_cipher_v2_trace(row["question"], row["gold_answer"])
        elif family == "bit_manipulation" and row_id in verified_bits:
            trace = build_bit_execution_trace(row["question"], row["gold_answer"], verified_bits[row_id])
        elif family == "equation_symbolic":
            trace = build_boxed_target(row["gold_answer"])
        elif family == "unit_conversion":
            trace = build_unit_short_trace(row["question"], row["gold_answer"])
        elif family == "gravity":
            trace = build_gravity_compact_trace(row["question"], row["gold_answer"])
            if trace is None:
                trace = build_gravity_trace(row["question"], row["gold_answer"])
        elif family == "numeral":
            trace = build_boxed_target(row["gold_answer"])

        if trace is None:
            rejected[family] = rejected.get(family, 0) + 1
            continue
        candidates[family].append(
            {
                "id": row_id,
                "question": row["question"],
                "trace": trace,
                "gold_answer": row["gold_answer"],
            }
        )

    output_rows: list[dict[str, str]] = []
    summary: dict[str, int] = {}
    for family, count in TARGET_COUNTS.items():
        selected = select_rows(candidates[family], count, family)
        if len(selected) != count:
            raise ValueError(f"Need {count} {family} rows, found {len(selected)}")
        summary[family] = len(selected)
        output_rows.extend(selected)

    output_rows = sorted(output_rows, key=lambda row: stable_key(row["id"], "exp06_trace_v2"))
    summary.update({f"candidate_{family}": len(rows) for family, rows in candidates.items()})
    summary.update({f"rejected_{family}": count for family, count in rejected.items()})
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
    parser.add_argument("--bit-audit-csv", type=Path, default=Path("data/input/verifier/bit_candidate_trace_audit.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/input/traces/trace_training_v2_2500.csv"))
    args = parser.parse_args()

    rows, summary = build_trace_rows(args.train_csv, args.bit_audit_csv)
    write_csv(args.output_csv, rows)
    print(f"wrote {args.output_csv} ({len(rows)} rows)")
    for key in sorted(summary):
        print(f"{key}: {summary[key]}")


if __name__ == "__main__":
    main()
