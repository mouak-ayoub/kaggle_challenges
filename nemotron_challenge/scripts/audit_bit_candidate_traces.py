"""Audit candidate bit-manipulation traces with a small 8-bit DSL verifier."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


BIT_MARKER = "secret bit manipulation rule"
PUBLIC_SANITY_IDS = {"00066667", "000b53cf", "00189f6a"}
RULE_PATTERN = re.compile(
    r"A rule that matches all example pairs is\s*y\s*=\s*([^\.\n]+)",
    re.IGNORECASE,
)
EXAMPLE_PATTERN = re.compile(r"([01]{8})\s*->\s*([01]{8})")
QUERY_PATTERN = re.compile(r"Now, determine the output for:\s*([01]{8})")
MASK = 0xFF


class ParseError(ValueError):
    pass


@dataclass(frozen=True)
class ParseResult:
    expr: object
    position: int


def bits_to_int(bits: str) -> int:
    return int(bits, 2)


def int_to_bits(value: int) -> str:
    return f"{value & MASK:08b}"


def shl(value: int, amount: int) -> int:
    return (value << amount) & MASK


def shr(value: int, amount: int) -> int:
    return (value & MASK) >> amount


def rotl(value: int, amount: int) -> int:
    amount %= 8
    value &= MASK
    return ((value << amount) | (value >> (8 - amount))) & MASK


def rotr(value: int, amount: int) -> int:
    amount %= 8
    value &= MASK
    return ((value >> amount) | (value << (8 - amount))) & MASK


def tokenize(expression: str) -> list[str]:
    tokens = re.findall(r"[A-Z]+|[01]{8}|\d+|x|[,()]", expression)
    compact = "".join(tokens)
    source = re.sub(r"\s+", "", expression)
    if compact != source:
        raise ParseError(f"unsupported characters in expression: {expression!r}")
    return tokens


def parse_expression(tokens: list[str], position: int = 0, min_precedence: int = 1) -> ParseResult:
    left_result = parse_atom(tokens, position)
    left = left_result.expr
    position = left_result.position

    precedence = {"OR": 1, "XOR": 2, "AND": 3}
    while position < len(tokens):
        op = tokens[position]
        if op not in precedence or precedence[op] < min_precedence:
            break
        next_min = precedence[op] + 1
        right_result = parse_expression(tokens, position + 1, next_min)
        left = (op, left, right_result.expr)
        position = right_result.position

    return ParseResult(left, position)


def parse_atom(tokens: list[str], position: int) -> ParseResult:
    if position >= len(tokens):
        raise ParseError("unexpected end of expression")

    token = tokens[position]
    if token == "x":
        return ParseResult(("VAR",), position + 1)
    if re.fullmatch(r"[01]{8}", token):
        return ParseResult(("CONST", bits_to_int(token)), position + 1)
    if token == "(":
        inner = parse_expression(tokens, position + 1)
        if inner.position >= len(tokens) or tokens[inner.position] != ")":
            raise ParseError("missing closing parenthesis")
        return ParseResult(inner.expr, inner.position + 1)

    if token not in {"SHL", "SHR", "ROTL", "ROTR", "NOT"}:
        raise ParseError(f"unexpected token: {token}")
    if position + 1 >= len(tokens) or tokens[position + 1] != "(":
        raise ParseError(f"expected '(' after {token}")

    if token == "NOT":
        inner = parse_expression(tokens, position + 2)
        if inner.position >= len(tokens) or tokens[inner.position] != ")":
            raise ParseError("missing closing parenthesis after NOT")
        return ParseResult(("NOT", inner.expr), inner.position + 1)

    if position + 2 >= len(tokens) or tokens[position + 2] != "x":
        raise ParseError(f"{token} currently supports only x as its first argument")
    if position + 3 >= len(tokens) or tokens[position + 3] != ",":
        raise ParseError(f"expected comma in {token}")
    if position + 4 >= len(tokens) or not re.fullmatch(r"\d+", tokens[position + 4]):
        raise ParseError(f"expected shift amount in {token}")
    if position + 5 >= len(tokens) or tokens[position + 5] != ")":
        raise ParseError(f"missing closing parenthesis in {token}")
    return ParseResult((token, int(tokens[position + 4])), position + 6)


def parse_rule_expression(expression: str) -> object:
    tokens = tokenize(expression)
    result = parse_expression(tokens)
    if result.position != len(tokens):
        raise ParseError(f"unparsed tokens: {' '.join(tokens[result.position:])}")
    return result.expr


def evaluate_expr(expr: object, x_value: int) -> int:
    op = expr[0]
    if op == "VAR":
        return x_value & MASK
    if op == "CONST":
        return expr[1] & MASK
    if op == "NOT":
        return (~evaluate_expr(expr[1], x_value)) & MASK
    if op == "SHL":
        return shl(x_value, expr[1])
    if op == "SHR":
        return shr(x_value, expr[1])
    if op == "ROTL":
        return rotl(x_value, expr[1])
    if op == "ROTR":
        return rotr(x_value, expr[1])
    if op == "XOR":
        return evaluate_expr(expr[1], x_value) ^ evaluate_expr(expr[2], x_value)
    if op == "AND":
        return evaluate_expr(expr[1], x_value) & evaluate_expr(expr[2], x_value)
    if op == "OR":
        return evaluate_expr(expr[1], x_value) | evaluate_expr(expr[2], x_value)
    raise ValueError(f"unknown expression op: {op}")


def parse_bit_prompt(prompt: str) -> tuple[list[tuple[str, str]], str]:
    examples = EXAMPLE_PATTERN.findall(prompt)
    query_match = QUERY_PATTERN.search(prompt)
    if not examples:
        raise ValueError("missing bit examples")
    if query_match is None:
        raise ValueError("missing bit query")
    return examples, query_match.group(1)


def extract_rule(trace: str) -> str | None:
    match = RULE_PATTERN.search(trace)
    if match is None:
        return None
    return match.group(1).strip()


def audit_row(row: dict[str, str]) -> dict[str, str]:
    row_id = row["id"]
    question = row["question"]
    gold_answer = row["gold_answer"].strip()
    trace = row["trace"]

    if BIT_MARKER not in question.lower():
        raise ValueError(f"row {row_id} is not bit manipulation")

    examples, query_bits = parse_bit_prompt(question)
    rule_text = extract_rule(trace)
    if rule_text is None:
        return {
            "id": row_id,
            "gold_answer": gold_answer,
            "query_bits": query_bits,
            "rule": "",
            "status": "no_parseable_rule",
            "examples_total": str(len(examples)),
            "examples_matched": "0",
            "query_prediction": "",
            "answer_correct": "false",
        }

    try:
        expr = parse_rule_expression(rule_text)
    except ParseError as exc:
        return {
            "id": row_id,
            "gold_answer": gold_answer,
            "query_bits": query_bits,
            "rule": rule_text,
            "status": f"parse_error:{exc}",
            "examples_total": str(len(examples)),
            "examples_matched": "0",
            "query_prediction": "",
            "answer_correct": "false",
        }

    matches = 0
    for input_bits, output_bits in examples:
        predicted = int_to_bits(evaluate_expr(expr, bits_to_int(input_bits)))
        if predicted == output_bits:
            matches += 1

    query_prediction = int_to_bits(evaluate_expr(expr, bits_to_int(query_bits)))
    answer_correct = query_prediction == gold_answer
    if matches == len(examples) and answer_correct:
        status = "valid_rule_and_correct_answer"
    elif matches == len(examples):
        status = "valid_rule_wrong_answer"
    elif answer_correct:
        status = "invalid_rule_but_correct_answer"
    else:
        status = "invalid_rule_wrong_answer"

    return {
        "id": row_id,
        "gold_answer": gold_answer,
        "query_bits": query_bits,
        "rule": rule_text,
        "status": status,
        "examples_total": str(len(examples)),
        "examples_matched": str(matches),
        "query_prediction": query_prediction,
        "answer_correct": str(answer_correct).lower(),
    }


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "gold_answer",
        "query_bits",
        "rule",
        "status",
        "examples_total",
        "examples_matched",
        "query_prediction",
        "answer_correct",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-csv",
        type=Path,
        required=True,
        help="Candidate trace CSV to audit. This is not part of the canonical data/input/traces set.",
    )
    parser.add_argument(
        "--audit-csv",
        type=Path,
        default=Path("data/input/verifier/bit_candidate_trace_audit.csv"),
    )
    args = parser.parse_args()

    bit_rows = [
        row
        for row in read_rows(args.candidate_csv)
        if row["id"] not in PUBLIC_SANITY_IDS and BIT_MARKER in row["question"].lower()
    ]
    audit_rows = [audit_row(row) for row in bit_rows]
    write_rows(args.audit_csv, audit_rows)

    status_counts: dict[str, int] = {}
    for row in audit_rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1

    print(f"audited {len(audit_rows)} bit rows")
    print(f"wrote {args.audit_csv}")
    print("status counts:", dict(sorted(status_counts.items())))


if __name__ == "__main__":
    main()
