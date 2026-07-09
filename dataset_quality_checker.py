"""CLI tool for auditing AI/LLM training and evaluation datasets stored as
JSONL, catching common data-quality issues before they reach a training run.

Every check below is a standalone, independently testable function; the CLI
in main() just wires them together and formats the result.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Loads a JSONL file (one JSON object per line) into a list of dicts.

    Blank lines are skipped. Raises ValueError with the offending line number
    if a line is not valid JSON, so bad input is caught early rather than
    surfacing as a confusing downstream error.
    """
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error
    return records


def _record_text(record: Dict[str, Any], text_fields: Optional[Sequence[str]] = None) -> str:
    """Concatenates the text content of a record into a single string.

    If `text_fields` is given, only those fields are used (in order);
    otherwise every string-valued field in the record is concatenated.
    """
    if text_fields:
        values = [str(record.get(field, "")) for field in text_fields]
    else:
        values = [value for value in record.values() if isinstance(value, str)]
    return " ".join(values)


def _record_hash(record: Dict[str, Any]) -> str:
    return json.dumps(record, sort_keys=True)


def find_exact_duplicates(records: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    """Finds pairs of records that are byte-for-byte identical (after key
    normalization).

    Time complexity: O(n) - each record is hashed once and looked up in a
    dict, rather than compared pairwise.
    """
    seen: Dict[str, int] = {}
    duplicates: List[Tuple[int, int]] = []

    for index, record in enumerate(records):
        key = _record_hash(record)
        if key in seen:
            duplicates.append((seen[key], index))
        else:
            seen[key] = index

    return duplicates


def _jaccard_similarity(a: str, b: str) -> float:
    """Jaccard similarity between the word-token sets of two strings.

    A simplicity/precision trade-off: this catches near-duplicate phrasing
    cheaply without pulling in an embedding model, at the cost of missing
    duplicates that are semantically identical but lexically different.
    """
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union


def find_near_duplicates(
    records: List[Dict[str, Any]],
    text_fields: Optional[Sequence[str]] = None,
    threshold: float = 0.9,
) -> List[Tuple[int, int, float]]:
    """Finds pairs of records whose text content is highly similar but not
    identical (similarity >= `threshold`, exact duplicates excluded).

    Time complexity: O(n^2) in the number of records - each pair is compared
    once. This is the most expensive check in the suite and is the reason
    dataset audits like this are usually run on a sample or a diffed subset
    for very large datasets, not the full corpus on every commit.
    """
    near_duplicates: List[Tuple[int, int, float]] = []
    texts = [_record_text(record, text_fields) for record in records]

    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            if texts[i] == texts[j]:
                continue
            similarity = _jaccard_similarity(texts[i], texts[j])
            if similarity >= threshold:
                near_duplicates.append((i, j, round(similarity, 4)))

    return near_duplicates


def find_missing_fields(
    records: List[Dict[str, Any]],
    required_fields: Sequence[str],
) -> List[Tuple[int, List[str]]]:
    """Finds records missing any of `required_fields`, or where the field is
    present but empty (empty string, None, or empty collection).

    Time complexity: O(n * k), k = number of required fields.
    """
    issues: List[Tuple[int, List[str]]] = []

    for index, record in enumerate(records):
        missing = [
            field
            for field in required_fields
            if field not in record or record[field] in (None, "", [], {})
        ]
        if missing:
            issues.append((index, missing))

    return issues


def find_length_outliers(
    records: List[Dict[str, Any]],
    text_fields: Optional[Sequence[str]] = None,
    num_std: float = 2.0,
) -> List[Tuple[int, int, float, float]]:
    """Finds records whose text length is more than `num_std` standard
    deviations from the dataset's mean length.

    Time complexity: O(n) - one pass to compute lengths and statistics, one
    pass to flag outliers.
    Returns a list of (index, length, mean, std_dev).
    """
    if len(records) < 2:
        return []

    lengths = [len(_record_text(record, text_fields)) for record in records]
    mean = statistics.mean(lengths)
    std_dev = statistics.pstdev(lengths)

    if std_dev == 0:
        return []

    outliers: List[Tuple[int, int, float, float]] = []
    for index, length in enumerate(lengths):
        if abs(length - mean) > num_std * std_dev:
            outliers.append((index, length, round(mean, 2), round(std_dev, 2)))

    return outliers


def label_distribution(records: List[Dict[str, Any]], label_field: str) -> Dict[str, int]:
    """Counts how many records fall under each value of `label_field`.

    Time complexity: O(n). Records missing the field are counted under the
    key "<missing>" so imbalance in labeling coverage is visible too.
    """
    counts: Dict[str, int] = {}
    for record in records:
        label = str(record.get(label_field, "<missing>"))
        counts[label] = counts.get(label, 0) + 1
    return counts


def build_report(
    records: List[Dict[str, Any]],
    required_fields: Sequence[str] = (),
    label_field: Optional[str] = None,
    text_fields: Optional[Sequence[str]] = None,
    similarity_threshold: float = 0.9,
    length_std: float = 2.0,
) -> Dict[str, Any]:
    """Runs every check and assembles a single structured report."""
    report: Dict[str, Any] = {
        "record_count": len(records),
        "exact_duplicates": find_exact_duplicates(records),
        "near_duplicates": find_near_duplicates(records, text_fields, similarity_threshold),
        "length_outliers": find_length_outliers(records, text_fields, length_std),
    }

    if required_fields:
        report["missing_fields"] = find_missing_fields(records, required_fields)

    if label_field:
        report["label_distribution"] = label_distribution(records, label_field)

    return report


def format_report_human(report: Dict[str, Any]) -> str:
    """Formats a report dict as a readable multi-line summary."""
    lines = [f"Records checked: {report['record_count']}", ""]

    lines.append(f"Exact duplicates: {len(report['exact_duplicates'])}")
    for i, j in report["exact_duplicates"]:
        lines.append(f"  - records {i} and {j} are identical")

    lines.append(f"Near-duplicates: {len(report['near_duplicates'])}")
    for i, j, similarity in report["near_duplicates"]:
        lines.append(f"  - records {i} and {j} are {similarity:.0%} similar")

    lines.append(f"Length outliers: {len(report['length_outliers'])}")
    for index, length, mean, std_dev in report["length_outliers"]:
        lines.append(f"  - record {index}: length {length} (dataset mean {mean}, std dev {std_dev})")

    if "missing_fields" in report:
        lines.append(f"Records with missing/empty required fields: {len(report['missing_fields'])}")
        for index, missing in report["missing_fields"]:
            lines.append(f"  - record {index}: missing {', '.join(missing)}")

    if "label_distribution" in report:
        lines.append("Label distribution:")
        total = sum(report["label_distribution"].values()) or 1
        for label, count in sorted(report["label_distribution"].items()):
            lines.append(f"  - {label}: {count} ({count / total:.1%})")

    return "
".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Audit an AI/LLM dataset (JSONL) for common data-quality issues.")
    parser.add_argument("dataset", help="Path to a JSONL dataset file.")
    parser.add_argument("--required-fields", nargs="*", default=[], help="Fields that every record must have (non-empty).")
    parser.add_argument("--label-field", default=None, help="Field to report label distribution for.")
    parser.add_argument("--text-fields", nargs="*", default=None, help="Fields to use for similarity/length checks (default: all string fields).")
    parser.add_argument("--similarity-threshold", type=float, default=0.9, help="Jaccard similarity threshold for near-duplicate detection (default: 0.9).")
    parser.add_argument("--length-std", type=float, default=2.0, help="Standard deviation threshold for length outliers (default: 2.0).")
    parser.add_argument("--format", choices=["human", "json"], default="human", help="Output format (default: human).")

    args = parser.parse_args(argv)

    try:
        records = load_jsonl(args.dataset)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    report = build_report(
        records,
        required_fields=args.required_fields,
        label_field=args.label_field,
        text_fields=args.text_fields,
        similarity_threshold=args.similarity_threshold,
        length_std=args.length_std,
    )

    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        print(format_report_human(report))

    return 0


if __name__ == "__main__":
    sys.exit(main())

