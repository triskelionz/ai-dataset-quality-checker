# AI Dataset Quality Checker

A command-line tool for auditing AI/LLM training and fine-tuning datasets (JSONL) before they get used for training or evaluation. It catches the kind of data-quality issues that quietly degrade model performance: exact duplicates, near-duplicates, missing/empty fields, label imbalance, and length outliers.

## Why this repository

Model quality is bounded by data quality. This tool is meant to be run as a fast, dependency-light pre-flight check on a dataset before it is used for training, fine-tuning, or evaluation - the same kind of check I would want in place before signing off on a batch of AI training data.

## What it checks

| Check | What it catches |
| --- | --- |
| Exact duplicates | Identical records (hash-based, O(n)) |
| Near-duplicates | Records with high text similarity (token Jaccard similarity) that are likely redundant |
| Missing/empty required fields | Records missing a configured required key or with an empty value |
| Length outliers | Records far outside the dataset length distribution (mean +/- N standard deviations) |
| Label distribution | Class/label imbalance, reported as counts and percentages |

## Usage

```bash
python dataset_quality_checker.py sample_data/sample_dataset.jsonl \
    --required-fields prompt response \
    --label-field label \
    --similarity-threshold 0.85
```

Output is a structured report (human-readable by default, `--format json` for machine-readable output) summarizing every issue found, with the record indices involved so they can be located and fixed.

## Example

Given `sample_data/sample_dataset.jsonl` (included in this repo), running the checker with the command above reports:

- 1 exact duplicate pair
- 1 near-duplicate pair
- 1 record with a missing/empty required field
- 1 length outlier
- Label distribution across the label field

## Design notes

- No third-party dependencies - only the Python standard library, so it can be dropped into any pipeline without a dependency negotiation.
- Similarity is computed with Jaccard similarity over word tokens, which adds no dependencies and is good enough to flag near-duplicates for a human to review - a deliberate precision/simplicity trade-off, documented in the code.
- Every check is a standalone function with a docstring describing what it does and its complexity, and is unit-tested independently so checks can be reused outside the CLI.

## Running the tests

```bash
python -m unittest discover -s tests -v
```

## License

MIT

