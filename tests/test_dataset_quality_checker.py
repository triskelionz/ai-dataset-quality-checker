import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_quality_checker import (
    build_report,
    find_exact_duplicates,
    find_length_outliers,
    find_missing_fields,
    find_near_duplicates,
    label_distribution,
)


class TestExactDuplicates(unittest.TestCase):
    def test_finds_identical_records(self):
        records = [{"a": 1}, {"a": 1}, {"a": 2}]
        self.assertEqual(find_exact_duplicates(records), [(0, 1)])

    def test_no_duplicates_returns_empty(self):
        records = [{"a": 1}, {"a": 2}, {"a": 3}]
        self.assertEqual(find_exact_duplicates(records), [])

    def test_key_order_does_not_matter(self):
        records = [{"a": 1, "b": 2}, {"b": 2, "a": 1}]
        self.assertEqual(find_exact_duplicates(records), [(0, 1)])


class TestNearDuplicates(unittest.TestCase):
    def test_finds_similar_but_not_identical(self):
        records = [
            {"text": "the quick brown fox"},
            {"text": "the quick brown fox jumps"},
            {"text": "completely unrelated content here"},
        ]
        near_duplicates = find_near_duplicates(records, text_fields=["text"], threshold=0.5)
        pairs = [(i, j) for i, j, _ in near_duplicates]
        self.assertIn((0, 1), pairs)
        self.assertNotIn((0, 2), pairs)

    def test_exact_duplicates_are_excluded(self):
        records = [{"text": "same text"}, {"text": "same text"}]
        self.assertEqual(find_near_duplicates(records, text_fields=["text"], threshold=0.5), [])

    def test_threshold_controls_sensitivity(self):
        records = [
            {"text": "alpha beta gamma"},
            {"text": "alpha beta delta"},
        ]
        self.assertEqual(find_near_duplicates(records, text_fields=["text"], threshold=0.99), [])
        self.assertEqual(len(find_near_duplicates(records, text_fields=["text"], threshold=0.4)), 1)


class TestMissingFields(unittest.TestCase):
    def test_finds_missing_key(self):
        records = [{"prompt": "hi"}, {"prompt": "hi", "response": "hello"}]
        issues = find_missing_fields(records, ["prompt", "response"])
        self.assertEqual(issues, [(0, ["response"])])

    def test_finds_empty_value(self):
        records = [{"prompt": "hi", "response": ""}]
        issues = find_missing_fields(records, ["prompt", "response"])
        self.assertEqual(issues, [(0, ["response"])])

    def test_no_issues_when_all_present(self):
        records = [{"prompt": "hi", "response": "hello"}]
        self.assertEqual(find_missing_fields(records, ["prompt", "response"]), [])


class TestLengthOutliers(unittest.TestCase):
    def test_flags_much_longer_record(self):
        records = [{"text": "short"}] * 5 + [{"text": "x" * 500}]
        outliers = find_length_outliers(records, text_fields=["text"], num_std=1.0)
        indices = [index for index, *_ in outliers]
        self.assertIn(5, indices)

    def test_uniform_lengths_have_no_outliers(self):
        records = [{"text": "abcde"} for _ in range(5)]
        self.assertEqual(find_length_outliers(records, text_fields=["text"]), [])

    def test_fewer_than_two_records_returns_empty(self):
        self.assertEqual(find_length_outliers([{"text": "abc"}], text_fields=["text"]), [])


class TestLabelDistribution(unittest.TestCase):
    def test_counts_by_label(self):
        records = [{"label": "a"}, {"label": "a"}, {"label": "b"}]
        self.assertEqual(label_distribution(records, "label"), {"a": 2, "b": 1})

    def test_missing_label_field_counted_separately(self):
        records = [{"label": "a"}, {}]
        self.assertEqual(label_distribution(records, "label"), {"a": 1, "<missing>": 1})


class TestBuildReport(unittest.TestCase):
    def test_report_contains_all_sections(self):
        records = [
            {"prompt": "a", "response": "b", "label": "x"},
            {"prompt": "a", "response": "b", "label": "x"},
        ]
        report = build_report(records, required_fields=["prompt", "response"], label_field="label")
        self.assertEqual(report["record_count"], 2)
        self.assertIn("exact_duplicates", report)
        self.assertIn("missing_fields", report)
        self.assertIn("label_distribution", report)


if __name__ == "__main__":
    unittest.main()

