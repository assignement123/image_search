import csv
import os
import tempfile
import unittest

from leave_data_analysis import analyze_leave_data, load_leave_data


class LeaveDataAnalysisTests(unittest.TestCase):
    def test_analyze_reports_expected_issues(self):
        rows = [
            {"leaf_id": "1", "area": "10", "species": "A"},
            {"leaf_id": "2", "area": "", "species": "A"},
            {"leaf_id": "2", "area": "", "species": "A"},
        ]
        columns = ["leaf_id", "area", "species"]

        report = analyze_leave_data(rows, columns)

        self.assertFalse(report["is_ready_for_attribute_analysis"])
        self.assertEqual(report["summary"]["duplicate_rows"], 1)
        self.assertEqual(report["column_reports"]["species"]["unique_count"], 1)
        self.assertGreater(report["column_reports"]["area"]["missing_ratio"], 0.2)

    def test_load_leave_data_from_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "leave_data.csv")
            with open(path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["leaf_id", "length"])
                writer.writeheader()
                writer.writerow({"leaf_id": "1", "length": "2.5"})

            rows, columns, resolved_path = load_leave_data(path)

            self.assertEqual(resolved_path, path)
            self.assertEqual(columns, ["leaf_id", "length"])
            self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
