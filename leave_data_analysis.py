import argparse
import csv
import json
import os
from collections import Counter
from statistics import mean


MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none"}
DEFAULT_DATASET_CANDIDATES = (
    "data/leave_data.csv",
    "leave_data.csv",
    "data/leave_data.json",
    "leave_data.json",
    "data/leaf_data.csv",
    "leaf_data.csv",
    "data/leaf_data.json",
    "leaf_data.json",
)


def _is_missing(value):
    if value is None:
        return True
    return str(value).strip().lower() in MISSING_TOKENS


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _infer_type(values):
    non_missing = [v for v in values if not _is_missing(v)]
    if not non_missing:
        return "empty"
    numeric_count = sum(_to_float(v) is not None for v in non_missing)
    if numeric_count == len(non_missing):
        return "numeric"
    return "categorical"


def _resolve_dataset_path(path=None):
    if path:
        return path
    for candidate in DEFAULT_DATASET_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return None


def load_leave_data(path=None):
    dataset_path = _resolve_dataset_path(path)
    if dataset_path is None:
        raise FileNotFoundError(
            "Không tìm thấy dataset leave_data/leaf_data. Hãy truyền --path hoặc đặt file ở data/leave_data.csv."
        )

    ext = os.path.splitext(dataset_path)[1].lower()
    if ext == ".csv":
        with open(dataset_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            columns = list(reader.fieldnames or [])
            return rows, columns, dataset_path

    if ext == ".json":
        with open(dataset_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            if "leave_data" in payload:
                payload = payload["leave_data"]
            elif "leaf_data" in payload:
                payload = payload["leaf_data"]
        if not isinstance(payload, list):
            raise ValueError("JSON dataset phải là list các object hoặc chứa key leave_data.")
        rows = [row for row in payload if isinstance(row, dict)]
        columns = sorted({k for row in rows for k in row.keys()})
        return rows, columns, dataset_path

    raise ValueError(f"Định dạng file chưa hỗ trợ: {ext}")


def analyze_leave_data(rows, columns):
    total_rows = len(rows)
    total_columns = len(columns)
    column_reports = {}
    constant_columns = []
    high_missing_columns = []

    for col in columns:
        values = [row.get(col) for row in rows]
        missing_count = sum(_is_missing(v) for v in values)
        non_missing_values = [v for v in values if not _is_missing(v)]
        unique_values = {str(v).strip() for v in non_missing_values}
        inferred_type = _infer_type(values)

        report = {
            "inferred_type": inferred_type,
            "missing_count": missing_count,
            "missing_ratio": (missing_count / total_rows) if total_rows else 0.0,
            "unique_count": len(unique_values),
            "sample_values": list(unique_values)[:5],
        }

        if inferred_type == "numeric":
            numeric_values = [_to_float(v) for v in non_missing_values]
            numeric_values = [v for v in numeric_values if v is not None]
            if numeric_values:
                report["stats"] = {
                    "min": min(numeric_values),
                    "max": max(numeric_values),
                    "mean": mean(numeric_values),
                }

        if report["unique_count"] <= 1 and total_rows > 0:
            constant_columns.append(col)
        if report["missing_ratio"] > 0.2:
            high_missing_columns.append(col)

        column_reports[col] = report

    seen = Counter(tuple(str(row.get(c, "")).strip() for c in columns) for row in rows)
    duplicate_rows = sum(count - 1 for count in seen.values() if count > 1)
    duplicate_ratio = (duplicate_rows / total_rows) if total_rows else 0.0

    issues = []
    if total_rows == 0:
        issues.append("Dataset rỗng, chưa thể phân tích thuộc tính.")
    if total_columns == 0:
        issues.append("Dataset chưa có cột dữ liệu.")
    if constant_columns:
        issues.append(f"Cột gần như không có biến thiên: {', '.join(constant_columns)}.")
    if high_missing_columns:
        issues.append(f"Cột có tỷ lệ thiếu >20%: {', '.join(high_missing_columns)}.")
    if duplicate_ratio > 0.1:
        issues.append(f"Tỷ lệ bản ghi trùng lặp cao ({duplicate_ratio:.1%}).")

    recommendations = []
    if total_rows < 30:
        recommendations.append("Nên tăng số lượng bản ghi (>=30) để phân tích thuộc tính ổn định hơn.")
    if high_missing_columns:
        recommendations.append("Xử lý giá trị thiếu (impute/loại bỏ cột) trước khi phân tích.")
    if constant_columns:
        recommendations.append("Cân nhắc loại bỏ các cột không biến thiên vì ít giá trị phân tích.")
    if duplicate_rows:
        recommendations.append("Loại bỏ bản ghi trùng lặp để tránh sai lệch thống kê.")
    if not recommendations:
        recommendations.append("Dataset đang ở trạng thái tốt để phân tích thuộc tính.")

    has_rows = total_rows > 0
    has_columns = total_columns > 0
    has_quality_issues = bool(high_missing_columns or constant_columns or duplicate_ratio > 0.1)
    is_ready = has_rows and has_columns and not has_quality_issues

    return {
        "is_ready_for_attribute_analysis": is_ready,
        "summary": {
            "total_rows": total_rows,
            "total_columns": total_columns,
            "duplicate_rows": duplicate_rows,
            "duplicate_ratio": duplicate_ratio,
        },
        "column_reports": column_reports,
        "issues": issues,
        "recommendations": recommendations,
    }


def build_text_report(analysis, dataset_path):
    status = "ĐẠT" if analysis["is_ready_for_attribute_analysis"] else "CHƯA ĐẠT"
    summary = analysis["summary"]
    lines = [
        f"Dataset: {dataset_path}",
        f"Kết luận mức sẵn sàng phân tích thuộc tính: {status}",
        f"- Số dòng: {summary['total_rows']}",
        f"- Số cột: {summary['total_columns']}",
        f"- Dòng trùng lặp: {summary['duplicate_rows']} ({summary['duplicate_ratio']:.1%})",
        "",
        "Các vấn đề:",
    ]
    if analysis["issues"]:
        lines.extend(f"- {issue}" for issue in analysis["issues"])
    else:
        lines.append("- Không phát hiện vấn đề nghiêm trọng.")

    lines.append("")
    lines.append("Khuyến nghị:")
    lines.extend(f"- {item}" for item in analysis["recommendations"])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Đánh giá dataset leave_data/leaf_data cho phân tích thuộc tính.")
    parser.add_argument("--path", type=str, default=None, help="Đường dẫn file dataset leave_data/leaf_data (.csv hoặc .json)")
    parser.add_argument("--json", action="store_true", help="In báo cáo dưới dạng JSON")
    args = parser.parse_args()

    rows, columns, dataset_path = load_leave_data(args.path)
    analysis = analyze_leave_data(rows, columns)

    if args.json:
        print(json.dumps({"dataset_path": dataset_path, **analysis}, ensure_ascii=False, indent=2))
        return

    print(build_text_report(analysis, dataset_path))


if __name__ == "__main__":
    main()
