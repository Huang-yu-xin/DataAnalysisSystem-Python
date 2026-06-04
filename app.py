from flask import Flask, render_template, request, send_file
from pathlib import Path
from uuid import uuid4

from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from modules.io_utils import read_data_file
from modules.dataset_info import get_dataset_info
from modules.clean import clean_data
from modules.analysis import (
    calculate_elbow_data,
    get_analysis_summary,
    run_kmeans_analysis,
    run_pca_for_clusters,
)
from modules.visualize import (
    create_custom_chart,
    create_default_charts,
    create_elbow_chart,
    create_missing_value_bar,
    create_pca_cluster_scatter,
)


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
DATA_DIR = BASE_DIR / "data"
CHART_DIR = BASE_DIR / "static" / "charts"

UPLOAD_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

# 当前系统主要面向课程本地演示，使用全局变量暂存当前数据。
current_df = None
current_filename = ""
clean_report = None
analysis_report = None


def make_table(df, rows=20):
    """生成网页预览表格，只显示前 rows 行，避免页面太长。"""
    if df is None or df.empty:
        return "<p>暂无数据</p>"

    preview = df.head(rows).copy()

    for col in preview.columns:
        if preview[col].dtype == "object":
            preview[col] = preview[col].astype(str).str.slice(0, 120)

    return preview.to_html(
        classes="table table-bordered table-hover table-sm align-middle",
        index=False
    )


def _render_no_data(message="请先上传数据文件"):
    return render_template("index.html", error=message)


def _render_preview(message=None):
    return render_template(
        "preview.html",
        filename=current_filename,
        rows=current_df.shape[0],
        cols=current_df.shape[1],
        columns=list(current_df.columns),
        tables=[make_table(current_df, 20)],
        message=message
    )


def _export_status():
    cleaned_path = DATA_DIR / "cleaned_data.csv"
    analyzed_path = DATA_DIR / "analyzed_data.csv"
    return {
        "cleaned_exists": cleaned_path.exists(),
        "analyzed_exists": analyzed_path.exists(),
    }


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(error):
    return render_template(
        "index.html",
        error="上传文件过大，请上传不超过 10MB 的 CSV 或 Excel 文件。"
    ), 413


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    global current_df, current_filename, clean_report, analysis_report

    file = request.files.get("file")
    if file is None:
        return render_template("index.html", error="未上传文件，请选择 CSV、XLSX 或 XLS 文件。")

    raw_filename = (file.filename or "").strip()
    if not raw_filename:
        return render_template("index.html", error="文件名为空，请重新选择有效文件。")

    suffix = Path(raw_filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return render_template("index.html", error="暂不支持该文件类型，请上传 CSV、XLSX 或 XLS 文件。")

    safe_filename = secure_filename(raw_filename)
    safe_stem = secure_filename(Path(raw_filename).stem) or "uploaded_data"
    if not safe_filename and not safe_stem:
        return render_template("index.html", error="文件名为空，请重新选择有效文件。")

    filename = f"{uuid4().hex[:10]}_{safe_stem}{suffix}"
    file_path = UPLOAD_DIR / filename

    try:
        file.save(file_path)
        df = read_data_file(file_path)
    except Exception as exc:
        return render_template("index.html", error=f"文件读取失败：{exc}")

    if df is None or df.empty or len(df.columns) == 0:
        return render_template("index.html", error="文件为空或没有可读取的数据，请重新上传。")

    current_df = df
    current_filename = raw_filename
    clean_report = None
    analysis_report = None

    return _render_preview(message="文件上传成功，已生成数据预览。")


@app.route("/preview")
def preview_page():
    if current_df is None:
        return _render_no_data()

    return _render_preview()


@app.route("/dataset_info")
def dataset_info_page():
    if current_df is None:
        return _render_no_data()

    info = get_dataset_info(current_df, current_filename)
    return render_template("dataset_info.html", info=info)


@app.route("/clean", methods=["GET", "POST"])
def clean_page():
    global current_df, clean_report

    if current_df is None:
        return _render_no_data()

    message = None
    error = None

    if request.method == "POST":
        numeric_strategy = request.form.get("numeric_strategy", "median")
        category_strategy = request.form.get("category_strategy", "mode")
        outlier_method = request.form.get("outlier_method", "iqr")
        outlier_action = request.form.get("outlier_action", "cap")
    else:
        numeric_strategy = "median"
        category_strategy = "mode"
        outlier_method = "iqr"
        outlier_action = "cap"

    if request.method == "POST":
        before_df = current_df.copy()
        try:
            current_df, clean_report = clean_data(current_df, request.form)
            clean_report["missing_chart_before"] = create_missing_value_bar(
                before_df,
                "清洗前各字段缺失值数量统计"
            )
            clean_report["missing_chart_after"] = create_missing_value_bar(
                current_df,
                "清洗后各字段缺失值数量统计"
            )

            cleaned_path = DATA_DIR / "cleaned_data.csv"
            current_df.to_csv(cleaned_path, index=False, encoding="utf-8-sig")
            message = "数据清洗完成，清洗结果已保存。"
        except Exception as exc:
            error = f"数据清洗失败：{exc}"

    return render_template(
        "clean.html",
        rows=current_df.shape[0],
        cols=current_df.shape[1],
        columns=list(current_df.columns),
        report=clean_report,
        current_missing_chart=create_missing_value_bar(current_df),
        tables=[make_table(current_df, 20)],
        message=message,
        error=error,
        numeric_strategy=numeric_strategy,
        category_strategy=category_strategy,
        outlier_method=outlier_method,
        outlier_action=outlier_action
    )


@app.route("/analysis", methods=["GET", "POST"])
def analysis_page():
    global current_df, analysis_report

    if current_df is None:
        return _render_no_data()

    numeric_cols = current_df.select_dtypes(include=["number"]).columns.tolist()
    message = None
    error = None
    cluster_count_selected = "3"

    if request.method == "POST":
        selected_cols = request.form.getlist("feature_cols")
        cluster_count_selected = request.form.get("cluster_count", "3")

        try:
            k = int(cluster_count_selected)
        except (TypeError, ValueError):
            k = 3
            cluster_count_selected = "3"
            error = "K 值不合法，已使用默认值 3。"

        try:
            current_df, analysis_report = run_kmeans_analysis(current_df, selected_cols, k)
            feature_cols = analysis_report.get("cluster_features", selected_cols)

            if feature_cols:
                elbow_data = calculate_elbow_data(current_df, feature_cols)
                analysis_report["elbow_message"] = elbow_data.get("message")
                analysis_report["elbow_chart"] = create_elbow_chart(
                    elbow_data.get("k_values", []),
                    elbow_data.get("inertias", [])
                )

                pca_df = run_pca_for_clusters(current_df, feature_cols)
                analysis_report["pca_chart"] = create_pca_cluster_scatter(pca_df)

            analyzed_path = DATA_DIR / "analyzed_data.csv"
            current_df.to_csv(analyzed_path, index=False, encoding="utf-8-sig")
            message = "数据分析完成，分析结果已保存。"
        except Exception as exc:
            error = f"数据分析失败：{exc}"

    if analysis_report is None:
        analysis_report = get_analysis_summary(current_df)

    return render_template(
        "analysis.html",
        rows=current_df.shape[0],
        cols=current_df.shape[1],
        numeric_cols=numeric_cols,
        report=analysis_report,
        tables=[make_table(current_df, 20)],
        message=message,
        error=error,
        cluster_count_selected=cluster_count_selected
    )


@app.route("/visual", methods=["GET", "POST"])
def visual_page():
    if current_df is None:
        return _render_no_data()

    numeric_cols = current_df.select_dtypes(include=["number"]).columns.tolist()
    category_cols = current_df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    all_cols = list(current_df.columns)

    custom_chart = None
    custom_message = None
    custom_error = None

    if request.method == "POST":
        chart_type = request.form.get("chart_type", "scatter")
        x_col = request.form.get("x_col", "")
        y_col = request.form.get("y_col", "")
    else:
        chart_type = "scatter"
        x_col = all_cols[0] if all_cols else ""
        y_col = ""

    if request.method == "POST":
        try:
            custom_chart = create_custom_chart(current_df, chart_type, x_col, y_col)
            custom_message = "自定义图表生成成功。"
        except Exception as exc:
            custom_error = f"自定义图表生成失败：{exc}"

    default_charts = create_default_charts(current_df)

    return render_template(
        "visual.html",
        all_cols=all_cols,
        numeric_cols=numeric_cols,
        category_cols=category_cols,
        default_charts=default_charts,
        custom_chart=custom_chart,
        custom_message=custom_message,
        custom_error=custom_error,
        chart_type_selected=chart_type,
        x_col_selected=x_col,
        y_col_selected=y_col
    )


@app.route("/exports")
def exports_page():
    status = _export_status()
    return render_template("exports.html", **status)


@app.route("/download_cleaned")
def download_cleaned():
    path = DATA_DIR / "cleaned_data.csv"
    if not path.exists():
        status = _export_status()
        return render_template("exports.html", error="还没有清洗后的数据，请先执行数据清洗。", **status)
    return send_file(path, as_attachment=True)


@app.route("/download_analyzed")
def download_analyzed():
    path = DATA_DIR / "analyzed_data.csv"
    if not path.exists():
        status = _export_status()
        return render_template("exports.html", error="还没有分析后的数据，请先执行数据分析。", **status)
    return send_file(path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
