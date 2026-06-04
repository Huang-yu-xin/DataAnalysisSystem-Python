import pandas as pd
import numpy as np


def _is_checked(value):
    return str(value).lower() in ["on", "true", "1", "yes"]


def _missing_count(df):
    result = {}
    for col in df.columns:
        count = int(df[col].isna().sum())
        if count > 0:
            result[col] = count
    return result


def _total_missing(df):
    return int(df.isna().sum().sum())


def get_cleaning_comparison(before_df, after_df):
    """
    返回清洗前后对比信息，便于页面展示清洗效果。
    """
    before_rows, before_cols = before_df.shape
    after_rows, after_cols = after_df.shape
    before_missing = _missing_count(before_df)
    after_missing = _missing_count(after_df)

    all_columns = list(dict.fromkeys(list(before_df.columns) + list(after_df.columns)))
    missing_by_column = []
    for col in all_columns:
        before_count = int(before_df[col].isna().sum()) if col in before_df.columns else 0
        after_count = int(after_df[col].isna().sum()) if col in after_df.columns else 0
        if before_count > 0 or after_count > 0:
            missing_by_column.append({
                "col": col,
                "before": before_count,
                "after": after_count,
                "reduced": before_count - after_count,
            })

    return {
        "before_rows": int(before_rows),
        "after_rows": int(after_rows),
        "removed_rows": int(max(before_rows - after_rows, 0)),
        "before_cols": int(before_cols),
        "after_cols": int(after_cols),
        "removed_cols": int(max(before_cols - after_cols, 0)),
        "added_cols": int(max(after_cols - before_cols, 0)),
        "missing_before_total": _total_missing(before_df),
        "missing_after_total": _total_missing(after_df),
        "missing_reduced": _total_missing(before_df) - _total_missing(after_df),
        "duplicates_before": int(before_df.duplicated().sum()),
        "duplicates_after": int(after_df.duplicated().sum()),
        "duplicates_reduced": int(max(before_df.duplicated().sum() - after_df.duplicated().sum(), 0)),
        "missing_before": before_missing,
        "missing_after": after_missing,
        "missing_by_column": missing_by_column,
    }


def _clean_text_columns(df):
    """清理文本列：去空格，把空字符串、NULL 等统一当作缺失值。"""
    object_cols = df.select_dtypes(include=["object", "string"]).columns

    for col in object_cols:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace({
            "": pd.NA,
            "nan": pd.NA,
            "NaN": pd.NA,
            "None": pd.NA,
            "NULL": pd.NA,
            "null": pd.NA,
            "--": pd.NA,
            "?": pd.NA
        })

    return df


def _convert_datetime_columns(df, report):
    """自动识别日期列并转换为 datetime 类型。"""
    for col in df.columns:
        col_lower = col.lower()

        if "date" in col_lower or "time" in col_lower or "日期" in col_lower:
            before_missing = df[col].isna().sum()
            converted = pd.to_datetime(df[col], errors="coerce")

            valid_old = df[col].notna().sum()
            valid_new = converted.notna().sum()

            if valid_old == 0:
                continue

            # 至少 70% 可以成功转成日期，才认为这是日期列
            if valid_new / valid_old >= 0.7:
                df[col] = converted
                after_missing = df[col].isna().sum()
                report["type_converted"].append(
                    f"{col} 转换为日期类型，新增缺失值 {int(after_missing - before_missing)} 个"
                )

    return df


def _convert_numeric_columns(df, report):
    """自动识别看起来像数字的字符串列，并转换为数值类型。"""
    object_cols = df.select_dtypes(include=["object", "string"]).columns

    for col in object_cols:
        non_null = df[col].dropna()

        if len(non_null) == 0:
            continue

        converted = pd.to_numeric(non_null, errors="coerce")
        ratio = converted.notna().sum() / len(non_null)

        if ratio >= 0.85:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            report["type_converted"].append(f"{col} 自动转换为数值类型")

    return df


def _handle_missing_values(df, numeric_strategy, category_strategy, report):
    """
    缺失值处理。
    数值列：均值 / 中位数 / 0 / 删除 / 不处理
    类别列：众数 / Unknown / 删除 / 不处理
    """
    report["missing_before"] = _missing_count(df)

    numeric_cols = df.select_dtypes(include=["number"]).columns
    category_cols = df.select_dtypes(include=["object", "string", "category"]).columns

    for col in numeric_cols:
        missing_num = int(df[col].isna().sum())
        if missing_num == 0:
            continue

        if numeric_strategy == "mean":
            value = df[col].mean()
            df[col] = df[col].fillna(value)
            report["missing_actions"].append(f"{col}：{missing_num} 个缺失值，用均值 {value:.2f} 填充")

        elif numeric_strategy == "median":
            value = df[col].median()
            df[col] = df[col].fillna(value)
            report["missing_actions"].append(f"{col}：{missing_num} 个缺失值，用中位数 {value:.2f} 填充")

        elif numeric_strategy == "zero":
            df[col] = df[col].fillna(0)
            report["missing_actions"].append(f"{col}：{missing_num} 个缺失值，用 0 填充")

        elif numeric_strategy == "drop":
            before = len(df)
            df = df.dropna(subset=[col])
            after = len(df)
            report["missing_actions"].append(f"{col}：删除缺失值所在行 {before - after} 行")

        elif numeric_strategy == "none":
            report["missing_actions"].append(f"{col}：检测到 {missing_num} 个缺失值，未处理")

    for col in category_cols:
        missing_num = int(df[col].isna().sum())
        if missing_num == 0:
            continue

        if category_strategy == "mode":
            mode_value = df[col].mode(dropna=True)

            if len(mode_value) > 0:
                value = mode_value.iloc[0]
                df[col] = df[col].fillna(value)
                report["missing_actions"].append(f"{col}：{missing_num} 个缺失值，用众数 {value} 填充")
            else:
                report["missing_actions"].append(f"{col}：无法计算众数，未处理")

        elif category_strategy == "unknown":
            df[col] = df[col].fillna("Unknown")
            report["missing_actions"].append(f"{col}：{missing_num} 个缺失值，用 Unknown 填充")

        elif category_strategy == "drop":
            before = len(df)
            df = df.dropna(subset=[col])
            after = len(df)
            report["missing_actions"].append(f"{col}：删除缺失值所在行 {before - after} 行")

        elif category_strategy == "none":
            report["missing_actions"].append(f"{col}：检测到 {missing_num} 个缺失值，未处理")

    report["missing_after"] = _missing_count(df)

    return df


def _apply_weather_rules(df, report):
    """
    天气数据专用清洗规则：
    1. 湿度限制在 0~100
    2. 云量限制在 0~8
    3. 降雨量、蒸发量、日照、风速不能为负数
    4. RainToday / RainTomorrow 统一成 Yes / No
    5. MinTemp 不能大于 MaxTemp，若出现则交换
    """
    yes_no_cols = ["RainToday", "RainTomorrow"]
    mapping = {
        "yes": "Yes", "y": "Yes", "true": "Yes", "1": "Yes",
        "no": "No", "n": "No", "false": "No", "0": "No"
    }

    for col in yes_no_cols:
        if col in df.columns:
            original = df[col].copy()
            normalized = df[col].astype("string").str.strip().str.lower()
            df[col] = normalized.map(mapping).fillna(original)
            report["domain_rules"].append(f"{col}：统一为 Yes / No 格式")

    wind_dir_cols = ["WindGustDir", "WindDir9am", "WindDir3pm"]
    for col in wind_dir_cols:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip().str.upper()
            report["domain_rules"].append(f"{col}：统一为大写格式")

    for col in df.columns:
        if "Humidity" in col and pd.api.types.is_numeric_dtype(df[col]):
            invalid = int(((df[col] < 0) | (df[col] > 100)).sum())
            if invalid > 0:
                df[col] = df[col].clip(0, 100)
            report["domain_rules"].append(f"{col}：湿度范围限制为 0~100，修正 {invalid} 个异常值")

    for col in df.columns:
        if "Cloud" in col and pd.api.types.is_numeric_dtype(df[col]):
            invalid = int(((df[col] < 0) | (df[col] > 8)).sum())
            if invalid > 0:
                df[col] = df[col].clip(0, 8)
            report["domain_rules"].append(f"{col}：云量范围限制为 0~8，修正 {invalid} 个异常值")

    non_negative_cols = [
        "Rainfall", "Evaporation", "Sunshine",
        "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm"
    ]

    for col in non_negative_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            invalid = int((df[col] < 0).sum())
            if invalid > 0:
                df[col] = df[col].clip(lower=0)
            report["domain_rules"].append(f"{col}：不允许为负数，修正 {invalid} 个异常值")

    if "MinTemp" in df.columns and "MaxTemp" in df.columns:
        mask = df["MinTemp"] > df["MaxTemp"]
        count = int(mask.sum())

        if count > 0:
            old_min = df.loc[mask, "MinTemp"].copy()
            df.loc[mask, "MinTemp"] = df.loc[mask, "MaxTemp"]
            df.loc[mask, "MaxTemp"] = old_min

        report["domain_rules"].append(f"MinTemp <= MaxTemp：修正 {count} 行")

    return df


def _detect_and_handle_outliers(df, method, action, outlier_columns, report):
    """
    异常值检测与处理。
    method:
        iqr: 四分位距法
        zscore: 标准分法
        none: 不处理
    action:
        cap: 盖帽处理
        remove: 删除异常行
        flag: 只标记
        none: 只检测不处理
    """
    if method == "none":
        return df

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if outlier_columns:
        selected = [col.strip() for col in outlier_columns.split(",") if col.strip()]
        numeric_cols = [col for col in selected if col in numeric_cols]

    for col in numeric_cols:
        series = df[col].dropna()

        if len(series) < 4:
            continue

        if method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            if iqr == 0:
                continue

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

        elif method == "zscore":
            mean = series.mean()
            std = series.std()

            if std == 0:
                continue

            lower = mean - 3 * std
            upper = mean + 3 * std
        else:
            continue

        mask = (df[col] < lower) | (df[col] > upper)
        count = int(mask.sum())

        if count == 0:
            continue

        report["outliers"][col] = {
            "method": method,
            "lower": round(float(lower), 3),
            "upper": round(float(upper), 3),
            "count": count,
            "action": action
        }

        if action == "cap":
            df[col] = df[col].clip(lower, upper)
            report["outlier_actions"].append(
                f"{col}：检测到 {count} 个异常值，已盖帽到 [{lower:.2f}, {upper:.2f}]"
            )

        elif action == "remove":
            before = len(df)
            df = df[~mask].copy()
            after = len(df)
            report["outlier_actions"].append(
                f"{col}：检测到 {count} 个异常值，删除 {before - after} 行"
            )

        elif action == "flag":
            flag_col = col + "_outlier"
            df[flag_col] = mask
            report["outlier_actions"].append(
                f"{col}：检测到 {count} 个异常值，已新增标记列 {flag_col}"
            )

        elif action == "none":
            report["outlier_actions"].append(
                f"{col}：检测到 {count} 个异常值，仅检测未处理"
            )

    return df


def clean_data(df, form=None):
    """主清洗函数，返回清洗后的 DataFrame 和清洗报告。"""
    report = {
        "original_rows": int(df.shape[0]),
        "original_cols": int(df.shape[1]),
        "final_rows": 0,
        "final_cols": 0,
        "removed_empty_rows": 0,
        "removed_empty_cols": 0,
        "duplicates_before": 0,
        "duplicates_removed": 0,
        "missing_before": {},
        "missing_after": {},
        "missing_actions": [],
        "type_converted": [],
        "domain_rules": [],
        "outliers": {},
        "outlier_actions": [],
        "duplicates_after": 0,
        "comparison": None
    }

    if form is None:
        numeric_strategy = "median"
        category_strategy = "mode"
        outlier_method = "iqr"
        outlier_action = "cap"
        outlier_columns = ""
        drop_duplicate = True
        use_weather_rules = True
    else:
        numeric_strategy = form.get("numeric_strategy", "median")
        category_strategy = form.get("category_strategy", "mode")
        outlier_method = form.get("outlier_method", "iqr")
        outlier_action = form.get("outlier_action", "cap")
        outlier_columns = form.get("outlier_columns", "")
        drop_duplicate = _is_checked(form.get("drop_duplicate"))
        use_weather_rules = _is_checked(form.get("use_weather_rules"))

    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    before_df = df.copy()

    before_rows = len(df)
    df = df.dropna(how="all")
    report["removed_empty_rows"] = before_rows - len(df)

    before_cols = df.shape[1]
    df = df.dropna(axis=1, how="all")
    report["removed_empty_cols"] = before_cols - df.shape[1]

    df = _clean_text_columns(df)
    df = _convert_datetime_columns(df, report)
    df = _convert_numeric_columns(df, report)

    duplicate_count = int(df.duplicated().sum())
    report["duplicates_before"] = duplicate_count

    if drop_duplicate:
        df = df.drop_duplicates()
        report["duplicates_removed"] = duplicate_count

    df = _handle_missing_values(df, numeric_strategy, category_strategy, report)

    if use_weather_rules:
        df = _apply_weather_rules(df, report)

    df = _detect_and_handle_outliers(
        df,
        method=outlier_method,
        action=outlier_action,
        outlier_columns=outlier_columns,
        report=report
    )

    if "Date" in df.columns:
        try:
            df = df.sort_values(by="Date")
        except Exception:
            pass

    report["final_rows"] = int(df.shape[0])
    report["final_cols"] = int(df.shape[1])
    report["duplicates_after"] = int(df.duplicated().sum())
    report["comparison"] = get_cleaning_comparison(before_df, df)
    report["comparison"]["outlier_handled_total"] = int(
        sum(info.get("count", 0) for info in report["outliers"].values())
    )
    report["comparison"]["type_converted_count"] = len(report["type_converted"])

    return df, report
