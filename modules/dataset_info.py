import pandas as pd


WEATHER_FIELD_DESCRIPTIONS = {
    "MinTemp": "当日最低温度",
    "MaxTemp": "当日最高温度",
    "Rainfall": "降雨量",
    "Evaporation": "蒸发量",
    "Sunshine": "日照时长",
    "WindGustDir": "最大阵风风向",
    "WindGustSpeed": "最大阵风风速",
    "Humidity9am": "上午 9 点湿度",
    "Humidity3pm": "下午 3 点湿度",
    "Pressure9am": "上午 9 点气压",
    "Pressure3pm": "下午 3 点气压",
    "Temp9am": "上午 9 点温度",
    "Temp3pm": "下午 3 点温度",
    "RainToday": "今天是否下雨",
    "RainTomorrow": "明天是否下雨",
}


def _is_date_like(series, column_name):
    """根据字段名、数据类型和可解析比例判断日期字段。"""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    name = str(column_name).lower()
    name_has_date_hint = any(key in name for key in ["date", "time", "日期", "时间"])
    if not name_has_date_hint:
        return False

    non_null = series.dropna()

    if non_null.empty:
        return name_has_date_hint

    sample = non_null.astype(str).head(200)
    parsed = pd.to_datetime(sample, errors="coerce")
    valid_ratio = parsed.notna().sum() / len(sample)

    return name_has_date_hint and valid_ratio >= 0.6


def _sample_values(series, limit=3):
    values = series.dropna().astype(str).head(limit).tolist()
    return "、".join(values) if values else "无"


def get_dataset_info(df, filename=""):
    """
    生成当前数据集说明信息。
    天气数据会额外返回字段解释，普通数据返回通用字段明细。
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    date_cols = [col for col in df.columns if _is_date_like(df[col], col)]
    category_cols = [
        col for col in df.columns
        if col not in numeric_cols and col not in date_cols
    ]

    weather_fields = [
        {"field": col, "description": WEATHER_FIELD_DESCRIPTIONS[col]}
        for col in WEATHER_FIELD_DESCRIPTIONS
        if col in df.columns
    ]

    field_details = []
    rows = max(int(df.shape[0]), 1)
    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        field_details.append({
            "name": col,
            "dtype": str(df[col].dtype),
            "non_null": int(df[col].notna().sum()),
            "missing_count": missing_count,
            "missing_rate": round(missing_count / rows * 100, 2),
            "sample_values": _sample_values(df[col]),
        })

    return {
        "filename": filename or "未命名数据集",
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": list(df.columns),
        "numeric_count": len(numeric_cols),
        "category_count": len(category_cols),
        "date_count": len(date_cols),
        "date_cols": date_cols,
        "missing_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "is_weather_data": len(weather_fields) >= 4,
        "weather_fields": weather_fields,
        "field_details": field_details,
    }
