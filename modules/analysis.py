import pandas as pd
import numpy as np
import os


os.environ.setdefault("OMP_NUM_THREADS", "1")


def get_analysis_summary(df):
    """基础统计分析，用于页面展示。"""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    summary = {
        "numeric_cols": numeric_cols,
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "basic": []
    }

    for col in numeric_cols[:10]:
        series = df[col].dropna()
        if series.empty:
            continue

        summary["basic"].append({
            "col": col,
            "mean": round(float(series.mean()), 3),
            "median": round(float(series.median()), 3),
            "min": round(float(series.min()), 3),
            "max": round(float(series.max()), 3),
            "std": round(float(series.std()), 3)
        })

    return summary


def _default_feature_cols(df, feature_cols):
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if feature_cols:
        return [col for col in feature_cols if col in numeric_cols]

    preferred = [
        "MinTemp", "MaxTemp", "Rainfall", "WindGustSpeed",
        "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm",
        "Temp9am", "Temp3pm"
    ]
    selected = [col for col in preferred if col in numeric_cols][:5]

    if len(selected) < 2:
        selected = numeric_cols[:min(5, len(numeric_cols))]

    return selected


def _prepare_feature_matrix(df, feature_cols):
    feature_cols = _default_feature_cols(df, feature_cols)
    if len(feature_cols) < 2:
        return None, feature_cols, "数值字段不足 2 个，无法进行 KMeans 聚类。"

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    valid_cols = [col for col in feature_cols if not X[col].isna().all()]
    X = X[valid_cols]

    if len(valid_cols) < 2:
        return None, valid_cols, "可用于聚类的有效数值字段不足 2 个。"

    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    if len(X) < 2:
        return None, valid_cols, "样本数量太少，无法进行 KMeans 聚类。"

    std = X.std(ddof=0).replace(0, 1)
    X_scaled = (X - X.mean()) / std

    return X_scaled, valid_cols, None


def _fallback_cluster(df, feature_cols, k):
    """
    如果用户没装 sklearn，使用简单分位数方式模拟聚类，保证项目仍能运行。
    """
    score = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True)).sum(axis=1)

    try:
        labels = pd.qcut(score, q=k, labels=False, duplicates="drop")
        labels = labels.fillna(0).astype(int)
    except Exception:
        labels = pd.Series(0, index=df.index)

    return labels


def run_kmeans_analysis(df, feature_cols, k=3):
    """
    KMeans 聚类分析，返回带 cluster 字段的数据和分析报告。
    """
    df = df.copy()
    report = get_analysis_summary(df)

    try:
        requested_k = int(k)
    except (TypeError, ValueError):
        requested_k = 3

    X_scaled, feature_cols, matrix_message = _prepare_feature_matrix(df, feature_cols)
    if matrix_message:
        report["cluster_message"] = matrix_message
        return df, report

    if requested_k < 2:
        report["cluster_message"] = "K 值必须大于或等于 2。"
        return df, report

    k = min(requested_k, 8)
    k_message = ""
    sample_count = len(X_scaled)

    if requested_k > 8:
        k_message = "K 值已按页面规则限制为 8。"

    if k > sample_count:
        k = sample_count
        k_message = f"选择的 K 值大于样本数量，已自动调整为 {k}。"

    if k < 2:
        report["cluster_message"] = "样本数量太少，无法形成至少 2 个聚类。"
        return df, report

    inertia = None
    silhouette = None
    silhouette_message = "当前结果不满足轮廓系数计算条件。"

    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        inertia = round(float(model.inertia_), 3)
        method = "KMeans"

        unique_labels = set(labels)
        if 2 <= len(unique_labels) < sample_count:
            silhouette = round(float(silhouette_score(X_scaled, labels)), 4)
            silhouette_message = (
                "轮廓系数越接近 1，说明样本在本类内越紧密、类间区分越明显；"
                "接近 0 表示聚类边界不明显；小于 0 表示聚类可能不合理。"
            )
        elif len(unique_labels) < 2:
            silhouette_message = "所有样本被分到同一类，无法计算轮廓系数。"
        else:
            silhouette_message = "样本数量与聚类数量过于接近，无法计算有效轮廓系数。"
    except Exception as exc:
        labels = _fallback_cluster(df, feature_cols, k)
        method = "Fallback 分位数聚类"
        silhouette_message = f"未能调用 sklearn 计算轮廓系数：{exc}"

    df["cluster"] = labels
    cluster_counts = df["cluster"].value_counts().sort_index().to_dict()
    cluster_profile = (
        df.groupby("cluster")[feature_cols]
        .mean(numeric_only=True)
        .round(3)
        .reset_index()
        .to_dict(orient="records")
    )

    report.update({
        "cluster_method": method,
        "cluster_k": int(k),
        "cluster_features": feature_cols,
        "cluster_counts": {int(key): int(value) for key, value in cluster_counts.items()},
        "cluster_profile": cluster_profile,
        "cluster_message": "聚类分析完成，已在数据表中新增 cluster 字段。",
        "cluster_k_message": k_message,
        "cluster_inertia": inertia,
        "silhouette_score": silhouette,
        "silhouette_message": silhouette_message,
    })

    return df, report


def calculate_elbow_data(df, feature_cols, k_range=range(2, 9)):
    """
    计算不同 K 值下的 KMeans inertia，用于绘制肘部法图。
    """
    X_scaled, feature_cols, matrix_message = _prepare_feature_matrix(df, feature_cols)
    if matrix_message:
        return {"k_values": [], "inertias": [], "message": matrix_message}

    try:
        from sklearn.cluster import KMeans
    except Exception as exc:
        return {"k_values": [], "inertias": [], "message": f"未能调用 sklearn 计算肘部法数据：{exc}"}

    k_values = []
    inertias = []
    sample_count = len(X_scaled)

    for k in k_range:
        if k < 2 or k > sample_count:
            continue

        model = KMeans(n_clusters=int(k), random_state=42, n_init=10)
        model.fit(X_scaled)
        k_values.append(int(k))
        inertias.append(round(float(model.inertia_), 3))

    if not k_values:
        return {"k_values": [], "inertias": [], "message": "样本数量不足，无法生成肘部法图。"}

    return {
        "k_values": k_values,
        "inertias": inertias,
        "message": "肘部法用于辅助选择合适的聚类数量 K。SSE 下降幅度明显变缓的位置，通常可以作为候选 K 值。"
    }


def run_pca_for_clusters(df, feature_cols, cluster_col="cluster"):
    """
    将参与聚类的高维特征通过 PCA 降到二维。
    返回包含 PC1、PC2、cluster 的 DataFrame。
    """
    if cluster_col not in df.columns:
        return pd.DataFrame()

    X_scaled, feature_cols, matrix_message = _prepare_feature_matrix(df, feature_cols)
    if matrix_message:
        return pd.DataFrame()

    try:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2, random_state=42)
        values = pca.fit_transform(X_scaled)
    except Exception:
        values = X_scaled.iloc[:, :2].to_numpy()

    return pd.DataFrame({
        "PC1": values[:, 0],
        "PC2": values[:, 1],
        "cluster": df.loc[X_scaled.index, cluster_col].astype(str).values,
    })
