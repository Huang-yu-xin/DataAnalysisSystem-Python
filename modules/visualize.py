import pandas as pd


def _fig_to_div(fig):
    """
    将 Plotly 图表转成 HTML 片段。
    include_plotlyjs='cdn'：页面需要联网加载 Plotly。
    如果你想完全离线，可以改成 include_plotlyjs=True，但页面会变大。
    """
    fig.update_layout(
        template="plotly_white",
        margin={"l": 48, "r": 28, "t": 64, "b": 48},
        font={"family": "Microsoft YaHei, Arial, sans-serif"}
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def _sample_df(df, n=500):
    if len(df) > n:
        return df.sample(n=n, random_state=42).copy()
    return df.copy()


def _require_column(df, column, field_name="字段"):
    if not column or column not in df.columns:
        raise ValueError(f"请选择有效的{field_name}")


def _require_numeric(df, column, field_name="字段"):
    _require_column(df, column, field_name)
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"{field_name}必须是数值字段")


def create_missing_value_bar(df, title="各字段缺失值数量统计"):
    """
    生成各字段缺失值数量柱状图，只展示缺失值数量大于 0 的字段。
    """
    import plotly.express as px

    if df is None or df.empty:
        return None

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        return None

    total_rows = max(len(df), 1)
    plot_df = pd.DataFrame({
        "字段": missing.index,
        "缺失数量": missing.values.astype(int),
        "缺失率": (missing.values / total_rows * 100).round(2),
    })

    fig = px.bar(
        plot_df,
        x="字段",
        y="缺失数量",
        title=title,
        hover_data={"缺失率": ":.2f"}
    )
    fig.update_traces(marker_color="#0f766e")
    fig.update_layout(xaxis_title="字段名", yaxis_title="缺失值数量")

    return _fig_to_div(fig)


def create_histogram(df, column):
    """根据指定数值字段生成直方图，用于观察数据分布。"""
    import plotly.express as px

    _require_numeric(df, column, "直方图字段")
    data = df[[column]].dropna()
    if data.empty:
        raise ValueError("所选字段没有可用于绘图的有效数据")

    fig = px.histogram(
        data,
        x=column,
        nbins=30,
        title=f"{column} 数据分布直方图",
        labels={column: column}
    )
    fig.update_traces(marker_color="#2563eb")

    return _fig_to_div(fig)


def create_boxplot(df, column, group_col=None):
    """
    根据指定数值字段生成箱线图，用于观察异常值。
    如果提供 group_col，则按类别字段分组绘制箱线图。
    """
    import plotly.express as px

    _require_numeric(df, column, "箱线图主字段")
    columns = [column]

    if group_col:
        _require_column(df, group_col, "分组字段")
        if pd.api.types.is_numeric_dtype(df[group_col]):
            raise ValueError("箱线图分组字段应选择类别字段")
        columns.append(group_col)

    data = df[columns].dropna()
    if data.empty:
        raise ValueError("所选字段没有可用于绘图的有效数据")

    if group_col:
        group_values = data[group_col].astype(str)
        top_groups = group_values.value_counts().head(20).index
        data = data[group_values.isin(top_groups)].copy()
        data[group_col] = data[group_col].astype(str)
        fig = px.box(data, x=group_col, y=column, title=f"{column} 按 {group_col} 分组的箱线图")
    else:
        fig = px.box(data, y=column, title=f"{column} 箱线图")

    fig.update_traces(marker_color="#7c3aed")

    return _fig_to_div(fig)


def create_elbow_chart(k_values, inertias):
    """
    使用 Plotly 绘制肘部法折线图。
    """
    import plotly.express as px

    if not k_values or not inertias:
        return None

    plot_df = pd.DataFrame({"K 值": k_values, "SSE / Inertia": inertias})
    fig = px.line(
        plot_df,
        x="K 值",
        y="SSE / Inertia",
        markers=True,
        title="KMeans 肘部法 K 值选择图"
    )
    fig.update_traces(line_color="#d97706", marker_size=9)

    return _fig_to_div(fig)


def create_pca_cluster_scatter(pca_df):
    """
    使用 Plotly 绘制 PCA 二维聚类散点图。
    """
    import plotly.express as px

    if pca_df is None or pca_df.empty:
        return None

    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="cluster",
        title="KMeans 聚类结果 PCA 二维可视化",
        labels={"cluster": "聚类类别"}
    )

    return _fig_to_div(fig)


def create_default_charts(df):
    """
    根据数据自动生成默认动态图表。
    天气数据 Weather_Data.csv 会生成温度趋势、降雨占比、湿度温度散点图、
    相关性热力图；普通数据会生成通用类别统计与数值关系图。
    """
    import plotly.express as px

    charts = []
    temp_df = df.copy()

    missing_chart = create_missing_value_bar(temp_df)
    if missing_chart:
        charts.append({
            "title": "缺失值图：各字段缺失值数量统计",
            "html": missing_chart
        })

    if "Date" in temp_df.columns and ("MaxTemp" in temp_df.columns or "MinTemp" in temp_df.columns):
        chart_df = temp_df.copy()
        chart_df["Date"] = pd.to_datetime(chart_df["Date"], errors="coerce")
        chart_df = chart_df.dropna(subset=["Date"]).sort_values("Date").head(300)
        y_cols = [col for col in ["MinTemp", "MaxTemp", "Temp9am", "Temp3pm"] if col in chart_df.columns]

        if y_cols:
            fig = px.line(
                chart_df,
                x="Date",
                y=y_cols,
                title="天气温度变化趋势",
                labels={"value": "温度", "variable": "字段"}
            )
            charts.append({
                "title": "折线图：温度变化趋势",
                "html": _fig_to_div(fig)
            })

    for col in ["RainToday", "RainTomorrow"]:
        if col in temp_df.columns:
            counts = temp_df[col].value_counts(dropna=False).reset_index()
            counts.columns = [col, "count"]

            fig = px.pie(
                counts,
                names=col,
                values="count",
                title=f"{col} 占比"
            )
            charts.append({
                "title": f"饼图：{col} 占比",
                "html": _fig_to_div(fig)
            })
            break

    if "Humidity3pm" in temp_df.columns and "Temp3pm" in temp_df.columns:
        chart_df = _sample_df(temp_df, 800)
        color_col = "RainTomorrow" if "RainTomorrow" in chart_df.columns else None

        fig = px.scatter(
            chart_df,
            x="Humidity3pm",
            y="Temp3pm",
            color=color_col,
            title="下午湿度与下午温度关系",
            labels={"Humidity3pm": "下午湿度", "Temp3pm": "下午温度"}
        )
        charts.append({
            "title": "散点图：湿度与温度关系",
            "html": _fig_to_div(fig)
        })

    numeric_cols = temp_df.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_cols) >= 2:
        corr = temp_df[numeric_cols[:12]].corr(numeric_only=True).round(2)

        fig = px.imshow(
            corr,
            text_auto=True,
            title="数值字段相关性热力图"
        )
        charts.append({
            "title": "热力图：数值字段相关性",
            "html": _fig_to_div(fig)
        })

        charts.append({
            "title": f"直方图：{numeric_cols[0]} 数据分布",
            "html": create_histogram(temp_df, numeric_cols[0])
        })

        charts.append({
            "title": f"箱线图：{numeric_cols[0]} 异常值观察",
            "html": create_boxplot(temp_df, numeric_cols[0])
        })

    if len(charts) <= 1:
        category_cols = temp_df.select_dtypes(include=["object", "string", "category"]).columns.tolist()

        if category_cols:
            col = category_cols[0]
            counts = temp_df[col].value_counts().head(10).reset_index()
            counts.columns = [col, "count"]
            fig = px.bar(counts, x=col, y="count", title=f"{col} 前10类别数量")
            charts.append({"title": "柱状图：类别数量统计", "html": _fig_to_div(fig)})

        if len(numeric_cols) >= 2:
            chart_df = _sample_df(temp_df, 800)
            fig = px.scatter(chart_df, x=numeric_cols[0], y=numeric_cols[1], title="数值字段散点图")
            charts.append({"title": "散点图：数值关系", "html": _fig_to_div(fig)})

    return charts


def create_custom_chart(df, chart_type, x_col, y_col=None):
    """根据用户在网页选择的列生成动态图表。"""
    import plotly.express as px

    chart_df = df.copy()

    if chart_type == "missing_bar":
        chart = create_missing_value_bar(chart_df)
        if not chart:
            raise ValueError("当前数据没有缺失值，无法生成缺失值柱状图")
        return chart

    if chart_type == "histogram":
        return create_histogram(chart_df, x_col)

    if chart_type == "boxplot":
        group_col = y_col if y_col else None
        return create_boxplot(chart_df, x_col, group_col)

    _require_column(df, x_col, "X 轴字段")

    if chart_type == "bar":
        if y_col and y_col in df.columns and pd.api.types.is_numeric_dtype(df[y_col]):
            data = chart_df.groupby(x_col)[y_col].mean().reset_index().head(30)
            fig = px.bar(data, x=x_col, y=y_col, title=f"{x_col} 与 {y_col} 的平均值柱状图")
        else:
            data = chart_df[x_col].value_counts().head(30).reset_index()
            data.columns = [x_col, "count"]
            fig = px.bar(data, x=x_col, y="count", title=f"{x_col} 类别数量柱状图")

    elif chart_type == "line":
        _require_numeric(df, y_col, "Y 轴字段")
        data = chart_df[[x_col, y_col]].dropna().head(500)
        fig = px.line(data, x=x_col, y=y_col, title=f"{x_col} - {y_col} 折线图")

    elif chart_type == "pie":
        data = chart_df[x_col].value_counts().head(10).reset_index()
        data.columns = [x_col, "count"]
        fig = px.pie(data, names=x_col, values="count", title=f"{x_col} 占比饼图")

    elif chart_type == "scatter":
        _require_numeric(df, x_col, "X 轴字段")
        _require_numeric(df, y_col, "Y 轴字段")
        data = _sample_df(chart_df[[x_col, y_col]].dropna(), 1000)
        fig = px.scatter(data, x=x_col, y=y_col, title=f"{x_col} 与 {y_col} 散点图")

    else:
        raise ValueError("不支持的图表类型")

    return _fig_to_div(fig)
