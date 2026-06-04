document.addEventListener("DOMContentLoaded", () => {
    const restoreButton = (button) => {
        if (!button) {
            return;
        }
        if (button.dataset.originalText) {
            button.textContent = button.dataset.originalText;
            delete button.dataset.originalText;
        }
        button.disabled = false;
    };

    const disableSubmitButton = (form) => {
        const active = document.activeElement;
        let button = null;

        if (active && form.contains(active) && active.type === "submit") {
            button = active;
        } else {
            button = form.querySelector("button[type='submit'], input[type='submit']");
        }

        if (!button || button.disabled) {
            return;
        }

        if (button.tagName.toLowerCase() === "input") {
            button.dataset.originalText = button.value;
            button.value = "正在处理，请稍候...";
        } else {
            button.dataset.originalText = button.textContent;
            button.textContent = "正在处理，请稍候...";
        }
        button.disabled = true;
    };

    const uploadForm = document.querySelector("form.upload-box");
    if (uploadForm) {
        uploadForm.addEventListener("submit", (event) => {
            const fileInput = uploadForm.querySelector("input[type='file'][name='file']");
            if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
                return;
            }

            const file = fileInput.files[0];
            const fileName = file.name.toLowerCase();
            const validExt = [".csv", ".xlsx", ".xls"].some((ext) => fileName.endsWith(ext));
            const maxSize = 10 * 1024 * 1024;

            if (!validExt) {
                event.preventDefault();
                event.stopImmediatePropagation();
                alert("暂不支持该文件类型，请上传 CSV、XLSX 或 XLS 文件。");
                fileInput.value = "";
                restoreButton(uploadForm.querySelector("button[type='submit'], input[type='submit']"));
                return;
            }

            if (file.size > maxSize) {
                event.preventDefault();
                event.stopImmediatePropagation();
                alert("上传文件过大，请上传不超过 10MB 的文件。");
                fileInput.value = "";
                restoreButton(uploadForm.querySelector("button[type='submit'], input[type='submit']"));
            }
        });
    }

    document.querySelectorAll("form").forEach((form) => {
        form.addEventListener("submit", () => {
            disableSubmitButton(form);
        });
    });

    const chartTypeSelect = document.getElementById("chart-type-select");
    const xFieldGroup = document.getElementById("x-field-group");
    const yFieldGroup = document.getElementById("y-field-group");
    const chartHelpText = document.getElementById("chart-help-text");

    const setGroupVisible = (group, visible) => {
        if (!group) {
            return;
        }
        group.classList.toggle("hidden", !visible);
    };

    const updateChartFields = () => {
        if (!chartTypeSelect || !chartHelpText) {
            return;
        }
        const type = chartTypeSelect.value;

        if (type === "bar") {
            setGroupVisible(xFieldGroup, true);
            setGroupVisible(yFieldGroup, true);
            chartHelpText.textContent = "柱状图适合展示类别字段与数值字段之间的对比，建议 X 轴选择类别字段，Y 轴选择数值字段。";
        } else if (type === "line") {
            setGroupVisible(xFieldGroup, true);
            setGroupVisible(yFieldGroup, true);
            chartHelpText.textContent = "折线图适合展示随时间或顺序变化的趋势，建议 X 轴选择日期或有序字段，Y 轴选择数值字段。";
        } else if (type === "pie") {
            setGroupVisible(xFieldGroup, true);
            setGroupVisible(yFieldGroup, false);
            chartHelpText.textContent = "饼图适合展示类别占比，只需要选择一个类别字段。";
        } else if (type === "scatter") {
            setGroupVisible(xFieldGroup, true);
            setGroupVisible(yFieldGroup, true);
            chartHelpText.textContent = "散点图适合观察两个数值字段之间的关系，建议 X 轴和 Y 轴都选择数值字段。";
        } else if (type === "histogram") {
            setGroupVisible(xFieldGroup, true);
            setGroupVisible(yFieldGroup, false);
            chartHelpText.textContent = "直方图适合观察单个数值字段的分布情况，请选择一个数值字段。";
        } else if (type === "boxplot") {
            setGroupVisible(xFieldGroup, true);
            setGroupVisible(yFieldGroup, true);
            chartHelpText.textContent = "箱线图适合观察数值字段的分布和异常值，也可以按类别字段分组展示。";
        } else if (type === "missing_bar") {
            setGroupVisible(xFieldGroup, false);
            setGroupVisible(yFieldGroup, false);
            chartHelpText.textContent = "缺失值柱状图会自动统计各字段缺失值数量，不需要手动选择字段。";
        }
    };

    if (chartTypeSelect) {
        chartTypeSelect.addEventListener("change", updateChartFields);
        updateChartFields();
    }

    const kmeansBox = document.getElementById("kmeans-feature-box");
    if (kmeansBox) {
        const checkboxes = kmeansBox.querySelectorAll("input[type='checkbox'][name='feature_cols']");
        const countEl = document.getElementById("kmeans-selected-count");
        const hintEl = document.getElementById("kmeans-hint");
        const selectAllBtn = document.getElementById("kmeans-select-all");
        const clearBtn = document.getElementById("kmeans-clear");

        const updateKmeansHint = () => {
            const selectedCount = Array.from(checkboxes).filter((cb) => cb.checked).length;
            if (countEl) {
                countEl.textContent = `已选择 ${selectedCount} 个字段`;
            }
            if (hintEl) {
                if (selectedCount >= 2) {
                    hintEl.textContent = "当前字段数量满足 KMeans 聚类分析要求。";
                    hintEl.classList.remove("warning-hint");
                    hintEl.classList.add("success-hint");
                } else {
                    hintEl.textContent = "建议至少选择两个数值字段进行聚类；如果不选择，系统会自动选择默认数值字段。";
                    hintEl.classList.remove("success-hint");
                    hintEl.classList.add("warning-hint");
                }
            }
        };

        checkboxes.forEach((checkbox) => {
            checkbox.addEventListener("change", updateKmeansHint);
        });

        if (selectAllBtn) {
            selectAllBtn.addEventListener("click", () => {
                checkboxes.forEach((checkbox) => {
                    checkbox.checked = true;
                });
                updateKmeansHint();
            });
        }

        if (clearBtn) {
            clearBtn.addEventListener("click", () => {
                checkboxes.forEach((checkbox) => {
                    checkbox.checked = false;
                });
                updateKmeansHint();
            });
        }

        updateKmeansHint();
    }

    const outlierMethod = document.getElementById("outlier-method");
    const outlierActionGroup = document.getElementById("outlier-action-group");
    const outlierColumnsGroup = document.getElementById("outlier-columns-group");
    const outlierHint = document.getElementById("outlier-hint");

    const updateOutlierUI = () => {
        if (!outlierMethod) {
            return;
        }
        const disabled = outlierMethod.value === "none";
        if (outlierActionGroup) {
            outlierActionGroup.classList.toggle("hidden", disabled);
        }
        if (outlierColumnsGroup) {
            outlierColumnsGroup.classList.toggle("hidden", disabled);
        }
        if (outlierHint) {
            outlierHint.textContent = disabled
                ? "当前未启用异常值检测，系统不会处理异常值。"
                : "";
            outlierHint.classList.toggle("warning-hint", disabled);
        }
    };

    if (outlierMethod) {
        outlierMethod.addEventListener("change", updateOutlierUI);
        updateOutlierUI();
    }

    const datetimeStrategy = document.getElementById("datetime-strategy");
    const datetimeHint = document.getElementById("datetime-hint");

    const updateDatetimeHint = () => {
        if (!datetimeStrategy || !datetimeHint) {
            return;
        }
        const value = datetimeStrategy.value;
        let message = "";

        if (value === "continuous") {
            message = "连续日期自动补全适合按日期顺序排列的时间序列数据。如果数据不是连续日期记录，建议选择删除日期缺失行或前向/后向填充。";
        } else if (value === "drop") {
            message = "该策略会删除日期字段缺失的记录，适合 Date 是关键时间索引的情况。";
        } else if (value === "ffill_bfill") {
            message = "该策略会使用相邻日期进行填充，适合缺失日期较少且数据顺序可靠的情况。";
        } else if (value === "none") {
            message = "系统将保留日期字段中的缺失值，后续时间趋势图可能受到影响。";
        }

        datetimeHint.textContent = message;
    };

    if (datetimeStrategy) {
        datetimeStrategy.addEventListener("change", updateDatetimeHint);
        updateDatetimeHint();
    }

    const weatherToggle = document.getElementById("weather-rules-checkbox");
    const weatherHint = document.getElementById("weather-hint");

    const updateWeatherHint = () => {
        if (!weatherToggle || !weatherHint) {
            return;
        }
        if (!weatherToggle.checked) {
            weatherHint.textContent = "关闭后系统不会自动修正湿度、云量、降雨量、MinTemp/MaxTemp 等天气字段。";
            weatherHint.classList.add("warning-hint");
        } else {
            weatherHint.textContent = "";
            weatherHint.classList.remove("warning-hint");
        }
    };

    if (weatherToggle) {
        weatherToggle.addEventListener("change", updateWeatherHint);
        updateWeatherHint();
    }

    document.querySelectorAll(".collapse-title").forEach((title) => {
        title.addEventListener("click", () => {
            const body = title.nextElementSibling;
            if (body) {
                body.classList.toggle("hidden");
            }
        });
    });

    const backToTop = document.getElementById("back-to-top");
    if (backToTop) {
        window.addEventListener("scroll", () => {
            if (window.scrollY > 300) {
                backToTop.classList.add("show");
            } else {
                backToTop.classList.remove("show");
            }
        });
        backToTop.addEventListener("click", () => {
            window.scrollTo({ top: 0, behavior: "smooth" });
        });
    }
});

