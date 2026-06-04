import pandas as pd
from pathlib import Path


def read_data_file(file_path):
    """
    读取 CSV / Excel 文件。
    CSV 尝试多种常见编码，避免中文乱码。
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        encodings = ["utf-8-sig", "utf-8", "gbk", "gb2312", "latin1"]
        last_error = None

        for enc in encodings:
            try:
                return pd.read_csv(file_path, encoding=enc)
            except Exception as e:
                last_error = e

        raise last_error

    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)

    raise ValueError("不支持的文件格式")
