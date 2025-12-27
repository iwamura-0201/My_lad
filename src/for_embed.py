import pandas as pd
from typing import Dict


def parse_content(content: str) -> Dict[str, str]:
    """
    Content を安全に dict 化する
    - key=value 形式のみ採用
    - 区切りは | のみ
    - = は最初の1回のみ使用
    - value 側は一切加工しない
    """
    if not isinstance(content, str) or content.strip() == "":
        return {}

    result = {}

    # 属性境界としてのみ | を信用
    fields = content.split("|")

    for field in fields:
        # = がなければ無視
        if "=" not in field:
            continue

        # 最初の = のみで分割
        key, value = field.split("=", 1)

        key = key.strip()
        if not key:
            continue

        # value は strip しない（意味を変えないため）
        result[key] = value

    return result


def eventid_content_to_csv(
    df: pd.DataFrame,
    output_csv: str
) -> pd.DataFrame:
    records = []

    for _, row in df.iterrows():
        record = {"EventID": row["EventID"]}

        content_dict = parse_content(row.get("Content"))
        record.update(content_dict)

        records.append(record)

    result_df = pd.DataFrame(records)
    result_df.to_csv(output_csv, index=False)

    return result_df
