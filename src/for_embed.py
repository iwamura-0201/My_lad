import pandas as pd
from typing import Dict, List, Set
from pathlib import Path
from collections import defaultdict


def parse_content(content: str) -> Dict[str, str]:
    if not isinstance(content, str) or content.strip() == "":
        return {}

    result = {}
    fields = content.split("|")

    for field in fields:
        if "=" not in field:
            continue

        key, value = field.split("=", 1)
        key = key.strip()
        if not key:
            continue

        result[key] = value

    return result
        
def export_csv_per_eventid(
    df: pd.DataFrame,
    output_dir: str | Path,
    filename_prefix: str = "event"
) -> None:
    """
    EventID ごとに、その EventID に出現した属性のみをカラムとして CSV 出力
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # EventID 単位で処理を完結させる
    for event_id, group_df in df.groupby("EventID"):
        records: List[Dict] = []
        keys_in_event = set()

        # まず属性キーの集合を EventID 単位で収集
        parsed_contents = []
        for _, row in group_df.iterrows():
            content_dict = parse_content(row.get("Content"))
            parsed_contents.append((row, content_dict))
            keys_in_event.update(content_dict.keys())

        # 行データ構築
        for row, content_dict in parsed_contents:
            record = {
                "EventID": row.get("EventID"),
                "timeline": row.get("timeline"),
                "Label": row.get("Label"),
            }

            # この EventID に存在する属性のみを列として採用
            for key in keys_in_event:
                record[key] = content_dict.get(key)

            records.append(record)

        event_df = pd.DataFrame(records)

        output_path = output_dir / f"{filename_prefix}_{event_id}.csv"
        event_df.to_csv(output_path, index=False)
        
        
def build_event_attribute_sets(df: pd.DataFrame) -> Dict[int, Set[str]]:
    """
    EventID -> 属性カラム集合 を構築
    """
    event_attrs: Dict[int, Set[str]] = {}

    for event_id, group_df in df.groupby("EventID"):
        attr_set = set()

        for content in group_df["Content"]:
            parsed = parse_content(content)
            attr_set.update(parsed.keys())

        event_attrs[event_id] = attr_set

    return event_attrs

def jaccard_similarity(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)

def group_events_by_attribute_similarity(
    event_attr_sets: Dict[int, Set[str]],
    threshold: float = 0.7
) -> List[Set[int]]:
    """
    属性カラム構成の類似度に基づいて EventID をグルーピング
    """
    event_ids = list(event_attr_sets.keys())
    visited = set()
    groups = []

    for i, eid in enumerate(event_ids):
        if eid in visited:
            continue

        base_attrs = event_attr_sets[eid]
        group = {eid}
        visited.add(eid)

        for other_id in event_ids[i + 1:]:
            if other_id in visited:
                continue

            sim = jaccard_similarity(
                base_attrs,
                event_attr_sets[other_id]
            )

            if sim >= threshold:
                group.add(other_id)
                visited.add(other_id)

        groups.append(group)

    return groups