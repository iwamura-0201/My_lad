"""
Process Grouper Module

EventID 4688（プロセス生成）をキーとして、関連するセキュリティイベントを
プロセスIDごとにグループ化する機能を提供します。
"""

import pandas as pd
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass
class ProcessInstance:
    """プロセスのインスタンスを表すクラス"""
    process_id: str
    process_name: str
    creation_row_index: int
    normalized_id: Optional[int] = None  # 正規化されたProcessID（整数）
    parent_process_id: Optional[str] = None  # 親プロセスID
    parent_process_name: Optional[str] = None  # 親プロセス名
    label: str = "normal"  # ラベル（normal または anomaly）
    related_rows: List[Tuple[int, pd.Series]] = field(default_factory=list)
    
    def add_related_row(self, row_index: int, row: pd.Series):
        """関連する行を追加（元の順序を保持）"""
        self.related_rows.append((row_index, row))


def parse_content(content: str) -> Dict[str, str]:
    """
    Contentカラムをパースして辞書形式で返す
    
    Args:
        content: key=value形式の文字列（|で区切り）
    
    Returns:
        パースされたキーと値の辞書
    """
    if pd.isna(content) or content == '':
        return {}
    
    result = {}
    # |で区切る（ただし、値の中にも|が含まれる可能性があるため注意）
    # 単純な分割で対応
    pairs = content.split('|')
    
    for pair in pairs:
        if '=' in pair:
            # 最初の=で分割（値の中に=が含まれる可能性があるため）
            key, value = pair.split('=', 1)
            result[key.strip()] = value.strip()
    
    return result


def normalize_process_id(process_id: str) -> Optional[int]:
    """
    ProcessIDを正規化して整数に変換
    16進数形式（0x...）と10進数形式の両方に対応
    
    Args:
        process_id: ProcessIDの文字列
    
    Returns:
        正規化された整数値（変換できない場合はNone）
    """
    if not process_id:
        return None
    
    try:
        # 16進数形式の場合 (例: 0x0000000000001234)
        if process_id.lower().startswith('0x'):
            return int(process_id, 16)
        # 10進数形式の場合
        return int(process_id)
    except ValueError:
        return None


def get_process_id_from_content(content_dict: Dict[str, str]) -> Optional[str]:
    """
    ContentのパースされたデータからProcessIDを取得
    ProcessIdとProcessIDの両方の表記に対応
    
    Args:
        content_dict: パースされたContentの辞書
    
    Returns:
        ProcessID（見つからない場合はNone）
    """
    # 表記ゆれに対応: ProcessId, ProcessID, processId, processid など
    for key in content_dict:
        if key.lower() == 'processid':
            return content_dict[key]
    return None


def get_new_process_info_from_4688(content_dict: Dict[str, str]) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    EventID 4688のContentから新規プロセスのIDと名前、親プロセス情報を取得
    
    Args:
        content_dict: パースされたContentの辞書
    
    Returns:
        (NewProcessId, NewProcessName, ParentProcessId, ParentProcessName) のタプル
        親プロセス情報がない場合はNone
    """
    new_process_id = None
    new_process_name = None
    parent_process_id = None
    parent_process_name = None
    
    for key, value in content_dict.items():
        key_lower = key.lower()
        if key_lower == 'newprocessid':
            new_process_id = value
        elif key_lower == 'newprocessname':
            new_process_name = value
        elif key_lower == 'processid':
            # EventID 4688では、ProcessIdは親プロセスのIDを指す
            parent_process_id = value
        elif key_lower == 'parentprocessname':
            parent_process_name = value
    
    return new_process_id, new_process_name, parent_process_id, parent_process_name


def group_events_by_process(csv_path: str) -> Dict[str, any]:
    """
    CSVファイルを読み込み、プロセスIDごとにイベントをグループ化する
    
    Args:
        csv_path: security2.csvのパス
    
    Returns:
        以下のキーを持つ辞書:
        - 'processes': ProcessIDをキーとしたProcessInstanceの辞書
        - 'else_rows': ProcessIDが存在しなかった行のリスト（(index, row)のタプル）
        - 'summary': 処理結果のサマリー情報
    """
    # CSVを読み込み
    df = pd.read_csv(csv_path)
    
    # プロセスインスタンスを保持する辞書（作成順を保持）
    processes: Dict[str, ProcessInstance] = OrderedDict()
    
    # 正規化されたProcessIDからキーへのマッピング（高速検索用）
    normalized_id_to_keys: Dict[int, List[str]] = {}
    
    # ProcessIDが存在しなかった行
    else_rows: List[Tuple[int, pd.Series]] = []
    
    # 統計情報
    total_rows = len(df)
    event_4688_count = 0
    matched_rows = 0
    
    # 全行を順番に処理
    for idx, row in df.iterrows():
        event_id = row['EventID']
        content = row['Content']
        content_dict = parse_content(content)
        
        if event_id == 4688:
            # プロセス生成イベント
            event_4688_count += 1
            new_process_id, new_process_name, parent_process_id, parent_process_name = get_new_process_info_from_4688(content_dict)
            
            if new_process_id:
                normalized_id = normalize_process_id(new_process_id)
                
                # Labelカラムを取得（"-" → "normal", "anomaly" → "anomaly"）
                raw_label = row.get('Label', '-')
                label = "anomaly" if raw_label == "anomaly" else "normal"
                
                # キーを一意にするためにインデックスを使用
                key = f"{new_process_id}_{idx}"
                
                processes[key] = ProcessInstance(
                    process_id=new_process_id,
                    process_name=new_process_name or "",
                    creation_row_index=idx,
                    normalized_id=normalized_id,
                    parent_process_id=parent_process_id,
                    parent_process_name=parent_process_name,
                    label=label
                )
                
                # 正規化IDからキーへのマッピングを更新
                if normalized_id is not None:
                    if normalized_id not in normalized_id_to_keys:
                        normalized_id_to_keys[normalized_id] = []
                    normalized_id_to_keys[normalized_id].append(key)
        else:
            # 4688以外のイベント：ProcessIDを探して対応するインスタンスに追加
            process_id = get_process_id_from_content(content_dict)
            
            if process_id:
                normalized_id = normalize_process_id(process_id)
                
                if normalized_id is not None and normalized_id in normalized_id_to_keys:
                    # 最新の対応するプロセスインスタンスに追加
                    # （同じProcessIDが複数回生成される場合、最後のものに追加）
                    latest_key = normalized_id_to_keys[normalized_id][-1]
                    processes[latest_key].add_related_row(idx, row)
                    matched_rows += 1
                else:
                    # ProcessIDはあるが、対応する4688イベントが見つからない
                    # elseとして扱う
                    else_rows.append((idx, row))
            else:
                # ProcessIDが存在しない行
                else_rows.append((idx, row))
    
    summary = {
        'total_rows': total_rows,
        'event_4688_count': event_4688_count,
        'unique_processes': len(processes),
        'matched_rows': matched_rows,
        'else_rows_count': len(else_rows)
    }
    
    return {
        'processes': processes,
        'else_rows': else_rows,
        'summary': summary
    }


def build_process_tree(result: Dict) -> Dict:
    """
    プロセスを親子関係に基づいて階層構造にまとめ、JSON形式で返す
    
    Args:
        result: group_events_by_processの戻り値
    
    Returns:
        階層構造のプロセスツリー（JSON互換の辞書形式）
        {
            "root_processes": [...],  # parent_process_idがNoneのプロセス
            "orphan_processes": [...],  # 親が見つからないプロセス
            "statistics": {...}
        }
    """
    processes = result['processes']
    
    # 正規化IDからProcessInstanceへのマッピング
    normalized_id_to_proc: Dict[int, List[ProcessInstance]] = {}
    for key, proc in processes.items():
        if proc.normalized_id is not None:
            if proc.normalized_id not in normalized_id_to_proc:
                normalized_id_to_proc[proc.normalized_id] = []
            normalized_id_to_proc[proc.normalized_id].append(proc)
    
    def proc_to_dict(proc: ProcessInstance, include_children: bool = True) -> Dict:
        """ProcessInstanceを辞書形式に変換"""
        result = {
            "process_id": proc.process_id,
            "process_name": proc.process_name,
            "normalized_id": proc.normalized_id,
            "parent_process_id": proc.parent_process_id,
            "parent_process_name": proc.parent_process_name,
            "label": proc.label,
            "creation_row_index": proc.creation_row_index,
            "related_events_count": len(proc.related_rows)
        }
        
        if include_children:
            result["children"] = []
        
        return result
    
    def find_children(parent_normalized_id: int, visited: set) -> List[Dict]:
        """指定された親IDを持つ子プロセスを再帰的に取得"""
        children = []
        for key, proc in processes.items():
            if proc.parent_process_id:
                parent_norm_id = normalize_process_id(proc.parent_process_id)
                if parent_norm_id == parent_normalized_id and key not in visited:
                    visited.add(key)
                    child_dict = proc_to_dict(proc)
                    if proc.normalized_id is not None:
                        child_dict["children"] = find_children(proc.normalized_id, visited)
                    children.append(child_dict)
        return children
    
    # ルートプロセス（parent_process_idがNoneのもの）
    root_processes = []
    # 親が見つからないプロセス（親IDはあるが、対応する4688が存在しない）
    orphan_processes = []
    # 処理済みのプロセスキー
    visited: set = set()
    
    # 1. まずルートプロセスを特定
    for key, proc in processes.items():
        if proc.parent_process_id is None:
            visited.add(key)
            root_dict = proc_to_dict(proc)
            if proc.normalized_id is not None:
                root_dict["children"] = find_children(proc.normalized_id, visited)
            root_processes.append(root_dict)
    
    # 2. 親プロセスが存在するが、親がルートでないものを処理
    # 親がprocesses内に存在するかチェック
    for key, proc in processes.items():
        if key not in visited:
            if proc.parent_process_id:
                parent_norm_id = normalize_process_id(proc.parent_process_id)
                # 親がprocesses内に存在するかチェック
                parent_exists = parent_norm_id in normalized_id_to_proc
                if not parent_exists:
                    # 親が見つからない場合、orphanとして扱う
                    visited.add(key)
                    orphan_dict = proc_to_dict(proc)
                    if proc.normalized_id is not None:
                        orphan_dict["children"] = find_children(proc.normalized_id, visited)
                    orphan_processes.append(orphan_dict)
    
    # 統計情報
    statistics = {
        "total_processes": len(processes),
        "root_processes_count": len(root_processes),
        "orphan_processes_count": len(orphan_processes),
        "processes_in_tree": len(visited)
    }
    
    return {
        "root_processes": root_processes,
        "orphan_processes": orphan_processes,
        "statistics": statistics
    }


def print_summary(result: Dict) -> None:
    """結果のサマリーを表示"""
    summary = result['summary']
    print("=" * 60)
    print("処理結果サマリー")
    print("=" * 60)
    print(f"総行数: {summary['total_rows']}")
    print(f"EventID 4688（プロセス生成）の数: {summary['event_4688_count']}")
    print(f"一意なプロセス数: {summary['unique_processes']}")
    print(f"プロセスに紐付けられた行数: {summary['matched_rows']}")
    print(f"else（ProcessIDなし）の行数: {summary['else_rows_count']}")
    print("=" * 60)


def get_process_details(result: Dict, limit: int = 5) -> None:
    """プロセスの詳細を表示（デバッグ用）"""
    processes = result['processes']
    
    print(f"\n最初の{limit}件のプロセス詳細:")
    print("-" * 60)
    
    for i, (key, proc) in enumerate(processes.items()):
        if i >= limit:
            break
        print(f"\nプロセス {i+1}:")
        print(f"  ProcessID: {proc.process_id}")
        print(f"  ProcessName: {proc.process_name}")
        print(f"  ParentProcessID: {proc.parent_process_id}")
        print(f"  ParentProcessName: {proc.parent_process_name}")
        print(f"  Label: {proc.label}")
        print(f"  作成行インデックス: {proc.creation_row_index}")
        print(f"  関連行数: {len(proc.related_rows)}")

if __name__ == "__main__":
    import json
    
    # テスト実行
    csv_path = "/home/ubuntu/My_lad/data/interim/WEB1/security2.csv"
    
    print("セキュリティイベントのプロセスグループ化を開始...")
    result = group_events_by_process(csv_path)
    
    print_summary(result)
    get_process_details(result, limit=10)
    
    # プロセスツリーのテスト
    print("\n" + "=" * 60)
    print("プロセスツリーの構築")
    print("=" * 60)
    tree = build_process_tree(result)
    print(f"統計: {json.dumps(tree['statistics'], indent=2)}")
    print(f"\nルートプロセス数: {len(tree['root_processes'])}")
    print(f"孤児プロセス数: {len(tree['orphan_processes'])}")
    
    # 最初の3つのルートプロセスを表示
    print("\n最初の3つのルートプロセス:")
    for i, root in enumerate(tree['root_processes'][:3]):
        print(f"  {i+1}. {root['process_name']} (子: {len(root['children'])})")
