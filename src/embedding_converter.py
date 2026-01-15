"""
EventID を Sentence-BERT 埋め込みに変換するモジュール。

data/processed/score/fixed 下のデータを変換し、
各要素を [EventIDの埋め込み(384次元), score_missing, mahalanobis_score] のベクトルにします。
"""

from pathlib import Path
import numpy as np
import pandas as pd
import json
from typing import Tuple, Optional


def load_embedding_mapping(
    embedding_json_path: Path,
    embedding_npy_path: Path,
) -> dict:
    """
    埋め込みデータを読み込み、EventID -> 埋め込みベクトルの辞書を作成する。
    
    Parameters
    ----------
    embedding_json_path : Path
        EventIDリストとメタデータを含むJSONファイルのパス
    embedding_npy_path : Path
        埋め込みベクトルの numpy 配列ファイルのパス
        
    Returns
    -------
    dict
        EventID (str) -> 埋め込みベクトル (np.ndarray) のマッピング
    """
    with open(embedding_json_path, "r") as f:
        meta = json.load(f)
    
    embeddings = np.load(embedding_npy_path)
    event_ids = meta["event_ids"]
    
    assert len(event_ids) == embeddings.shape[0], \
        f"event_ids count ({len(event_ids)}) != embeddings count ({embeddings.shape[0]})"
    
    return {eid: embeddings[i] for i, eid in enumerate(event_ids)}


def convert_df_to_embedded_array(
    df: pd.DataFrame,
    embedding_map: dict,
    event_id_col: str = "EventID",
    additional_cols: list = ["score_missing", "mahalanobis_score"],
) -> Tuple[np.ndarray, list]:
    """
    deeplog_file_generator の入力と同形式のDataFrameを、埋め込みベクトルに変換する。
    
    入力DataFrame形式:
        - 各行がスライディングウィンドウの1インスタンス
        - event_id_col カラム: イベントIDのリスト
        - additional_cols の各カラム: 追加データ（数値）のリスト
    
    出力形式:
        - 3次元配列: (n_instances, window_size, embedding_dim + len(additional_cols))
        - 各要素は [EventID埋め込み(384次元), score_missing, mahalanobis_score] のベクトル
    
    Parameters
    ----------
    df : pd.DataFrame
        変換対象のDataFrame
    embedding_map : dict
        EventID -> 埋め込みベクトルのマッピング
    event_id_col : str
        EventIDカラム名
    additional_cols : list
        追加で含める数値カラム名のリスト
        
    Returns
    -------
    Tuple[np.ndarray, list]
        - 変換後の3次元配列
        - 未知のEventIDリスト
    """
    unknown_event_ids = set()
    embedding_dim = len(next(iter(embedding_map.values())))  # 384
    total_dim = embedding_dim + len(additional_cols)  # 386
    
    all_instances = []
    
    for idx, row in df.iterrows():
        event_ids = row[event_id_col]
        
        # 各追加カラムのデータを取得
        additional_data = [row[col] for col in additional_cols]
        
        window_size = len(event_ids)
        instance = np.zeros((window_size, total_dim), dtype=np.float32)
        
        for i, eid in enumerate(event_ids):
            eid_str = str(int(eid)) if isinstance(eid, (int, float, np.integer, np.floating)) else str(eid)
            
            if eid_str in embedding_map:
                instance[i, :embedding_dim] = embedding_map[eid_str]
            else:
                unknown_event_ids.add(eid_str)
                # 未知のイベントIDはゼロベクトルのまま
            
            # 追加のカラム値を設定
            for j, col_data in enumerate(additional_data):
                instance[i, embedding_dim + j] = col_data[i]
        
        all_instances.append(instance)
    
    # 全インスタンスを配列に変換
    # ウィンドウサイズが一定でない場合はリストのまま保存
    if len(all_instances) > 0:
        window_sizes = [inst.shape[0] for inst in all_instances]
        if len(set(window_sizes)) == 1:
            # 全て同じウィンドウサイズの場合は3次元配列に
            result = np.stack(all_instances, axis=0)
        else:
            # 異なるウィンドウサイズがある場合はobject配列
            result = np.array(all_instances, dtype=object)
    else:
        result = np.array([])
    
    return result, list(unknown_event_ids)


def save_embedded_data(
    output_path: Path,
    data: np.ndarray,
    labels: Optional[np.ndarray] = None,
    metadata: Optional[dict] = None,
) -> None:
    """
    埋め込みデータをnpzファイルに保存する。
    
    Parameters
    ----------
    output_path : Path
        出力ファイルパス (.npz)
    data : np.ndarray
        埋め込みデータ
    labels : np.ndarray, optional
        ラベルデータ
    metadata : dict, optional
        メタデータ（JSON文字列として保存）
    """
    save_dict = {"data": data}
    
    if labels is not None:
        save_dict["labels"] = labels
    
    if metadata is not None:
        save_dict["metadata"] = np.array([json.dumps(metadata)])
    
    np.savez_compressed(output_path, **save_dict)
    print(f"Saved embedded data to {output_path}")


def convert_deeplog_df_to_npz(
    df: pd.DataFrame,
    output_path: Path,
    embedding_json_path: Path,
    embedding_npy_path: Path,
    event_id_col: str = "EventID",
    additional_cols: list = ["score_missing", "mahalanobis_score"],
    label_col: Optional[str] = "Label",
) -> list:
    """
    deeplog_file_generator形式のDataFrameを埋め込みに変換してnpzで保存する。
    
    Parameters
    ----------
    df : pd.DataFrame
        変換対象のDataFrame（deeplog_file_generatorの入力形式）
    output_path : Path
        出力先のnpzファイルパス
    embedding_json_path : Path
        EventIDリストとメタデータを含むJSONファイルのパス
    embedding_npy_path : Path
        埋め込みベクトルのnumpy配列ファイルのパス
    event_id_col : str
        EventIDカラム名
    additional_cols : list
        追加で含める数値カラム名のリスト
    label_col : str, optional
        ラベルカラム名（Noneの場合はラベルを保存しない）
        
    Returns
    -------
    list
        未知のEventIDリスト
    """
    # 埋め込みマッピングを読み込み
    embedding_map = load_embedding_mapping(embedding_json_path, embedding_npy_path)
    
    # 変換
    data, unknown_ids = convert_df_to_embedded_array(
        df=df,
        embedding_map=embedding_map,
        event_id_col=event_id_col,
        additional_cols=additional_cols,
    )
    
    # ラベル取得
    labels = None
    if label_col is not None and label_col in df.columns:
        labels = df[label_col].values
    
    # メタデータ
    metadata = {
        "event_id_col": event_id_col,
        "additional_cols": additional_cols,
        "embedding_dim": len(next(iter(embedding_map.values()))),
        "total_dim": len(next(iter(embedding_map.values()))) + len(additional_cols),
        "n_instances": len(df),
        "unknown_event_ids": unknown_ids,
    }
    
    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_embedded_data(output_path, data, labels, metadata)
    
    # 未知のEventIDを表示
    if unknown_ids:
        print(f"[WARNING] 未知のEventID（{len(unknown_ids)}件）:")
        for uid in sorted(unknown_ids, key=lambda x: int(x) if x.isdigit() else x):
            print(f"  - {uid}")
    
    return unknown_ids
    
 
# deeplog形式ファイルの読み込みとDataFrame変換用関数
def load_deeplog_file(filepath: Path, features: list) -> pd.DataFrame:
    """
    deeplog_file_generator で生成されたファイルを読み込み、DataFrameに変換する。
    """
    rows = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # スペース区切りで各要素を取得
            elements = line.split()
            
            # 各要素をカンマ区切りで分解
            feature_data = {feat: [] for feat in features}
            for elem in elements:
                parts = elem.split(",")
                for i, feat in enumerate(features):
                    if i < len(parts):
                        try:
                            feature_data[feat].append(float(parts[i]))
                        except ValueError:
                            feature_data[feat].append(parts[i])
            
            rows.append(feature_data)
    
    return pd.DataFrame(rows)



def main(project_root: Path):
    # パス設定
    embedding_json = project_root / "data/embedding/eventid_sentence_bert_embeddings.json"
    embedding_npy  = project_root / "data/embedding/eventid_sentence_bert_embeddings.npy"

    input_dir  = project_root / "data/processed/scores/fixed"
    output_dir = project_root / "data/processed/scores/fixed_embedded"
    output_dir.mkdir(parents=True, exist_ok=True)

    features = ["EventID","mahalanobis_score","score_missing"]

    print("=== Loading embedding mapping ===")
    embedding_map = load_embedding_mapping(embedding_json, embedding_npy)
    print(f"Loaded {len(embedding_map)} embeddings "
          f"(dim={len(next(iter(embedding_map.values())))})")

    for split in ["train", "test_normal", "test_abnormal"]:
        input_file = input_dir / split
        if not input_file.exists():
            print(f"Skipping {split}: file not found")
            continue

        print(f"\n=== Converting {split} ===")

        # deeplog ファイル読み込み
        df = load_deeplog_file(input_file, features)
        print(f"Loaded {len(df)} instances")

        # ラベル設定（test_abnormalは1、それ以外は0）
        if split == "test_abnormal":
            df["Label"] = 1
        elif split == "test_normal":
            df["Label"] = 0
        else:
            df["Label"] = 0  # train is assumed normal

        output_file = output_dir / f"{split}.npz"

        unknown_ids = convert_deeplog_df_to_npz(
            df=df,
            output_path=output_file,
            embedding_json_path=embedding_json,
            embedding_npy_path=embedding_npy,
            event_id_col="EventID",
            additional_cols=["mahalanobis_score", "score_missing"],
            label_col="Label",
        )
        if unknown_ids:
            print(f"Unknown EventIDs: {len(unknown_ids)}")

    print("\n=== Conversion complete ===")

if __name__ == "__main__":
    main(Path(__file__).parent.parent)