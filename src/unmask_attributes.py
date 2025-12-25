"""
EventTemplate の属性マスク解除スクリプト

no_mask.json で指定された EventID と属性に基づいて、
EventTemplate カラムの <*> マスクを Content カラムの実際の値で置き換える。
"""

from pathlib import Path
import pandas as pd
import json
import re
import hashlib
from typing import Dict, List


def load_no_mask_config(config_path: Path) -> Dict[str, List[str]]:
    """
    no_mask.json を読み込む
    
    Parameters
    ----------
    config_path : Path
        no_mask.json のパス
        
    Returns
    -------
    Dict[str, List[str]]
        EventID をキー、マスク解除する属性名のリストを値とする辞書
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_key_value_string(kv_string: str) -> Dict[str, str]:
    """
    Key=Value|Key=Value 形式の文字列を辞書に変換
    
    Parameters
    ----------
    kv_string : str
        パース対象の文字列（例: "SubjectUserSid=S-1-5-21|SubjectUserName=taro"）
        
    Returns
    -------
    Dict[str, str]
        キーと値のペアを持つ辞書
    """
    if pd.isna(kv_string) or kv_string == '':
        return {}
    
    result = {}
    # | で分割して各 Key=Value ペアを処理
    pairs = kv_string.split('|')
    for pair in pairs:
        if '=' in pair:
            key, value = pair.split('=', 1)  # 最初の = で分割（値に = が含まれる可能性を考慮）
            result[key] = value
    
    return result


def compute_md5_hash(text: str) -> str:
    """
    文字列のMD5ハッシュ値を計算（8文字）
    
    Parameters
    ----------
    text : str
        ハッシュ化する文字列
        
    Returns
    -------
    str
        MD5ハッシュ値の最初の8文字（16進数文字列）
    """
    if pd.isna(text) or text == '':
        return ''
    
    # UTF-8エンコードしてMD5ハッシュを計算し、最初の8文字を返す
    return hashlib.md5(str(text).encode('utf-8')).hexdigest()[0:8]


def unmask_template(event_template: str, content: str, attributes_to_unmask: List[str]) -> str:
    """
    EventTemplate の指定された属性のマスクを解除
    
    Parameters
    ----------
    event_template : str
        マスクされたテンプレート文字列
    content : str
        実際の値を含む Content 文字列
    attributes_to_unmask : List[str]
        マスクを解除する属性名のリスト
        
    Returns
    -------
    str
        マスクが解除されたテンプレート文字列
    """
    if pd.isna(event_template) or pd.isna(content):
        return event_template
    
    # Content から実際の値を取得
    content_dict = parse_key_value_string(content)
    
    # EventTemplate をパース
    template_dict = parse_key_value_string(event_template)
    
    # 指定された属性のマスクを解除
    for attr in attributes_to_unmask:
        if attr in template_dict and attr in content_dict:
            template_dict[attr] = content_dict[attr]
    
    # 辞書を Key=Value|Key=Value 形式に戻す
    result = '|'.join([f"{k}={v}" for k, v in template_dict.items()])
    
    return result


def process_csv(
    input_csv_path: Path,
    output_csv_path: Path,
    no_mask_config: Dict[str, List[str]],
    new_column_name: str = 'EventTemplateUnmasked',
    add_hash: bool = True
) -> None:
    """
    CSV ファイルを処理してマスク解除された新しいカラムを追加
    
    Parameters
    ----------
    input_csv_path : Path
        入力 CSV ファイルのパス
    output_csv_path : Path
        出力 CSV ファイルのパス
    no_mask_config : Dict[str, List[str]]
        EventID ごとのマスク解除する属性の設定
    new_column_name : str, optional
        新しく作成するカラムの名前（デフォルト: 'EventTemplateUnmasked'）
    add_hash : bool, optional
        MD5ハッシュカラムを追加するかどうか（デフォルト: True）
    """
    # CSV を読み込み
    print(f"Loading CSV from: {input_csv_path}")
    df = pd.read_csv(input_csv_path, index_col=0)
    
    print(f"Total rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # 新しいカラムを初期化（EventTemplate のコピー）
    df[new_column_name] = df['EventTemplate'].copy()
    
    # EventID ごとに処理
    for event_id_str, attributes in no_mask_config.items():
        print(f"\nProcessing EventID: {event_id_str}, Attributes: {attributes}")
        
        # EventID が一致する行を抽出
        # EventID カラムは整数型の可能性があるため、両方の型で試す
        try:
            event_id_int = int(event_id_str)
            mask = df['EventID'] == event_id_int
        except ValueError:
            mask = df['EventID'] == event_id_str
        
        matched_count = mask.sum()
        print(f"  Found {matched_count} rows with EventID={event_id_str}")
        
        if matched_count == 0:
            continue
        
        # マスク解除を適用
        df.loc[mask, new_column_name] = df.loc[mask].apply(
            lambda row: unmask_template(
                row['EventTemplate'],
                row['Content'],
                attributes
            ),
            axis=1
        )
    
    # MD5ハッシュカラムを追加
    if add_hash:
        hash_column_name = f"{new_column_name}Hash"
        print(f"\nGenerating MD5 hash column: {hash_column_name}")
        df[hash_column_name] = df[new_column_name].apply(compute_md5_hash)
        print(f"  Hash column created with {df[hash_column_name].notna().sum()} values")
    
    # 結果を保存
    print(f"\nSaving result to: {output_csv_path}")
    df.to_csv(output_csv_path, index=False)
    print("Done!")
    
    # サンプルを表示
    print("\n--- Sample of unmasked data ---")
    for event_id_str in no_mask_config.keys():
        try:
            event_id_int = int(event_id_str)
            sample = df[df['EventID'] == event_id_int].head(2)
        except ValueError:
            sample = df[df['EventID'] == event_id_str].head(2)
        
        if len(sample) > 0:
            print(f"\nEventID {event_id_str}:")
            for idx, row in sample.iterrows():
                print(f"  Original: {row['EventTemplate'][:100]}...")
                print(f"  Unmasked: {row[new_column_name][:100]}...")
                if add_hash:
                    hash_col = f"{new_column_name}Hash"
                    print(f"  Hash:     {row[hash_col]}")


def main():
    """
    メイン処理
    """
    # パスの設定
    base_dir = Path(__file__).parent.parent
    config_path = base_dir / 'src' / 'no_mask.json'
    input_csv_path = base_dir / 'data' / 'interim' / 'refine' / 'security2_structured.csv'
    output_csv_path = base_dir / 'data' / 'interim' / 'refine' / 'security2_structured_unmasked.csv'
    
    # 設定ファイルを読み込み
    print("Loading no_mask configuration...")
    no_mask_config = load_no_mask_config(config_path)
    print(f"Configuration loaded: {no_mask_config}")
    
    # CSV を処理
    process_csv(
        input_csv_path=input_csv_path,
        output_csv_path=output_csv_path,
        no_mask_config=no_mask_config,
        new_column_name='EventTemplateUnmasked',
        add_hash=True
    )


if __name__ == '__main__':
    main()
