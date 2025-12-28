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
from typing import Dict, List, Literal

# LSH用のパラメータ
LSH_HASH_SIZE = 8  # 出力ハッシュのサイズ（文字数）
LSH_NUM_PERMUTATIONS = 128  # MinHashの順列数


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


def compute_lsh_hash(text: str, hash_size: int = LSH_HASH_SIZE, num_permutations: int = LSH_NUM_PERMUTATIONS) -> str:
    """
    文字列のLSH（Locality Sensitive Hashing）ハッシュ値を計算
    
    MinHash を使用して、類似した文字列が類似したハッシュ値を持つようにする。
    文字列を shingle（n-gram）に分割し、各 shingle のハッシュ値の最小値を使用。
    
    Parameters
    ----------
    text : str
        ハッシュ化する文字列
    hash_size : int, optional
        出力ハッシュのサイズ（デフォルト: 8）
    num_permutations : int, optional
        MinHashの順列数（デフォルト: 128）
        
    Returns
    -------
    str
        LSHハッシュ値（16進数文字列）
    """
    if pd.isna(text) or text == '':
        return ''
    
    text = str(text)
    
    # shingle（3-gram）を生成
    shingle_size = 3
    shingles = set()
    if len(text) >= shingle_size:
        for i in range(len(text) - shingle_size + 1):
            shingles.add(text[i:i + shingle_size])
    else:
        # テキストが短い場合はテキスト全体を1つのshingleとして使用
        shingles.add(text)
    
    # MinHash署名を計算
    # 各順列について、shingleのハッシュ値の最小値を計算
    signature = []
    for i in range(num_permutations):
        min_hash = float('inf')
        for shingle in shingles:
            # shingleとシードを組み合わせてハッシュを計算
            combined = f"{shingle}_{i}"
            h = int(hashlib.md5(combined.encode('utf-8')).hexdigest(), 16)
            min_hash = min(min_hash, h)
        signature.append(min_hash)
    
    # 署名をバンドに分割してLSHバケットを計算
    # 各バンドの結合ハッシュを計算してコンパクトな表現を作成
    bands = 16  # バンド数
    rows_per_band = num_permutations // bands
    
    band_hashes = []
    for b in range(bands):
        band_start = b * rows_per_band
        band_end = band_start + rows_per_band
        band_values = signature[band_start:band_end]
        # バンド内の値を結合してハッシュ
        band_str = '|'.join(str(v) for v in band_values)
        band_hash = hashlib.md5(band_str.encode('utf-8')).hexdigest()
        band_hashes.append(band_hash[:2])  # 各バンドから2文字使用
    
    # 最終的なLSHハッシュを生成（指定サイズに切り詰め）
    combined_hash = ''.join(band_hashes)
    return combined_hash[:hash_size]


def compute_hash(text: str, method: Literal['md5', 'lsh'] = 'md5') -> str:
    """
    指定された手法でハッシュ値を計算
    
    Parameters
    ----------
    text : str
        ハッシュ化する文字列
    method : Literal['md5', 'lsh'], optional
        ハッシュ手法（'md5' または 'lsh'、デフォルト: 'md5'）
        
    Returns
    -------
    str
        ハッシュ値
    """
    if method == 'lsh':
        return compute_lsh_hash(text)
    else:
        return compute_md5_hash(text)


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


from typing import Union

def process_csv(
    input_csv_path: Path,
    output_csv_path: Path,
    no_mask_config: Union[str, Path, Dict[str, List[str]]],
    new_column_name: str = 'EventTemplateUnmasked',
    add_hash: bool = True,
    hash_method: Literal['md5', 'lsh'] = 'md5',
    use_gpu: bool = True,
    use_multiprocess: bool = True,
    n_workers: int = None
) -> None:
    """
    CSV ファイルを処理してマスク解除された新しいカラムを追加
    
    Parameters
    ----------
    input_csv_path : Path
        入力 CSV ファイルのパス
    output_csv_path : Path
        出力 CSV ファイルのパス
    no_mask_config : Union[str, Path, Dict[str, List[str]]]
        EventID ごとのマスク解除する属性の設定。
        - 辞書: 直接設定を渡す
        - 文字列またはPath: JSONファイルのパスを指定（自動的に読み込む）
    new_column_name : str, optional
        新しく作成するカラムの名前（デフォルト: 'EventTemplateUnmasked'）
    add_hash : bool, optional
        ハッシュカラムを追加するかどうか（デフォルト: True）
    hash_method : Literal['md5', 'lsh'], optional
        ハッシュ変換手法（デフォルト: 'md5'）
        - 'md5': MD5ハッシュ（決定論的、同一文字列は同一ハッシュ）
        - 'lsh': Locality Sensitive Hashing（類似文字列が類似ハッシュを持つ）
    use_gpu : bool, optional
        GPUを使用するかどうか（デフォルト: True）
        Trueの場合、CuPyが利用可能であればGPU上で計算を実行。
        hash_method='lsh'の場合のみ有効。
    use_multiprocess : bool, optional
        マルチプロセス並列化を使用するかどうか（デフォルト: True）
        LSH計算時にCPUコアを並列使用して高速化する。
    n_workers : int, optional
        並列ワーカー数（デフォルト: None = CPUコア数を自動検出）
    """
    # no_mask_config が文字列またはPathの場合はJSONファイルを読み込む
    if isinstance(no_mask_config, (str, Path)):
        config_path = Path(no_mask_config)
        print(f"Loading no_mask config from: {config_path}")
        no_mask_config = load_no_mask_config(config_path)
    
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
    
    # ハッシュカラムを追加
    if add_hash:
        hash_column_name = f"{new_column_name}Hash_{hash_method}"
        
        # ラベル生成
        method_labels = {'md5': 'MD5', 'lsh': 'LSH'}
        method_label = method_labels.get(hash_method, hash_method)
        if hash_method == 'lsh':
            if use_gpu:
                method_label += ' (GPU)'
            elif use_multiprocess:
                method_label += ' (CPU + multiprocess)'
            else:
                method_label += ' (CPU)'
        
        print(f"\nGenerating {method_label} hash column: {hash_column_name}")
        
        if hash_method == 'lsh':
            # LSHの場合
            if use_gpu:
                # GPU版を使用
                from src.lsh_gpu import compute_lsh_hash_gpu_batch
                texts = df[new_column_name].fillna('').tolist()
                df[hash_column_name] = compute_lsh_hash_gpu_batch(
                    texts, use_multiprocess=use_multiprocess, n_workers=n_workers
                )
            elif use_multiprocess:
                # CPU版 + マルチプロセス
                from src.lsh_gpu import compute_lsh_hash_cpu_batch
                texts = df[new_column_name].fillna('').tolist()
                df[hash_column_name] = compute_lsh_hash_cpu_batch(
                    texts, use_multiprocess=True, n_workers=n_workers
                )
            else:
                # CPU版シングルプロセス
                df[hash_column_name] = df[new_column_name].apply(lambda x: compute_hash(x, 'lsh'))
        else:
            # MD5の場合
            df[hash_column_name] = df[new_column_name].apply(lambda x: compute_hash(x, hash_method))
        
        print(f"  Hash column created with {df[hash_column_name].notna().sum()} values ({method_label})")
    
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
                    hash_col = f"{new_column_name}Hash_{hash_method}"
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
    
    # ハッシュ手法を指定（'md5' または 'lsh'）
    hash_method: Literal['md5', 'lsh'] = 'md5'  # デフォルトはMD5
    
    # CSV を処理
    process_csv(
        input_csv_path=input_csv_path,
        output_csv_path=output_csv_path,
        no_mask_config=no_mask_config,
        new_column_name='EventTemplateUnmasked',
        add_hash=True,
        hash_method=hash_method
    )


if __name__ == '__main__':
    main()
