from pathlib import Path
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from xml.etree.ElementTree import fromstring, ElementTree
from Evtx.Evtx import Evtx
import csv
from typing import List, Union
from xml.etree.ElementTree import ElementTree, fromstring, iterparse, tostring
from itertools import islice

from logparser.Drain import LogParser as Original_Drain
# from src.logparser.Spell import LogParser as Spell

INTERIM_DIR = Path('../data/interim')
PROCESSED_DIR = Path('../data/processed')
RAW_DIR = Path('../data/raw')

# ----------------------------------ここからsecurity.evtxのcsvへのパース関連--------------------------------------#

def strip_namespace(element):
    """
    名前空間を取り除いたタグ名を返す
    """
    return element.tag.split("}")[-1]


def clean_text(value):
    """
    テキストから改行を削除するユーティリティ関数
    """
    if value:
        return value.replace("\n", "").replace("\r", "").strip()
    return value


def parse_event_xml_with_content(xml_content):
    """
    単一のXMLログを解析し、共通部分とContentカラムを生成
    """
    tree = ElementTree(fromstring(xml_content))
    root = tree.getroot()

    record_data = {}  # 共通部分
    content_items = []  # Contentカラム用の項目

    # <System> 部分のデータを抽出
    system_element = root.find(
        "{http://schemas.microsoft.com/win/2004/08/events/event}System"
    )
    if system_element is not None:
        for child in system_element:
            tag = strip_namespace(child)
            if tag == "Channel" and child.text:
                # Channel を列に残す
                record_data[tag] = clean_text(child.text)
            elif child.text:
                record_data[tag] = clean_text(child.text)
            elif child.attrib:
                for key, value in child.attrib.items():
                    record_data[f"{tag}_{key}"] = clean_text(value)
                    
    # <EventData> 部分のデータを抽出
    event_data_element = root.find(
        "{http://schemas.microsoft.com/win/2004/08/events/event}EventData"
    )
    if event_data_element is not None:
        for data_element in event_data_element.findall(
            "{http://schemas.microsoft.com/win/2004/08/events/event}Data"
        ):
            name = data_element.attrib.get("Name", None)
            if name:
                value = clean_text(data_element.text) if data_element.text else None
                if value:  
                    # Content用
                    content_items.append(f"{name}={value}")
                    # カラムとしても保持
                    record_data[name] = value

    # Contentカラムの値を作成
    #record_data["Content"] = ";".join(content_items)
    record_data["Content"] = "|".join(content_items)

    return record_data


def detect_common_fields(records):
    """
    全レコードを解析して共通フィールドを検出（積集合）
    """
    if not records:
        return []

    # 最初のレコードのフィールドを基準にする
    common_fields = set(records[0].keys())

    # 他のレコードと比較して共通するフィールドを見つける
    for record in records[1:]:
        common_fields.intersection_update(record.keys())

    return list(common_fields)

def detect_all_fields(records):
    """
    全レコードを解析して「一度でも出てきた全フィールド（和集合）」を検出
    """
    if not records:
        return []

    all_fields = set()
    for record in records:
        all_fields.update(record.keys())

    return list(all_fields)
    


def evtx_to_csv_without_eventdata_columns(
    evtx_filepath:Path, 
    output_dir:Path,
    output_filename:str, 
    max_records: int | None = None,
):
    """
    evtx ファイルを解析して csv へパースする。
    ：<System> の中身は全レコードの積集合として定義。
    ：<EventData> の中身は Content にまとめて CSV に変換。
    """
    records = []
    
    output_dir.mkdir(exist_ok=True)

    # evtx ファイルを2回処理：1回目は総レコード数の取得、2回目は解析
    with Evtx(str(evtx_filepath)) as log:
        record_count = sum(1 for _ in log.records())  # 総レコード数をカウント

    # evtx ファイルを再度開いて解析
    with Evtx(str(evtx_filepath)) as log:
        for i, record in enumerate(
            tqdm(log.records(), total=record_count, desc="Processing records"), start=1
        ):

            xml_content = record.xml()
            record_data = parse_event_xml_with_content(xml_content)
            records.append(record_data)

    # 共通フィールドを検出
    fields = detect_common_fields(records)

    # Contentカラムは共通フィールドに含める
    if "Content" not in fields:
        fields.append("Content")

    output_path = output_dir/output_filename
    output_path = output_path.with_suffix(".csv")

    # CSV に書き込む
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in tqdm(records, desc="Writing to CSV"):
            # 不足しているフィールドは空白で補完
            complete_record = {field: record.get(field, "") for field in fields}
            writer.writerow(complete_record)

    return 

def evtx_to_csv_without_eventdata_columns_samplingver(
    evtx_filepath:Path, 
    output_dir:Path,
    output_filename:str, 
    max_records:int = 50,
    sample_rate:int = 100      # 1: 全件, 100: 100件に1件 など
):
    """
    evtx ファイルを解析して csv へパースする。
    ：<System> の中身は全レコードの積集合として定義。
    ：<EventData> の中身は Content にまとめて CSV に変換。
    開発用にサンプリング（間引き）も可。
    """
    records = []

    output_dir.mkdir(exist_ok=True)

    # 1回目: 総レコード数カウント（プログレスバー用）
    with Evtx(evtx_filepath) as log:
        record_count = sum(1 for _ in log.records())

    # 2回目: 実際の解析（ここでサンプリング＆max_records を適用）
    with Evtx(evtx_filepath) as log:
        for i, record in enumerate(
            tqdm(log.records(), total=record_count, desc="Processing records"),
            start=0,  # 0 から始めた方が sample_rate と相性が良い
        ):
            # サンプリング：sample_rate ごとに 1 件だけ採用
            if sample_rate > 1 and (i % sample_rate != 0):
                continue

            xml_content = record.xml()
            record_data = parse_event_xml_with_content(xml_content)
            records.append(record_data)

            # max_records に達したら打ち切り
            if max_records is not None and len(records) >= max_records:
                break

    # 共通フィールドを検出
    fields = detect_common_fields(records)

    # Content カラムは必ず含める
    if "Content" not in fields:
        fields.append("Content")

    output_path = output_dir/output_filename
    output_path = output_path.with_suffix(".csv")

    # CSV に書き込む
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in tqdm(records, desc="Writing to CSV"):
            # 不足しているフィールドは空白で補完
            complete_record = {field: record.get(field, "") for field in fields}
            writer.writerow(complete_record)

    return 

def xml_to_csv_parse(
    xml_filepath: Path,
    output_dir: Path,
    output_filename: str,
    max_records: int | None = None,
):
    """
    Windows イベントログの XML ファイルを解析して CSV へパースする。

    ・<Event> ごとに parse_event_xml_with_content() を適用
    ・<System> の中身は全レコードの積集合として定義
    ・<EventData> の中身は Content にまとめて CSV に変換
    ・max_records を指定すると、その件数で打ち切る
    """

    records = []
    output_dir.mkdir(exist_ok=True)

    # iterparse で <Event> 単位にストリーム処理
    context = iterparse(xml_filepath, events=("end",))

    for _, elem in tqdm(context, desc="Processing events"):
        # <Event> 要素の終端に来たときに処理
        if strip_namespace(elem) == "Event":
            xml_content = tostring(elem, encoding="unicode")
            record_data = parse_event_xml_with_content(xml_content)
            records.append(record_data)

            # メモリ節約のため、処理済みノードを開放
            elem.clear()

            # max_records に達したら打ち切り
            if max_records is not None and len(records) >= max_records:
                break

    if not records:
        print("No records were parsed from XML.")
        return

    # 共通フィールド（積集合）を検出
    fields = detect_common_fields(records)

    # Content カラムは必ず含める
    if "Content" not in fields:
        fields.append("Content")

    output_path = output_dir / output_filename
    output_path = output_path.with_suffix(".csv")

    # CSV に書き込み
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in tqdm(records, desc="Writing to CSV"):
            complete_record = {field: record.get(field, "") for field in fields}
            writer.writerow(complete_record)

    print(f"Saved {len(records)} records to {output_path}")
    return



# -----------------------------------------ここからcsvのパース前下準備-----------------------------------------#

# _clean.csv 作成工程の実装をするかも？


# -----------------------------------------ここからログパーサー関連--------------------------------------------#

# オリジナルのDrainを継承・一部メソッドをオーバーライドした独自クラス
class Drain(Original_Drain):
    def __init__(
        self,
        log_format,
        indir="./",
        outdir="./result/",
        depth=4,
        st=0.4,
        maxChild=100,
        rex=...,
        keep_para=True,
    ):
        super().__init__(log_format, indir, outdir, depth, st, maxChild, rex, keep_para)

    def log_to_dataframe(self, log_file, regex, headers, logformat):
        """
        CSVファイルをDataFrameに変換し、LineIdを追加する関数。
        """
        try:
            # CSVファイルを直接読み込む
            #logdf = pd.read_csv(log_file, names=headers, header=0, encoding="utf-8")
            logdf = pd.read_csv(log_file, encoding="utf-8")

            # デバッグ用: 読み込んだデータを表示
            #print("Initial DataFrame:")
            #print(logdf.head())
            
            # Contentカラムの処理
            if "Content" in logdf.columns:
                logdf["Content"] = logdf["Content"].astype(str)
            else:
                print("Warning: 'Content' column not found in CSV.")

            # LineIdを追加（1から始まる連番）
            logdf.insert(0, "LineId", logdf.index + 1)

            # 総データ数を出力
            print("Total size after reading CSV:", len(logdf))

            return logdf
        except Exception as e:
            # エラー発生時の処理
            print(f"Failed to read CSV file {log_file}: {str(e)}")
            return pd.DataFrame(columns=headers)  # 空のDataFrameを返す
        
def parse_log(
    input_dir:Path, 
    output_dir:Path, 
    logfile_name:str, 
    parser_type:str,
    integrated:bool = False,
):
    """
    指定されたパーサーによってログのパースを行う関数。
    パーサーは"Content"のみを参照することに注意。
    logfile_name：拡張子無し, csvを想定
    """
    
    #log_format = "<Timestamp> <RuleTitle> <Level> <Computer> <Channel> <EventID> <RecordID> <Details> <ExtraFieldInfo> <Label> <Content>"
    log_format = "<Version>,<Computer>,<Execution_ThreadID>,<Channel>,<Content>,<Provider_Name>,<Correlation_RelatedActivityID>,<Keywords>,<Opcode>,<Correlation_ActivityID>,<Execution_ProcessID>,<Security_UserID>,<Task>,<Level>,<Provider_Guid>,<TimeCreated_SystemTime>,<EventRecordID>,<EventID>"
    if integrated == True:
        log_format = "<Version>,<Computer>,<Execution_ThreadID>,<Channel>,<Content>,<Provider_Name>,<Correlation_RelatedActivityID>,<Keywords>,<Opcode>,<Correlation_ActivityID>,<Execution_ProcessID>,<Security_UserID>,<Task>,<Level>,<Provider_Guid>,<TimeCreated_SystemTime>,<EventRecordID>,<EventID>,<project>"
    #regex = [
    #    r"(0x)[0-9a-fA-F]+",  # hexadecimal
    #    r"\d+.\d+.\d+.\d+",
    #    #r'/\w+( )$'
    #    r"\d+",
    #]
    regex = [
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',
        r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',
        r'0x[0-9a-fA-F]+',
        r'\b\d+\b',
        r'(?<=\=)[^|]+(?=\||$)'
    ]

    tmp = pd.read_csv(input_dir / f"{logfile_name}.csv")
    tmp = tmp.dropna(subset=["Content"])
    tmp.to_csv(input_dir / "buffer.csv", index=False)
    
    keep_para = False
    if parser_type == "drain":
        # the hyper parameter is set according to http://jmzhu.logpai.com/pub/pjhe_icws2017.pdf
        st = 0.3  # Similarity threshold
        depth = 3  # Depth of all leaf nodes
        parser = Drain(
            log_format,
            indir=input_dir,
            outdir=output_dir,
            depth=depth,
            st=st,
            rex=regex,
            keep_para=keep_para,
        )
        parser.parse("buffer.csv")
    elif parser_type == "spell":
        tau = 0.55
        parser = Spell(
            indir=input_dir,
            outdir=output_dir,
            log_format=log_format,
            tau=tau,
            rex=regex,
            keep_para=keep_para,
        )
        parser.parse("buffer.csv")

    # リネーム
    os.rename(
        f"{output_dir}/buffer.csv_structured.csv",
        f"{output_dir}/{logfile_name}_structured.csv"
    )
    os.rename(
        f"{output_dir}/buffer.csv_templates.csv",
        f"{output_dir}/{logfile_name}_templates.csv"
    )
    os.remove(f"{output_dir}/buffer.csv")

# -------------------------------------------- ここからアノテーション作業関連 ----------------------------------------------------#
def anotate_csv(
    csv_filepath:Path,
    ano_df:pd.DataFrame,
    output_dir:Path,
) -> pd.DataFrame:
    """
    csvにアノテーションを行う関数。
    """

    data = pd.read_csv(csv_filepath)
    # Securityのみ抽出
    ano_df = ano_df[ano_df["Channel"] == 'Sec']

    # Label カラム作成
    data["Label"] = "-"
    mask = data["EventRecordID"].isin(ano_df["EventRecordID"])
    data.loc[mask, "Label"] = "anomaly"

    output_dir.mkdir(exist_ok=True)
    filename = csv_filepath.stem
    data.to_csv(output_dir / f"{filename}.csv", index=False)
    
    return data
    

# --------------------------------------------- ここから モデル前データ作成 関連--------------------------------------------------#

def sliding_window(
    raw_data: pd.DataFrame,
    para: dict,
    mode: str,  # "time" or "fixed"
) -> pd.DataFrame:
    """
    Split logs into sliding windows.

    Parameters
    ----------
    raw_data : pd.DataFrame
        例）columns = [timestamp, Label, EventID, EventId, deltaT] など
        少なくとも ["timestamp", "Label", "deltaT"] を含むこと。

    para : dict
        mode="time" の場合:
            {
                "window_size": float,  # 1ウィンドウの時間幅 [秒]
                "step_size"  : float,  # ウィンドウを進める時間 [秒]
            }
        mode="fixed" の場合:
            {
                "window_size": int,    # 1ウィンドウに含めるイベント数（固定長）
                "step_size"  : int,    # 次のウィンドウへ進むときに何イベントずらすか
            }

    mode : str
        "time"  : 時間ベースのスライディングウィンドウ
        "fixed" : イベント数ベース（固定長）のスライディングウィンドウ

    Returns
    -------
    pd.DataFrame
        columns = raw_data.columns
        各セルには配列（timestamp列は時刻配列、Labelはウィンドウラベル、など）が入る。
    """
    if raw_data.shape[0] == 0:
        raise ValueError("raw_data is empty")

    # 必須列チェック
    required_cols = ["timestamp", "Label", "deltaT"]
    for c in required_cols:
        if c not in raw_data.columns:
            raise ValueError(f"required column '{c}' not found in raw_data")

    # 共通で使う列
    time_data   = raw_data["timestamp"]
    label_data  = raw_data["Label"]
    deltaT_data = raw_data["deltaT"]

    new_data = []

    # -----------------------------
    # 1) 時間ベースのウィンドウ
    # -----------------------------
    if mode == "time":
        window_size = float(para["window_size"])
        step_size   = float(para["step_size"])

        if window_size <= 0 or step_size <= 0:
            raise ValueError("window_size and step_size must be positive.")

        log_size = len(time_data)
        start_end_index_pair = []

        # time_data は昇順（時系列順）を前提
        start_index = 0
        num_session = 0

        while start_index < log_size:
            start_time = time_data.iloc[start_index]
            end_time   = start_time + window_size

            # end_index を時間で伸ばす
            end_index = start_index
            while end_index < log_size and time_data.iloc[end_index] < end_time:
                end_index += 1

            if start_index != end_index:
                start_end_index_pair.append((start_index, end_index))
                num_session += 1
                if num_session % 1000 == 0:
                    print(f"process {num_session} time window", end="\r")

            # 次のウィンドウの開始時刻
            next_start_time = start_time + step_size

            # next_start_time 以降の最初のインデックスを探す
            new_start_index = start_index
            while new_start_index < log_size and time_data.iloc[new_start_index] < next_start_time:
                new_start_index += 1

            # 念のため無限ループ回避（すべて同じ時刻などの変なケース）
            if new_start_index <= start_index:
                new_start_index = start_index + 1

            start_index = new_start_index

        # ウィンドウごとのデータを作成
        for start_index, end_index in start_end_index_pair:
            ts_seq    = time_data.iloc[start_index:end_index].values
            label_seq = label_data.iloc[start_index:end_index].values
            dt_seq    = deltaT_data.iloc[start_index:end_index].values.copy()
            dt_seq[0] = 0  # 先頭は0
            window_label = label_seq.max()

            # 列順 raw_data.columns に合わせて1行分を組み立てる
            row = []
            for col in raw_data.columns:
                if col == "timestamp":
                    row.append(ts_seq)
                elif col == "Label":
                    row.append(window_label)
                elif col == "deltaT":
                    row.append(dt_seq)
                else:
                    # EventID / EventId / project など、それ以外の列はそのままシーケンス化
                    row.append(raw_data[col].iloc[start_index:end_index].values)

            new_data.append(row)

        print("there are %d instances (sliding windows) in this dataset\n" % len(new_data))
        return pd.DataFrame(new_data, columns=raw_data.columns)

    # -----------------------------
    # 2) イベント数ベース（固定長）
    # -----------------------------
    elif mode == "fixed":
        window_size = int(para["window_size"])
        step_size   = int(para.get("step_size", window_size))  # 指定なければ非オーバーラップ

        if window_size <= 0 or step_size <= 0:
            raise ValueError("window_size and step_size must be positive integers")

        log_size = raw_data.shape[0]
        num_session = 0

        for start_index in range(0, log_size - window_size + 1, step_size):
            end_index = start_index + window_size

            ts_seq    = time_data.iloc[start_index:end_index].values
            label_seq = label_data.iloc[start_index:end_index].values
            dt_seq    = deltaT_data.iloc[start_index:end_index].values.copy()
            dt_seq[0] = 0
            window_label = label_seq.max()

            row = []
            for col in raw_data.columns:
                if col == "timestamp":
                    row.append(ts_seq)
                elif col == "Label":
                    row.append(window_label)
                elif col == "deltaT":
                    row.append(dt_seq)
                else:
                    row.append(raw_data[col].iloc[start_index:end_index].values)

            new_data.append(row)

            num_session += 1
            if num_session % 1000 == 0:
                print(f"process {num_session} count window", end="\r")

        print("there are %d instances (sliding windows) in this dataset\n" % num_session)
        return pd.DataFrame(new_data, columns=raw_data.columns)

    else:
        raise ValueError('mode must be either "time" or "fixed".')


def deeplog_file_generator(
    filename, 
    df, 
    features
) -> None:
    """
    データフレームを deeplog_file に変換して保存する関数。
    """
    with open(filename, "w") as f:
        for _, row in df.iterrows():
            for val in zip(*row[features]):
                f.write(",".join([str(v) for v in val]) + " ")
            f.write("\n")

def calculate_seq_length_stats(df: pd.DataFrame, seq_column: str = "EventId") -> dict:
    """
    DataFrameの指定カラムからシーケンス長統計を計算する。
    
    Parameters
    ----------
    df : pd.DataFrame
        シーケンスデータを含むDataFrame
    seq_column : str
        シーケンス（配列）が格納されているカラム名
    
    Returns
    -------
    dict
        統計情報の辞書 {"count", "avg_len", "min_len", "max_len", "std_len"}
    """
    if len(df) == 0:
        return {"count": 0, "avg_len": 0.0, "min_len": 0, "max_len": 0, "std_len": 0.0}
    
    lengths = df[seq_column].apply(lambda x: len(x) if hasattr(x, '__len__') else 0)
    return {
        "count": len(df),
        "avg_len": float(lengths.mean()),
        "min_len": int(lengths.min()),
        "max_len": int(lengths.max()),
        "std_len": float(lengths.std()),
    }


def save_seq_stats_report(
    stats_dict: dict,
    output_path: Path,
    mode: str = "fixed",
    window_size: int = 0,
    step_size: int = 0,
) -> None:
    """
    シーケンス長統計をtxtファイルに保存し、コンソールに表示する。
    
    Parameters
    ----------
    stats_dict : dict
        {ratio: {"train": stats, "test_normal": stats, "test_abnormal": stats}, ...}
    output_path : Path
        保存先のファイルパス
    mode : str
        sliding_windowのモード ("time" or "fixed")
    window_size : int
        ウィンドウサイズ
    step_size : int
        ステップサイズ
    """
    from datetime import datetime
    
    lines = []
    lines.append("=" * 70)
    lines.append("Sequence Length Statistics Report")
    lines.append("=" * 70)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Mode: {mode}")
    lines.append(f"Window Size: {window_size}")
    lines.append(f"Step Size: {step_size}")
    lines.append("=" * 70)
    
    for ratio, ratio_stats in sorted(stats_dict.items()):
        lines.append(f"\n[Ratio: {ratio}]")
        lines.append("-" * 50)
        
        for data_type, stats in ratio_stats.items():
            lines.append(f"  {data_type}:")
            lines.append(f"    Count:    {stats['count']:,}")
            lines.append(f"    Avg Len:  {stats['avg_len']:.2f}")
            lines.append(f"    Min Len:  {stats['min_len']}")
            lines.append(f"    Max Len:  {stats['max_len']}")
            lines.append(f"    Std Dev:  {stats['std_len']:.2f}")
    
    lines.append("\n" + "=" * 70)
    
    report = "\n".join(lines)
    
    # コンソールに表示
    print(report)
    
    # ファイルに保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\nStatistics saved to: {output_path}")


def prepare_model_data(
    logdata_filepath:Path,
    output_dir:Path,
    window_size:int = 300,
    step_size:int = 60,
    mode: str = "fixed", 
) -> None:
    """
    モデル前データ作成工程の親関数。
    vocabファイル作成まで行う。
    アノテーション済みのcsvを入力に想定。
    """
    output_dir.mkdir(exist_ok=True)
    
    data = pd.read_csv(logdata_filepath)
    
    #----------- 諸操作 ----------#
    if "Label" not in data.columns:
        raise ValueError("Label column not found in CSV.")
    data["Label"] = data["Label"].apply(lambda x: int(x != "-"))
    data["datetime"] = pd.to_datetime(data["TimeCreated_SystemTime"], format='mixed')
    data["timestamp"] = data["datetime"].view("int64") // 10**9  
    data["deltaT"] = data["datetime"].diff().dt.total_seconds().fillna(0)
    
    # ----------- データフレーム → モデル前データ ----------#
    # sampling with sliding window
    deeplog_df = sliding_window(
        data[["timestamp", "Label", "EventId", "deltaT"]],
        #para={"window_size": int(window_size) * 60, "step_size": int(step_size) * 60},
        para={"window_size": window_size, "step_size": step_size},
        mode = mode
    )
    
    # normalとabnormalを切り分け
    df_normal = deeplog_df[deeplog_df["Label"] == 0]
    df_abnormal = deeplog_df[deeplog_df["Label"] == 1]

    # shuffle
    df_normal = df_normal.sample(frac=1, random_state=12).reset_index(drop=True)  
    normal_len = len(df_normal)
    
    # シーケンス長統計を格納する辞書
    seq_stats = {}
    
    train_ratio_list = [0.6, 0.8]
    for train_ratio in train_ratio_list:

        train_len = int(normal_len * train_ratio)
        save_dir = output_dir/f'ratio_{train_ratio}'

        os.makedirs(save_dir, exist_ok=True)

        # train
        train = df_normal[:train_len]
        deeplog_file_generator(
            filename = str(save_dir) + '/train',
            df = train,
            features = ["EventId", "deltaT"],
        )
        print("training size {}".format(train_len))

        # test(normal)
        test_normal = df_normal[train_len:]
        deeplog_file_generator(
            filename = str(save_dir) + '/test_normal',
            df = test_normal,
            features = ["EventId", "deltaT"],
        )
        print("test normal size {}".format(normal_len - train_len))

        # abnormal
        
        # 必要なら EventId のマッピングを復活
        # df_abnormal["EventId"] = df_abnormal["EventId"].progress_apply(
        #     lambda e: event_index_map[e] if event_index_map.get(e) else UNK
        # )

        deeplog_file_generator(
            filename = str(save_dir) + '/test_abnormal',
            df = df_abnormal,
            features = ["EventId", "deltaT"], 
        )
        print("test abnormal size {}".format(len(df_abnormal)))
        
        # シーケンス長統計を計算
        seq_stats[train_ratio] = {
            "train": calculate_seq_length_stats(train, "EventId"),
            "test_normal": calculate_seq_length_stats(test_normal, "EventId"),
            "test_abnormal": calculate_seq_length_stats(df_abnormal, "EventId"),
        }
        
    # vocab 作成
    train_ratio = 1.0

    train_len = int(normal_len * train_ratio)
    save_dir = output_dir/f'vocab'
    os.makedirs(save_dir, exist_ok=True)

    # train
    train = df_normal[:train_len]
    deeplog_file_generator(
        filename = str(save_dir) + '/train',
        df = train,
        features = ["EventId"], # EventId only
    )
    print("training size {}".format(train_len))
    
    # シーケンス長統計をファイルに保存
    stats_output_path = output_dir / "seq_stats.txt"
    save_seq_stats_report(
        stats_dict=seq_stats,
        output_path=stats_output_path,
        mode=mode,
        window_size=window_size,
        step_size=step_size,
    )
        
    return

def prepare_model_train_data(
    logdata_filepath:Path,
    output_dir:Path,
    train_ratio: int,
    features: list = ["EventId", "deltaT"],
    use_columns: list = ["timestamp", "Label", "EventID", "EventId", "deltaT"],
    window_size:int = 300,
    step_size:int = 60,
    mode: str = "fixed", 
    shuffle: bool = True,
) -> None:
    """
    モデル前データ作成工程の親関数。
    vocabファイル作成は行わない。
    引数のtrain_ratioに応じて訓練データを生成。
    アノテーション済みのcsvを入力に想定。
    """
    output_dir.mkdir(exist_ok=True)
    
    data = pd.read_csv(logdata_filepath)
    
    #----------- 諸操作 ----------#
    if "Label" not in data.columns:
        raise ValueError("Label column not found in CSV.")
    data["Label"] = data["Label"].apply(lambda x: int(x != "-"))
    data["datetime"] = pd.to_datetime(data["TimeCreated_SystemTime"], format='mixed')
    data["timestamp"] = data["datetime"].view("int64") // 10**9  
    data["deltaT"] = data["datetime"].diff().dt.total_seconds().fillna(0)
    
    # ----------- データフレーム → モデル前データ ----------#
    # sampling with sliding window
    deeplog_df = sliding_window(
        data[use_columns],
        #para={"window_size": int(window_size) * 60, "step_size": int(step_size) * 60},
        para={"window_size": window_size, "step_size": step_size},
        mode = mode
    )
    
    # normalとabnormalを切り分け
    df_normal = deeplog_df[deeplog_df["Label"] == 0]
    df_abnormal = deeplog_df[deeplog_df["Label"] == 1]

    if shuffle:
        # shuffle
        df_normal = df_normal.sample(frac=1, random_state=12).reset_index(drop=True)  
        normal_len = len(df_normal)
    
    # シーケンス長統計を格納する辞書
    seq_stats = {}

    train_len = int(normal_len * train_ratio)
    save_dir = output_dir
    os.makedirs(save_dir, exist_ok=True)

    # train
    train = df_normal[:train_len]
    deeplog_file_generator(
        filename = str(save_dir) + '/train',
        df = train,
        features = features,
    )
    print("training size {}".format(train_len))
    
    # シーケンス長統計を計算
    seq_stats[train_ratio] = {
        "train": calculate_seq_length_stats(train, features),
    }
        
    # シーケンス長統計をファイルに保存
    stats_output_path = output_dir / "seq_stats.txt"
    save_seq_stats_report(
        stats_dict=seq_stats,
        output_path=stats_output_path,
        mode=mode,
        window_size=window_size,
        step_size=step_size,
    )
        
    return

def prepare_model_test_data(
    logdata_filepath:Path,
    output_dir:Path,
    train_ratio: int,
    features: list = ["EventId", "deltaT"],
    use_columns: list = ["timestamp", "Label", "EventID", "EventId", "deltaT"],
    window_size:int = 300,
    step_size:int = 60,
    mode: str = "fixed",
    shuffle: bool = True, 
) -> None:
    """
    モデル前データ作成工程の親関数。
    vocabファイル作成は行わない。
    引数のtrain_ratioに応じて訓練データを生成。
    アノテーション済みのcsvを入力に想定。
    """
    output_dir.mkdir(exist_ok=True)
    
    data = pd.read_csv(logdata_filepath)
    
    #----------- 諸操作 ----------#
    if "Label" not in data.columns:
        raise ValueError("Label column not found in CSV.")
    data["Label"] = data["Label"].apply(lambda x: int(x != "-"))
    data["datetime"] = pd.to_datetime(data["TimeCreated_SystemTime"], format='mixed')
    data["timestamp"] = data["datetime"].view("int64") // 10**9  
    data["deltaT"] = data["datetime"].diff().dt.total_seconds().fillna(0)
    
    # ----------- データフレーム → モデル前データ ----------#
    # sampling with sliding window
    deeplog_df = sliding_window(
        data[use_columns],
        #para={"window_size": int(window_size) * 60, "step_size": int(step_size) * 60},
        para={"window_size": window_size, "step_size": step_size},
        mode = mode
    )
    
    # normalとabnormalを切り分け
    df_normal = deeplog_df[deeplog_df["Label"] == 0]
    df_abnormal = deeplog_df[deeplog_df["Label"] == 1]

    if shuffle:
        # shuffle
        df_normal = df_normal.sample(frac=1, random_state=12).reset_index(drop=True)  
        normal_len = len(df_normal)
    
    # シーケンス長統計を格納する辞書
    seq_stats = {}

    train_len = int(normal_len * train_ratio)
    save_dir = output_dir
    os.makedirs(save_dir, exist_ok=True)

    # test(normal)
    test_normal = df_normal[train_len:]
    deeplog_file_generator(
        filename = str(save_dir) + '/test_normal',
        df = test_normal,
        features = features,
    )
    print("test normal size {}".format(normal_len - train_len))

    # abnormal
    deeplog_file_generator(
        filename = str(save_dir) + '/test_abnormal',
        df = df_abnormal,
        features = features, 
    )
    print("test abnormal size {}".format(len(df_abnormal)))
    
    # シーケンス長統計を計算
    seq_stats[train_ratio] = {
        "test_normal": calculate_seq_length_stats(test_normal, features),
        "test_abnormal": calculate_seq_length_stats(df_abnormal, features),
    }
    
    # シーケンス長統計をファイルに保存
    stats_output_path = output_dir / "seq_stats.txt"
    save_seq_stats_report(
        stats_dict=seq_stats,
        output_path=stats_output_path,
        mode=mode,
        window_size=window_size,
        step_size=step_size,
    )
        
    return


def prepare_integrated_model_data(
    logdata_filepath:Path,
    output_dir:Path,
    window_size:int = 300,
    step_size:int = 60,
    mode: str = "time", 
) -> None:
    """
    統合データ用。
    モデル前データ作成工程の親関数。
    vocabファイル作成まで行う。
    """
    output_dir.mkdir(exist_ok=True)
    
    data = pd.read_csv(logdata_filepath)
    
    #----------- 諸操作 ----------#
    if "Label" not in data.columns:
        raise ValueError("Label column not found in CSV.")
    data["Label"] = data["Label"].apply(lambda x: int(x != "-"))
    data["datetime"] = pd.to_datetime(data["TimeCreated_SystemTime"], format='mixed')
    data["timestamp"] = data["datetime"].view("int64") // 10**9  
    data["deltaT"] = data["datetime"].diff().dt.total_seconds().fillna(0)
    
    # ----------- データフレーム → モデル前データ ----------#
    
    # ratio = 1.0 はvocab作成用
    train_ratio_list = [0.8, 1.0]
    
    # シーケンス長統計を格納する辞書
    seq_stats = {}
    
    project_list = data["project"].unique()

    for train_ratio in train_ratio_list:
        integrated_train = pd.DataFrame()
        df_normal = pd.DataFrame()
        
        # このratioの統計を格納
        ratio_stats = {"train": None, "test_normal_all": [], "test_abnormal_all": []}
        
        # projectごとに処理
        for project in project_list:
            # プロジェクトごとにデータをフィルタリング
            project_data = data[data["project"] == project]

            # sampling with sliding window
            deeplog_df = sliding_window(
                project_data[["timestamp", "Label", "EventId", "deltaT"]],
                para={"window_size": window_size, "step_size": step_size},
                mode=mode,
            )
            deeplog_df["project"] = project

            # 余事象データは即ち正常データなので、即座に統合
            if project.endswith("_C"):
                integrated_train = pd.concat([integrated_train, deeplog_df], ignore_index=True)
                continue

            # normalとabnormalを切り分け
            temp_normal = deeplog_df[deeplog_df["Label"] == 0]
            temp_abnormal = deeplog_df[deeplog_df["Label"] == 1]

            if(train_ratio == 1.0):
                df_normal = pd.concat([df_normal, temp_normal], ignore_index=True)
                continue

            save_dir = output_dir/f"ratio_{str(train_ratio)}"/project
            os.makedirs(save_dir, exist_ok=True)
        
            # shuffle
            temp_normal = temp_normal.sample(frac=1, random_state=12).reset_index(drop=True)  
            temp_abnormal = temp_abnormal.sample(frac=1, random_state=12).reset_index(drop=True)  
            normal_len = len(temp_normal)

            train_len = int(normal_len * train_ratio)

            # train
            train = temp_normal[:train_len]
            integrated_train = pd.concat([integrated_train, train], ignore_index=True)

            # test(normal)
            test_normal = temp_normal[train_len:]
            deeplog_file_generator(
                filename = str(save_dir) + '/test_normal',
                df = test_normal,
                features = ["EventId", "deltaT"], 
            )
            
            # test_normalの統計を収集
            ratio_stats["test_normal_all"].append(test_normal)

            # test(abnormal)
            test_abnormal = temp_abnormal
            deeplog_file_generator(
                filename = str(save_dir) + '/test_abnormal',
                df = test_abnormal,
                features = ["EventId", "deltaT"], 
            )
            
            # test_abnormalの統計を収集
            ratio_stats["test_abnormal_all"].append(test_abnormal)
            
        if(train_ratio == 1.0):
            continue

        save_dir = output_dir/f"ratio_{str(train_ratio)}"
        deeplog_file_generator(
            filename = str(save_dir) + '/train',
            df = integrated_train,
            features = ["EventId", "deltaT"], 
        )
        
        # このratioの統計を計算
        test_normal_combined = pd.concat(ratio_stats["test_normal_all"], ignore_index=True) if ratio_stats["test_normal_all"] else pd.DataFrame()
        test_abnormal_combined = pd.concat(ratio_stats["test_abnormal_all"], ignore_index=True) if ratio_stats["test_abnormal_all"] else pd.DataFrame()
        
        seq_stats[train_ratio] = {
            "train": calculate_seq_length_stats(integrated_train, "EventId"),
            "test_normal": calculate_seq_length_stats(test_normal_combined, "EventId"),
            "test_abnormal": calculate_seq_length_stats(test_abnormal_combined, "EventId"),
        }
        
    # vocab 作成
    save_dir = output_dir/'vocab'
    os.makedirs(save_dir, exist_ok=True)

    deeplog_file_generator(
        filename = str(save_dir) + '/train',
        df = df_normal,
        features = ["EventId"], # EventId only
    )
    print("vocab size {}".format(len(df_normal)))
    
    # シーケンス長統計をファイルに保存
    stats_output_path = output_dir / "seq_stats.txt"
    save_seq_stats_report(
        stats_dict=seq_stats,
        output_path=stats_output_path,
        mode=mode,
        window_size=window_size,
        step_size=step_size,
    )
    
    return

def prepare_refine_model_data(
    logdata_filepath: Path,
    output_dir: Path,
    event_id: str = "EventId",
    features: list = ["EventId", "deltaT"],
    use_columns: list = ["timestamp", "Label", "EventID", "EventId", "deltaT"],
    window_size: int = 300,
    step_size: int = 60,
    mode: str = "time",
) -> None:
    """
    refineデータ作成。
    vocabファイル作成まで行う。
    プロジェクト情報も別ファイルに保存する。
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    data = pd.read_csv(logdata_filepath)
    
    #----------- 諸操作 ----------#
    if "Label" not in data.columns:
        raise ValueError("Label column not found in CSV.")
    data["Label"] = data["Label"].apply(lambda x: int(x != "-"))
    data["datetime"] = pd.to_datetime(data["TimeCreated_SystemTime"], format='mixed')
    data["timestamp"] = data["datetime"].view("int64") // 10**9
    data["deltaT"] = data["datetime"].diff().dt.total_seconds().fillna(0)
    
    # ----------- データフレーム → モデル前データ ----------#
    train_ratio_list = [0.8, 1.0]
    
    # シーケンス長統計を格納する辞書
    seq_stats = {}
    project_list = data["project"].unique()
    
    # 全プロジェクトのウィンドウ分割データを統合
    all_windowed_data = pd.DataFrame()
    
    # projectごとにウィンドウ分割処理のみ実行
    for project in project_list:
        # プロジェクトごとにデータをフィルタリング
        project_data = data[data["project"] == project]
        
        # sampling with sliding window
        deeplog_df = sliding_window(
            project_data[use_columns],
            para={"window_size": window_size, "step_size": step_size},
            mode=mode,
        )
        deeplog_df["project"] = project
        
        # 全プロジェクトのデータを統合
        all_windowed_data = pd.concat([all_windowed_data, deeplog_df], ignore_index=True)
    
    # 統合データをnormalとabnormalに分割
    all_normal = all_windowed_data[all_windowed_data["Label"] == 0]
    all_abnormal = all_windowed_data[all_windowed_data["Label"] == 1]
    
    # train_ratioごとに処理
    for train_ratio in train_ratio_list:
        # シャッフル（random_stateを指定して再現性を確保）
        all_normal_shuffled = all_normal.sample(frac=1, random_state=12).reset_index(drop=True)
        all_abnormal_shuffled = all_abnormal.sample(frac=1, random_state=12).reset_index(drop=True)
        
        normal_len = len(all_normal_shuffled)
        train_len = int(normal_len * train_ratio)
        
        # train/test分割
        train = all_normal_shuffled[:train_len]
        test_normal = all_normal_shuffled[train_len:]
        test_abnormal = all_abnormal_shuffled
        
        if train_ratio == 1.0:
            # vocab作成用
            save_dir = output_dir / 'vocab'
            os.makedirs(save_dir, exist_ok=True)
            deeplog_file_generator(
                filename=str(save_dir) + '/train',
                df=all_normal_shuffled,
                features=[event_id]  # vocabはEventIdのみ
            )
            # プロジェクト情報を保存
            save_project_info(
                filename=str(save_dir) + '/train_projects.csv',
                df=all_normal_shuffled
            )
            print("vocab size {}".format(len(all_normal_shuffled)))
            continue
        
        # 通常のtrain_ratio (0.8など)の場合
        save_dir = output_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # trainデータ保存
        deeplog_file_generator(
            filename=str(save_dir) + '/train',
            df=train,
            features=features
        )
        # trainのプロジェクト情報を保存
        save_project_info(
            filename=str(save_dir) + '/train_projects.csv',
            df=train
        )
        
        # test_normal保存
        deeplog_file_generator(
            filename=str(save_dir) + '/test_normal',
            df=test_normal,
            features=features
        )
        # test_normalのプロジェクト情報を保存
        save_project_info(
            filename=str(save_dir) + '/test_normal_projects.csv',
            df=test_normal
        )
        
        # test_abnormal保存
        deeplog_file_generator(
            filename=str(save_dir) + '/test_abnormal',
            df=test_abnormal,
            features=features
        )
        # test_abnormalのプロジェクト情報を保存
        save_project_info(
            filename=str(save_dir) + '/test_abnormal_projects.csv',
            df=test_abnormal
        )
        
        # 統計情報を収集
        seq_stats[train_ratio] = {
            "train": calculate_seq_length_stats(train, event_id),
            "test_normal": calculate_seq_length_stats(test_normal, event_id),
            "test_abnormal": calculate_seq_length_stats(test_abnormal, event_id),
        }
    
    # シーケンス長統計をファイルに保存
    stats_output_path = output_dir / "seq_stats.txt"
    save_seq_stats_report(
        stats_dict=seq_stats,
        output_path=stats_output_path,
        mode=mode,
        window_size=window_size,
        step_size=step_size,
    )
    
    return

#def eventid2embeddings():

    

def prepare_scores_model_data(
    output_dir: Path = PROCESSED_DIR / "scores",
    window_size: int = 300,
    step_size: int = 60,
    mode: str = "time",
) -> None:
    """
    scoreデータ作成。
    プロジェクト情報も別ファイルに保存する。
    """

    projects = ["T1105(full)", "WEB1", "WEB2"]
    train_ratio = {"T1105(full)": 0.94, "WEB1": 0.7, "WEB2": 0.7}
    split = ["train", "test"]
    use_columns = ["timestamp","deltaT","Label", "EventID", "score_missing", "mahalanobis_score"]
    features = ["EventID", "mahalanobis_score", "score_missing"] # 順序！！！

    # シーケンス長統計を格納する辞書
    seq_stats = {}

    # 全プロジェクトのウィンドウ分割データを統合
    all_windowed_data_train = pd.DataFrame()
    all_windowed_data_test = pd.DataFrame()

    for project in projects:
        # --- score 側の読み込み ---
        dfs = []
        for s in split:
            input_root = Path(f"../scores/{project}/{s}")
            for input_path in input_root.rglob("event_*.csv"):
                tmp = pd.read_csv(input_path)
                dfs.append(tmp)
        df = pd.concat(dfs, ignore_index=True)
        df = df.sort_values(by="Number", ascending=True) # 昇順

        # --- root 側 ---
        df_root = pd.read_csv(INTERIM_DIR/f"{project}/security2.csv")
        df_new = df_root[["TimeCreated_SystemTime", "Label", "EventID"]]
        df_new = df_new.rename(columns={"TimeCreated_SystemTime": "Timestamp"})
        df_new["Number"] = df_new.index
        # 初期化
        df_new["mahalanobis_score"] = 0.0
        df_new["knn_score"] = 0.0
        df_new["score_missing"] = 1

        # ----------------------- 統合 -----------------------
        score_df = df[["Number", "Timestamp", "mahalanobis_score", "knn_score"]].copy()
        score_df["Number"] = score_df["Number"].astype(int)
        df_new = df_new.merge(
            score_df,
            on="Number",
            how="left",
            suffixes=("", "_src")
        )
        # ===== Timestamp 整合性チェック =====
        # score が存在する行のみ検証
        mask_score_present = df_new["Timestamp_src"].notna()
        # 不一致検出
        timestamp_mismatch = (
            mask_score_present
            & (df_new["Timestamp"] != df_new["Timestamp_src"])
        )
        if timestamp_mismatch.any():
            bad_rows = df_new.loc[timestamp_mismatch, ["Number", "Timestamp", "Timestamp_src"]]
            raise ValueError(
                f"Timestamp mismatch detected after score merge.\n"
                f"Examples:\n{bad_rows.head(5)}"
            )
        
        df_new["mahalanobis_score"] = df_new["mahalanobis_score_src"].fillna(
            df_new["mahalanobis_score"]
        )
        df_new["knn_score"] = df_new["knn_score_src"].fillna(
            df_new["knn_score"]
        )
        # score が存在した行は missing=0
        mask_present = df_new["mahalanobis_score_src"].notna() | df_new["knn_score_src"].notna()
        df_new.loc[mask_present, "score_missing"] = 0

        # 後始末
        df_new = df_new.drop(columns=["mahalanobis_score_src", "knn_score_src"])

        # ------- 諸操作 -------
        df_new["Label"] = df_new["Label"].apply(lambda x: int(x != "-"))
        df_new["datetime"] = pd.to_datetime(df_new["Timestamp"], errors="coerce")
        df_new["timestamp"] = df_new["datetime"].astype("int64") // 10**9
        df_new["deltaT"] = df_new["datetime"].diff().dt.total_seconds().fillna(0)

        # train/test 分割()
        train_len = int(len(df_new) * train_ratio[project])
        train = df_new[:train_len]
        test = df_new[train_len:]

        for name, data in [("train", train), ("test", test)]:
            # sampling with sliding window
            deeplog_df = sliding_window(
                data[use_columns],
                para={"window_size": window_size, "step_size": step_size},
                mode=mode,
            )
            deeplog_df["project"] = project

            if name == "train":
                all_windowed_data_train = pd.concat([all_windowed_data_train, deeplog_df], ignore_index=True)
            else:
                all_windowed_data_test = pd.concat([all_windowed_data_test, deeplog_df], ignore_index=True)

    all_windowed_data_test_normal = all_windowed_data_test[all_windowed_data_test["Label"] == 0]
    all_windowed_data_test_abnormal = all_windowed_data_test[all_windowed_data_test["Label"] == 1]

    save_dir = output_dir / mode
    save_dir.mkdir(exist_ok=True, parents=True)

    # trainデータ保存
    deeplog_file_generator(
        filename=str(save_dir) + '/train',
        df=all_windowed_data_train,
        features=features
    )
    # trainのプロジェクト情報を保存
    save_project_info(
        filename=str(save_dir) + '/train_projects.csv',
        df=all_windowed_data_train
    )
    
    # test_normal保存
    deeplog_file_generator(
        filename=str(save_dir) + '/test_normal',
        df=all_windowed_data_test_normal,
        features=features
    )
    # test_normalのプロジェクト情報を保存
    save_project_info(
        filename=str(save_dir) + '/test_normal_projects.csv',
        df=all_windowed_data_test_normal
    )
    
    # test_abnormal保存
    deeplog_file_generator(
        filename=str(save_dir) + '/test_abnormal',
        df=all_windowed_data_test_abnormal,
        features=features
    )
    # test_abnormalのプロジェクト情報を保存
    save_project_info(
        filename=str(save_dir) + '/test_abnormal_projects.csv',
        df=all_windowed_data_test_abnormal
    )
    
    # 統計情報を収集
    # seq_stats = {
    #     "train": calculate_seq_length_stats(all_windowed_data_train, event_id),
    #     "test_normal": calculate_seq_length_stats(all_windowed_data_test_normal, event_id),
    #     "test_abnormal": calculate_seq_length_stats(all_windowed_data_test_abnormal, event_id),
    # }

    # シーケンス長統計をファイルに保存
    # stats_output_path = output_dir / "seq_stats.txt"
    # save_seq_stats_report(
    #     stats_dict=seq_stats,
    #     output_path=stats_output_path,
    #     mode=mode,
    #     window_size=window_size,
    #     step_size=step_size,
    # )

    return


def save_project_info(filename: str, df: pd.DataFrame) -> None:
    """
    各行に対応するプロジェクト情報をCSVファイルに保存する。
    
    Args:
        filename: 保存先ファイル名
        df: プロジェクト情報を含むデータフレーム
    """
    # プロジェクト情報とインデックスを保存
    project_info = df[["project"]].copy()
    project_info.reset_index(drop=True, inplace=True)
    project_info.to_csv(filename, index=True, header=True)
    print(f"Project info saved to {filename}")
    
    
def prepare_bgl_model_data(
    logdata_filepath: Path,
    output_dir: Path,
    features: list = ["EventId", "deltaT"],
    window_size: int = 300,
    step_size: int = 60,
    mode: str = "time",
) -> None:
    """
    BGLデータ用。
    vocabファイル作成まで行う。
    プロジェクト情報も別ファイルに保存する。
    """
    output_dir.mkdir(exist_ok=True)
    data = pd.read_csv(logdata_filepath)
    
    #----------- 諸操作 ----------#
    if "Label" not in data.columns:
        raise ValueError("Label column not found in CSV.")
    
    # data preprocess
    data['datetime'] = pd.to_datetime(data['Time'], format='%Y-%m-%d-%H.%M.%S.%f')
    data["Label"] = data["Label"].apply(lambda x: int(x != "-"))
    data['timestamp'] = data["datetime"].values.astype(np.int64) // 10 ** 9
    data['deltaT'] = data['datetime'].diff() / np.timedelta64(1, 's')
    data['deltaT'].fillna(0, inplace=True)
    
    # ----------- データフレーム → モデル前データ ----------#
    train_ratio_list = [0.8, 1.0]
    
    # シーケンス長統計を格納する辞書
    seq_stats = {}
        
    # sampling with sliding window
    deeplog_df = sliding_window(
        data[["timestamp", "Label", "EventId", "deltaT"]],
        para={"window_size": window_size, "step_size": step_size},
        mode=mode,
    )
    
    # 統合データをnormalとabnormalに分割
    df_normal = deeplog_df[deeplog_df["Label"] == 0]
    df_abnormal = deeplog_df[deeplog_df["Label"] == 1]
    
    # train_ratioごとに処理
    for train_ratio in train_ratio_list:
        # シャッフル（random_stateを指定して再現性を確保）
        df_normal_shuffled = df_normal.sample(frac=1, random_state=12).reset_index(drop=True)
        df_abnormal_shuffled = df_abnormal.sample(frac=1, random_state=12).reset_index(drop=True)
        
        normal_len = len(df_normal_shuffled)
        train_len = int(normal_len * train_ratio)
        
        # train/test分割
        train = df_normal_shuffled[:train_len]
        test_normal = df_normal_shuffled[train_len:]
        test_abnormal = df_abnormal_shuffled
        
        if train_ratio == 1.0:
            # vocab作成用
            save_dir = output_dir / 'vocab'
            os.makedirs(save_dir, exist_ok=True)
            
            # Train (vocab) - EventIdのみで語彙を構築
            deeplog_file_generator(
                filename=str(save_dir) + '/train',
                df=df_normal_shuffled,
                features=["EventId"]  # vocabはEventIdのみ
            )
            print("vocab size {}".format(len(df_normal_shuffled)))
            continue
            
        # 通常のtrain_ratio (0.8など)の場合
        save_dir = output_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Train 
        deeplog_file_generator(
            filename=str(save_dir) + '/train',
            df=train,
            features=features
        )
        
        # Test Normal 
        deeplog_file_generator(
            filename=str(save_dir) + '/test_normal',
            df=test_normal,
            features=features
        )
        
        # Test Abnormal 
        deeplog_file_generator(
            filename=str(save_dir) + '/test_abnormal',
            df=test_abnormal,
            features=features
        )
        
        # 統計情報を収集
        seq_stats[train_ratio] = {
            "train": calculate_seq_length_stats(train, "EventId"),
            "test_normal": calculate_seq_length_stats(test_normal, "EventId"),
            "test_abnormal": calculate_seq_length_stats(test_abnormal, "EventId"),
        }
    
    # シーケンス長統計をファイルに保存
    stats_output_path = output_dir / "seq_stats.txt"
    save_seq_stats_report(
        stats_dict=seq_stats,
        output_path=stats_output_path,
        mode=mode,
        window_size=window_size,
        step_size=step_size,
    )
    
    return


def prepare_tbird_model_data(
    logdata_filepath: Path,
    output_dir: Path,
    features: list = ["EventId", "deltaT"],
    window_size: int = 300,
    step_size: int = 60,
    mode: str = "time",
) -> None:
    """
    BGLデータ用。
    vocabファイル作成まで行う。
    プロジェクト情報も別ファイルに保存する。
    """
    output_dir.mkdir(exist_ok=True)
    data = pd.read_csv(logdata_filepath)
    
    #----------- 諸操作 ----------#
    if "Label" not in data.columns:
        raise ValueError("Label column not found in CSV.")
    
    # data preprocess
    data['datetime'] = pd.to_datetime(data["Date"] + " " + data['Time'], format='%Y.%m.%d %H:%M:%S')
    data["Label"] = data["Label"].apply(lambda x: int(x != "-"))
    data['timestamp'] = data["datetime"].values.astype(np.int64) // 10 ** 9
    data['deltaT'] = data['datetime'].diff() / np.timedelta64(1, 's')
    data['deltaT'].fillna(0, inplace=True)

    
    # ----------- データフレーム → モデル前データ ----------#
    train_ratio_list = [0.8, 1.0]
    
    # シーケンス長統計を格納する辞書
    seq_stats = {}
        
    # sampling with sliding window
    deeplog_df = sliding_window(
        data[["timestamp", "Label", "EventId", "deltaT"]],
        para={"window_size": window_size, "step_size": step_size},
        mode=mode,
    )
    
    # 統合データをnormalとabnormalに分割
    df_normal = deeplog_df[deeplog_df["Label"] == 0]
    df_abnormal = deeplog_df[deeplog_df["Label"] == 1]
    
    # train_ratioごとに処理
    for train_ratio in train_ratio_list:
        # シャッフル（random_stateを指定して再現性を確保）
        df_normal_shuffled = df_normal.sample(frac=1, random_state=12).reset_index(drop=True)
        df_abnormal_shuffled = df_abnormal.sample(frac=1, random_state=12).reset_index(drop=True)
        
        normal_len = len(df_normal_shuffled)
        train_len = int(normal_len * train_ratio)
        
        # train/test分割
        train = df_normal_shuffled[:train_len]
        test_normal = df_normal_shuffled[train_len:]
        test_abnormal = df_abnormal_shuffled
        
        if train_ratio == 1.0:
            # vocab作成用
            save_dir = output_dir / 'vocab'
            os.makedirs(save_dir, exist_ok=True)
            
            # Train (vocab) - EventIdのみで語彙を構築
            deeplog_file_generator(
                filename=str(save_dir) + '/train',
                df=df_normal_shuffled,
                features=["EventId"]  # vocabはEventIdのみ
            )
            print("vocab size {}".format(len(df_normal_shuffled)))
            continue
            
        # 通常のtrain_ratio (0.8など)の場合
        save_dir = output_dir / f"ratio_{str(train_ratio)}"
        os.makedirs(save_dir, exist_ok=True)
        
        # Train 
        deeplog_file_generator(
            filename=str(save_dir) + '/train',
            df=train,
            features=features
        )
        
        # Test Normal 
        deeplog_file_generator(
            filename=str(save_dir) + '/test_normal',
            df=test_normal,
            features=features
        )
        
        # Test Abnormal 
        deeplog_file_generator(
            filename=str(save_dir) + '/test_abnormal',
            df=test_abnormal,
            features=features
        )
        
        # 統計情報を収集
        seq_stats[train_ratio] = {
            "train": calculate_seq_length_stats(train, "EventId"),
            "test_normal": calculate_seq_length_stats(test_normal, "EventId"),
            "test_abnormal": calculate_seq_length_stats(test_abnormal, "EventId"),
        }
    
    # シーケンス長統計をファイルに保存
    stats_output_path = output_dir / "seq_stats.txt"
    save_seq_stats_report(
        stats_dict=seq_stats,
        output_path=stats_output_path,
        mode=mode,
        window_size=window_size,
        step_size=step_size,
    )
    
    return

def extract_line(
    input_file: Path,
    output_file: Path,
    num: int = 5000,
)-> None:
    
    with open(input_file, "r", encoding="utf-8") as fin, \
        open(output_file, "w", encoding="utf-8") as fout:
        for line in islice(fin, num):
            fout.write(line)

    

# ---------------------------------------------------- 余談部分 -----------------------------------------------------#
def _stratified_sample_one(
    df: pd.DataFrame,
    target_n: int,
    event_id_col: str = "EventID",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    単一の DataFrame から、EventID ごとに層別抽出して target_n 件サンプリングする。
    すべての EventID から最低 1 件は含めることを保証する版。
    """
    df = df.dropna(subset=[event_id_col]).copy()
    df[event_id_col] = df[event_id_col].astype(str)

    counts = df[event_id_col].value_counts()
    event_ids = counts.index
    K = len(event_ids)

    if target_n < K:
        raise ValueError(
            f"target_n={target_n} ではユニーク EventID 数 {K} に対して "
            "各 EventID 最低1件を保証できません。"
        )

    # ① まず各 EventID に 1 件ずつ割り当て（最低保証）
    alloc = pd.Series(1, index=event_ids, dtype=int)

    # 残りを出現頻度に応じて配分
    remaining = target_n - K
    if remaining > 0:
        extra = (counts / counts.sum() * remaining).round().astype(int)
        alloc += extra

        # 合計が target_n からずれたら微調整（>=1 を維持しつつ）
        diff = target_n - alloc.sum()
        if diff != 0:
            # 出現数の多い順
            sorted_ids = counts.index
            step = 1 if diff > 0 else -1
            diff_abs = abs(diff)
            i = 0
            while diff_abs > 0:
                eid = sorted_ids[i % len(sorted_ids)]
                # 減らす場合は 1 未満にはしない
                if step < 0 and alloc[eid] <= 1:
                    i += 1
                    continue
                alloc[eid] += step
                diff_abs -= 1
                i += 1

    # ② 実際にサンプリング
    sampled_list = []
    for event_id, n in alloc.items():
        group = df[df[event_id_col] == event_id]
        n_actual = min(n, len(group))
        if n_actual <= 0:
            continue
        # group が少なすぎて n_actual < 1 になりうるケースへの対処を入れるならここ
        sampled = group.sample(n=n_actual, random_state=random_state)
        sampled_list.append(sampled)

    sampled_df = pd.concat(sampled_list, ignore_index=False)

    # ここでは基本的に len(sampled_df) == target_n になる想定。
    # もし多少ずれるのを許容するなら、そのままでもOK。
    return sampled_df


def stratified_sample_by_eventid_two_sets(
    input_files: List[Union[str, Path]],
    output_file1: Union[str, Path],
    output_file2: Union[str, Path],
    target_n_each: int = 5000,
    event_id_col: str = "EventID",
    random_state: int = 42,
):
    """
    複数のイベントログ CSV を結合し、EventID ごとの層別抽出で
    ・サンプル1: target_n_each 件
    ・サンプル2: target_n_each 件
    の2セットを重複なしで作成し、別々の CSV に保存する。

    Parameters
    ----------
    input_files : list of str or Path
        結合する元 CSV ファイルのリスト
    output_file1 : str or Path
        1つ目のサンプルの保存パス
    output_file2 : str or Path
        2つ目のサンプルの保存パス
    target_n_each : int, default 5000
        各サンプルに含めたい件数
    event_id_col : str, default "EventID"
        EventID の列名
    random_state : int, default 42
        ランダムシード（2つ目は random_state+1 を使う）
    """

    paths = [Path(f) for f in input_files]
    dfs = [pd.read_csv(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)

    print(f"結合後の全レコード数: {len(df)}")

    # 2セット分の件数があるかチェック
    if len(df) < 2 * target_n_each:
        raise ValueError(
            f"データ数が不足しています: レコード数 {len(df)} に対して "
            f"2×{target_n_each} 件は確保できません。"
        )

    # ---- サンプル1 ----
    sampled1 = _stratified_sample_one(
        df=df,
        target_n=target_n_each,
        event_id_col=event_id_col,
        random_state=random_state,
    )
    print(f"サンプル1件数: {len(sampled1)}")

    # サンプル1を元データから除外（index ベースで削除）
    remaining_df = df.drop(index=sampled1.index)

    print(f"サンプル1除外後の残りレコード数: {len(remaining_df)}")

    # ---- サンプル2 ----
    sampled2 = _stratified_sample_one(
        df=remaining_df,
        target_n=target_n_each,
        event_id_col=event_id_col,
        random_state=random_state + 1,
    )
    print(f"サンプル2件数: {len(sampled2)}")

    # 念のためサンプル間の重複確認
    overlap = set(sampled1.index) & set(sampled2.index)
    print(f"サンプル1・2の重複インデックス数: {len(overlap)}")

    # ---- 保存 ----
    output_path1 = Path(output_file1)
    output_path2 = Path(output_file2)

    sampled1.to_csv(output_path1, index=False)
    sampled2.to_csv(output_path2, index=False)

    print(f"サンプル1を保存しました: {output_path1}")
    print(f"サンプル2を保存しました: {output_path2}")

    return sampled1, sampled2

def delete_unwanted_logs(
    input_filepath:Path,
    start_date:str,
    end_date:str,
    output_filepath:Path = None,
) -> pd.DataFrame:
    """
    指定された日付範囲のログを削除する。
    
    Parameters
    ----------
    input_filepath : Path
        元のログファイルのパス
    start_date : str
        開始日付（YYYY-MM-DD）
    end_date : str
        終了日付（YYYY-MM-DD）
    output_filepath : Path, optional
        出力ファイルのパス（デフォルト: 元のファイル）
        
    Returns
    -------
    pd.DataFrame
        削除後のデータフレーム
    """
    data = pd.read_csv(input_filepath)

    data["TimeCreated_SystemTime"] = pd.to_datetime(
        data["TimeCreated_SystemTime"], 
        format='mixed',      
    )
    data["date"] = data["TimeCreated_SystemTime"].dt.date

    filtered = data[
        (data["date"] >= pd.to_datetime(start_date).date()) &
        (data["date"] <= pd.to_datetime(end_date).date())
    ]
    # 出力ファイルが指定されていない場合、元のファイルに上書き保存
    if output_filepath is None:
        output_filepath = input_filepath

    # 結果を保存
    filtered.to_csv(output_filepath, index=False)
    return filtered
    

def calculate_equivalent_window_params(
    csv_filepath: Union[str, Path],
    source_mode: str,  # "time" or "fixed"
    source_window_size: float,
    source_step_size: float,
    target_mode: str,  # "time" or "fixed"
) -> dict:
    """
    CSVファイルを読み込み、指定されたwindow_sizeおよびstep_sizeに対して、
    平均シーケンス長を合わせるために、もう一方の手法で必要なパラメータを算出する。

    Parameters
    ----------
    csv_filepath : Union[str, Path]
        入力CSVファイルのパス。
        必須カラム: TimeCreated_SystemTime（タイムスタンプ）
        ※ T1105/security2.csv の形式を想定

    source_mode : str
        変換元の方式。"time" または "fixed"

    source_window_size : float
        変換元のwindow_size。
        - timeモード: 秒単位の時間幅
        - fixedモード: イベント数

    source_step_size : float
        変換元のstep_size。
        - timeモード: 秒単位の時間幅
        - fixedモード: イベント数

    target_mode : str
        変換先の方式。"time" または "fixed"

    Returns
    -------
    dict
        {
            "window_size": float or int,  # 変換先のwindow_size
            "step_size": float or int,    # 変換先のstep_size
            "avg_event_rate": float,      # 平均イベント発生率 (events/second)
            "avg_time_per_event": float,  # 平均イベント間隔 (seconds/event)
            "total_events": int,          # 総イベント数
            "total_duration": float,      # 総時間 (seconds)
            "source_avg_sequence_length": float,  # 変換元の平均シーケンス長
        }

    Raises
    ------
    ValueError
        source_mode と target_mode が同じ場合、または無効な値の場合
    FileNotFoundError
        CSVファイルが見つからない場合
    """
    # モードの検証
    if source_mode == target_mode:
        raise ValueError("source_mode と target_mode は異なる値を指定してください。")
    
    if source_mode not in ("time", "fixed"):
        raise ValueError("source_mode は 'time' または 'fixed' を指定してください。")
    
    if target_mode not in ("time", "fixed"):
        raise ValueError("target_mode は 'time' または 'fixed' を指定してください。")

    # CSVファイルの読み込み
    csv_filepath = Path(csv_filepath)
    if not csv_filepath.exists():
        raise FileNotFoundError(f"CSVファイルが見つかりません: {csv_filepath}")
    
    data = pd.read_csv(csv_filepath)
    
    if data.shape[0] == 0:
        raise ValueError("CSVファイルにデータがありません。")
    
    # 必須カラムの確認
    if "TimeCreated_SystemTime" not in data.columns:
        raise ValueError("TimeCreated_SystemTime カラムが見つかりません。")
    
    # タイムスタンプを変換してUNIX秒に
    data["datetime"] = pd.to_datetime(data["TimeCreated_SystemTime"], format='mixed')
    data["timestamp"] = data["datetime"].view("int64") // 10**9
    
    # 統計値の計算
    time_data = data["timestamp"]
    total_events = len(time_data)
    total_duration = float(time_data.max() - time_data.min())
    
    if total_duration <= 0:
        raise ValueError("データの時間範囲が0秒以下です。時間ベースの変換ができません。")
    
    # 平均イベント率 (events/second)
    avg_event_rate = total_events / total_duration
    # 平均イベント間隔 (seconds/event)
    avg_time_per_event = total_duration / total_events

    result = {
        "csv_filepath": str(csv_filepath),
        "avg_event_rate": avg_event_rate,
        "avg_time_per_event": avg_time_per_event,
        "total_events": total_events,
        "total_duration": total_duration,
    }

    if source_mode == "time" and target_mode == "fixed":
        # time → fixed への変換
        # timeモードでのwindow_size秒間に含まれる平均イベント数 = window_size * avg_event_rate
        source_avg_seq_len = source_window_size * avg_event_rate
        target_window_size = int(round(source_avg_seq_len))
        
        # step_sizeも同様に変換
        source_step_events = source_step_size * avg_event_rate
        target_step_size = int(round(source_step_events))
        
        # 最小値を1に制限
        target_window_size = max(1, target_window_size)
        target_step_size = max(1, target_step_size)
        
        result["window_size"] = target_window_size
        result["step_size"] = target_step_size
        result["source_avg_sequence_length"] = source_avg_seq_len
        
    elif source_mode == "fixed" and target_mode == "time":
        # fixed → time への変換
        # fixedモードでのwindow_sizeイベントに相当する時間 = window_size * avg_time_per_event
        source_avg_seq_len = source_window_size  # fixedでは固定
        target_window_size = source_window_size * avg_time_per_event
        
        # step_sizeも同様に変換
        target_step_size = source_step_size * avg_time_per_event
        
        result["window_size"] = target_window_size
        result["step_size"] = target_step_size
        result["source_avg_sequence_length"] = source_avg_seq_len

    return result


def print_equivalent_params_summary(params: dict, source_mode: str, target_mode: str) -> None:
    """
    calculate_equivalent_window_params の結果を整形して表示するユーティリティ関数。

    Parameters
    ----------
    params : dict
        calculate_equivalent_window_params の戻り値
    source_mode : str
        変換元モード
    target_mode : str
        変換先モード
    """
    print("=" * 60)
    print("Sliding Window パラメータ変換結果")
    print("=" * 60)
    print(f"入力CSV: {params.get('csv_filepath', 'N/A')}")
    print(f"変換方向: {source_mode} → {target_mode}")
    print("-" * 60)
    print("【データ統計】")
    print(f"  総イベント数:    {params['total_events']:,}")
    print(f"  総時間:          {params['total_duration']:.2f} 秒")
    print(f"  平均イベント率:  {params['avg_event_rate']:.4f} events/sec")
    print(f"  平均イベント間隔: {params['avg_time_per_event']:.4f} sec/event")
    print("-" * 60)
    print("【変換結果】")
    print(f"  推奨 window_size: {params['window_size']}")
    print(f"  推奨 step_size:   {params['step_size']}")
    print(f"  元の平均シーケンス長: {params['source_avg_sequence_length']:.2f}")
    print("=" * 60)


def shuffle_elements_per_row(
    input_dir: Path,
    output_dir: Path,
    file_names: List[str] = None,
    random_seed: int = None,
) -> None:
    """
    deeplog形式のファイルについて、各行の要素をシャッフルする。
    行間での要素の交換は行わず、各行内でのみシャッフルを実行する。
    
    Parameters
    ----------
    input_dir : Path
        入力ファイルのディレクトリパス
    output_dir : Path
        出力ファイルのディレクトリパス
    file_names : List[str], optional
        処理対象のファイル名リスト。Noneの場合はデフォルトで
        ["test_abnormal", "test_normal", "train"]を使用
    random_seed : int, optional
        乱数シード。再現性のために指定可能
    
    Notes
    -----
    ファイル形式:
        各行は「element1 element2 element3 ...」のようにスペース区切り
        各要素は「EventId,deltaT」の形式（例: "704f24ce,0.0"）
    
    Example
    -------
    >>> shuffle_elements_per_row(
    ...     input_dir=Path("data/processed/refine2/fixed"),
    ...     output_dir=Path("data/processed/refine2/shuffled"),
    ...     random_seed=42
    ... )
    """
    import random
    
    if file_names is None:
        file_names = ["test_abnormal", "test_normal", "train"]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if random_seed is not None:
        random.seed(random_seed)
    
    for file_name in file_names:
        input_path = input_dir / file_name
        output_path = output_dir / file_name
        
        if not input_path.exists():
            print(f"Warning: {input_path} does not exist. Skipping.")
            continue
        
        print(f"Processing: {input_path}")
        
        shuffled_lines = []
        line_count = 0
        
        with open(input_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Shuffling {file_name}"):
                line = line.strip()
                if not line:
                    shuffled_lines.append("")
                    continue
                
                # 行の要素をスペースで分割
                elements = line.split(" ")
                
                # 末尾の空要素を除去（行末にスペースがある場合）
                elements = [e for e in elements if e]
                
                # 行内でシャッフル
                random.shuffle(elements)
                
                # シャッフルした要素を再結合
                shuffled_line = " ".join(elements)
                shuffled_lines.append(shuffled_line)
                line_count += 1
        
        # 出力ファイルに書き込み
        with open(output_path, "w", encoding="utf-8") as f:
            for line in shuffled_lines:
                f.write(line + "\n")
        
        print(f"  Processed {line_count} lines -> {output_path}")
    
    print(f"\nShuffle complete. Output saved to: {output_dir}")


def shuffle_fixed_data(
    random_seed: int = None,
) -> None:
    """
    /home/ubuntu/My_lad/data/processed/refine2/fixed の各ファイルをシャッフルする
    便利関数。出力先は shuffled ディレクトリ。
    
    Parameters
    ----------
    random_seed : int, optional
        乱数シード
    """
    base_dir = Path("/home/ubuntu/My_lad/data/processed/refine2")
    input_dir = base_dir / "fixed"
    output_dir = base_dir / "shuffled"
    
    shuffle_elements_per_row(
        input_dir=input_dir,
        output_dir=output_dir,
        random_seed=random_seed,
    )


def merge_processed_files(
    *file_paths: Path,
    output_dir: Path,
    output_filename: str = "merged",
    mode: str = "fixed",
    window_size: int = 0,
    step_size: int = 0,
) -> None:
    """
    data/processed 下のファイル形式を複数結合し、seq_stats.txt を生成する関数。
    
    Parameters
    ----------
    *file_paths : Path
        結合するファイルパス (2つ以上, 例: data/processed/xxx/train, data/processed/xxx/test_normal)
    output_dir : Path
        出力先ディレクトリ
    output_filename : str
        出力ファイル名 (デフォルト: "merged")
    mode : str
        sliding_windowのモード (統計レポート用, デフォルト: "fixed")
    window_size : int
        ウィンドウサイズ (統計レポート用, デフォルト: 0)
    step_size : int
        ステップサイズ (統計レポート用, デフォルト: 0)
    """
    from datetime import datetime
    
    # 最低2ファイル必要
    if len(file_paths) < 2:
        raise ValueError(f"At least 2 files are required to merge. Got {len(file_paths)} file(s).")
    
    # 出力ディレクトリ作成
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ファイルの存在確認
    for file_path in file_paths:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
    
    # ファイル読み込みと結合
    merged_lines = []
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        merged_lines.extend(lines)
        print(f"Merged {len(lines)} lines from {file_path.name}")
    
    # 出力ファイルに書き込み
    output_path = output_dir / output_filename
    with open(output_path, "w", encoding="utf-8") as f_out:
        f_out.writelines(merged_lines)
    
    print(f"Total: {len(merged_lines)} lines")
    print(f"Output saved to: {output_path}")
    
    # シーケンス長統計を計算
    # 各行のトークン数（スペース区切り）をシーケンス長として扱う
    seq_lengths = []
    for line in merged_lines:
        line = line.strip()
        if line:
            tokens = line.split()
            seq_lengths.append(len(tokens))
    
    if len(seq_lengths) == 0:
        print("Warning: No valid lines found in merged file.")
        return
    
    import numpy as np
    lengths_array = np.array(seq_lengths)
    
    stats = {
        "count": len(seq_lengths),
        "avg_len": float(lengths_array.mean()),
        "min_len": int(lengths_array.min()),
        "max_len": int(lengths_array.max()),
        "std_len": float(lengths_array.std()),
    }
    
    # seq_stats dict 形式で作成（既存のsave_seq_stats_report形式に合わせる）
    seq_stats = {
        "merged": {
            output_filename: stats,
        }
    }
    
    # 統計レポートを保存
    stats_output_path = output_dir / "seq_stats.txt"
    save_seq_stats_report(
        stats_dict=seq_stats,
        output_path=stats_output_path,
        mode=mode,
        window_size=window_size,
        step_size=step_size,
    )


def extract_lines_from_processed(
    input_path: Path,
    output_dir: Path,
    output_filename: str = None,
    start_line: int = None,
    end_line: int = None,
    num_lines: int = None,
    from_head: int = None,
    from_tail: int = None,
    mode: str = "fixed",
    window_size: int = 0,
    step_size: int = 0,
) -> None:
    """
    data/processed 下のファイル形式から指定した行数を抽出して保存する関数。
    
    Parameters
    ----------
    input_path : Path
        入力ファイルパス (例: data/processed/xxx/train)
    output_dir : Path
        出力先ディレクトリ
    output_filename : str, optional
        出力ファイル名 (デフォルト: 入力ファイル名と同じ)
    start_line : int, optional
        抽出開始行 (1-indexed)
    end_line : int, optional
        抽出終了行 (1-indexed, inclusive)
    num_lines : int, optional
        start_lineから抽出する行数
    from_head : int, optional
        前端（先頭）からN行を抽出
    from_tail : int, optional
        後端（末尾）からN行を抽出
    mode : str
        sliding_windowのモード (統計レポート用, デフォルト: "fixed")
    window_size : int
        ウィンドウサイズ (統計レポート用, デフォルト: 0)
    step_size : int
        ステップサイズ (統計レポート用, デフォルト: 0)
    
    Notes
    -----
    抽出方法は以下のいずれかを指定:
    - from_head: 前端からN行
    - from_tail: 後端からN行
    - start_line + end_line: 開始行〜終了行
    - start_line + num_lines: 開始行からN行
    - 何も指定しない場合: 全行
    
    Examples
    --------
    >>> # 前端から100行を抽出
    >>> extract_lines_from_processed(
    ...     input_path=Path("data/processed/ex2/train"),
    ...     output_dir=Path("data/processed/ex2_subset"),
    ...     from_head=100,
    ... )
    
    >>> # 後端から50行を抽出
    >>> extract_lines_from_processed(
    ...     input_path=Path("data/processed/ex2/train"),
    ...     output_dir=Path("data/processed/ex2_subset"),
    ...     from_tail=50,
    ... )
    
    >>> # 1行目から100行目まで抽出
    >>> extract_lines_from_processed(
    ...     input_path=Path("data/processed/ex2/train"),
    ...     output_dir=Path("data/processed/ex2_subset"),
    ...     start_line=1,
    ...     end_line=100,
    ... )
    """
    # 出力ディレクトリ作成
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ファイルの存在確認
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")
    
    # 出力ファイル名
    if output_filename is None:
        output_filename = input_path.name
    
    # ファイル読み込み
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    print(f"Input file: {input_path}")
    print(f"Total lines in input: {total_lines}")
    
    # 引数の排他チェック
    specified_options = sum([
        from_head is not None,
        from_tail is not None,
        start_line is not None or end_line is not None or num_lines is not None,
    ])
    if specified_options > 1 and (from_head is not None or from_tail is not None):
        raise ValueError("from_head/from_tail は他の抽出オプション(start_line/end_line/num_lines)と同時に指定できません。")
    
    if end_line is not None and num_lines is not None:
        raise ValueError("end_line と num_lines の両方を指定することはできません。")
    
    # 行範囲の決定
    if from_head is not None:
        # 前端からN行
        start_idx = 0
        end_idx = min(from_head, total_lines)
        print(f"Extracting {from_head} lines from head")
    elif from_tail is not None:
        # 後端からN行
        start_idx = max(0, total_lines - from_tail)
        end_idx = total_lines
        print(f"Extracting {from_tail} lines from tail")
    else:
        # start_line/end_line/num_lines による指定
        if start_line is None:
            start_line = 1
        start_idx = start_line - 1
    
        if start_idx < 0:
            raise ValueError(f"start_line は 1 以上である必要があります: {start_line}")
        if start_idx >= total_lines:
            raise ValueError(f"start_line ({start_line}) がファイルの行数 ({total_lines}) を超えています")
        
        if end_line is not None:
            end_idx = end_line  # exclusive for slicing
            if end_idx > total_lines:
                print(f"Warning: end_line ({end_line}) がファイルの行数 ({total_lines}) を超えています。全行を抽出します。")
                end_idx = total_lines
        elif num_lines is not None:
            end_idx = start_idx + num_lines
            if end_idx > total_lines:
                print(f"Warning: 指定された行数 ({num_lines}) が残り行数を超えています。利用可能な行のみ抽出します。")
                end_idx = total_lines
        else:
            # デフォルトは全行（start_lineから最後まで）
            end_idx = total_lines
    
    # 行の抽出
    extracted_lines = lines[start_idx:end_idx]
    
    # 出力ファイルに書き込み
    output_path = output_dir / output_filename
    with open(output_path, "w", encoding="utf-8") as f_out:
        f_out.writelines(extracted_lines)
    
    print(f"Extracted lines: {start_idx + 1} to {end_idx} ({len(extracted_lines)} lines)")
    print(f"Output saved to: {output_path}")
    
    # シーケンス長統計を計算
    seq_lengths = []
    for line in extracted_lines:
        line = line.strip()
        if line:
            tokens = line.split()
            seq_lengths.append(len(tokens))
    
    if len(seq_lengths) == 0:
        print("Warning: No valid lines found in extracted file.")
        return
    
    import numpy as np
    lengths_array = np.array(seq_lengths)
    
    stats = {
        "count": len(seq_lengths),
        "avg_len": float(lengths_array.mean()),
        "min_len": int(lengths_array.min()),
        "max_len": int(lengths_array.max()),
        "std_len": float(lengths_array.std()),
    }
    
    # seq_stats dict 形式で作成
    seq_stats = {
        "extracted": {
            output_filename: stats,
        }
    }
    
    # 統計レポートを保存
    stats_output_path = output_dir / "seq_stats.txt"
    save_seq_stats_report(
        stats_dict=seq_stats,
        output_path=stats_output_path,
        mode=mode,
        window_size=window_size,
        step_size=step_size,
    )

