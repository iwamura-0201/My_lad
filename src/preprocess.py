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
    mode: str,  # "time" or "count"
) -> pd.DataFrame:
    """
    Split logs into sliding windows.

    Parameters
    ----------
    raw_data : pd.DataFrame
        columns = [timestamp, label, eventid, time duration]
        ※列順は元コードと同じ想定

    para : dict
        mode="time" の場合:
            {
                "window_size": float,  # 1ウィンドウの時間幅 [秒]
                "step_size"  : float,  # ウィンドウを進める時間 [秒]
            }
        mode="count" の場合:
            {
                "window_size": int,    # 1ウィンドウに含めるイベント数（固定長）
                "step_size"  : int,    # 次のウィンドウへ進むときに何イベントずらすか
            }

    mode : str
        "time"  : 時間ベースのスライディングウィンドウ
        "count" : イベント数ベース（固定長）のスライディングウィンドウ

    Returns
    -------
    pd.DataFrame
        columns = raw_data.columns
        各セルには配列（timestamp列は時刻配列、labelはウィンドウラベル、など）が入る。
    """
    if raw_data.shape[0] == 0:
        raise ValueError("raw_data is empty")

    # 列の取り方は元コードと同じ
    time_data   = raw_data.iloc[:, 0]
    label_data  = raw_data.iloc[:, 1]
    logkey_data = raw_data.iloc[:, 2]
    deltaT_data = raw_data.iloc[:, 3]

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
            ts_seq    = time_data[start_index:end_index].values
            label_seq = label_data[start_index:end_index].values
            key_seq   = logkey_data[start_index:end_index].values
            dt_seq    = deltaT_data[start_index:end_index].values.copy()
            dt_seq[0] = 0  # 先頭は0

            window_label = label_seq.max()

            new_data.append([
                ts_seq,
                window_label,
                key_seq,
                dt_seq,
            ])

        print(
            "there are %d instances (sliding windows) in this dataset\n"
            % len(new_data)
        )
        return pd.DataFrame(new_data, columns=raw_data.columns)

    # -----------------------------
    # 2) イベント数ベース（固定長）
    # -----------------------------
    elif mode == "count":
        window_size = int(para["window_size"])
        step_size   = int(para.get("step_size", window_size))  # 指定なければ非オーバーラップ

        if window_size <= 0 or step_size <= 0:
            raise ValueError("window_size and step_size must be positive integers")

        log_size = raw_data.shape[0]
        num_session = 0

        for start_index in range(0, log_size - window_size + 1, step_size):
            end_index = start_index + window_size

            ts_seq    = time_data[start_index:end_index].values
            label_seq = label_data[start_index:end_index].values
            key_seq   = logkey_data[start_index:end_index].values
            dt_seq    = deltaT_data[start_index:end_index].values.copy()
            dt_seq[0] = 0

            window_label = label_seq.max()

            new_data.append([
                ts_seq,
                window_label,
                key_seq,
                dt_seq,
            ])

            num_session += 1
            if num_session % 1000 == 0:
                print(f"process {num_session} count window", end="\r")

        print(
            "there are %d instances (sliding windows) in this dataset\n"
            % num_session
        )
        return pd.DataFrame(new_data, columns=raw_data.columns)

    else:
        raise ValueError("mode must be either 'time' or 'count'.")


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

def prepare_model_data(
    logdata_filepath:Path,
    output_dir:Path,
    window_size:int = 300,
    step_size:int = 60,
    mode: str = "time", 
) -> None:
    """
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
        
    return

def prepare_integrated_model_data(
    logdata_filepath:Path,
    output_dir:Path,
    project_list:list,
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
    train_ratio_list = [0.6, 0.8, 1.0]

    for train_ratio in train_ratio_list:
        integrated_train = pd.DataFrame()
        df_normal = pd.DataFrame()
        
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

            # test(abnormal)
            test_abnormal = temp_abnormal
            deeplog_file_generator(
                filename = str(save_dir) + '/test_abnormal',
                df = test_abnormal,
                features = ["EventId", "deltaT"], 
            )
        if(train_ratio == 1.0):
            continue

        save_dir = output_dir/f"ratio_{str(train_ratio)}"
        deeplog_file_generator(
            filename = str(save_dir) + '/train',
            df = integrated_train,
            features = ["EventId", "deltaT"], 
        )
        
    # vocab 作成
    save_dir = output_dir/'vocab'
    os.makedirs(save_dir, exist_ok=True)

    deeplog_file_generator(
        filename = str(save_dir) + '/train',
        df = df_normal,
        features = ["EventId"], # EventId only
    )
    print("vocab size {}".format(len(df_normal)))
    return
    

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
    