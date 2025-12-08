import sys

sys.path.append("../")

import os
import gc
import pandas as pd
import numpy as np


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from logparser.Drain import LogParser as OriDrain
from logparser.Spell import LogParser as Spell

import csv
from Evtx.Evtx import Evtx
from xml.etree.ElementTree import fromstring, ElementTree
from tqdm import tqdm
import io  # StringIOを使うために必要


class Drain(OriDrain):
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
        CSVファイルをDataFrameに変換し、LineIdを追加する
        """
        try:
            # CSVファイルを直接読み込む
            logdf = pd.read_csv(log_file, names=headers, header=0, encoding="utf-8")

            # デバッグ用: 読み込んだデータを表示
            print("Initial DataFrame:")
            print(logdf.head())

            # LineIdを追加（1から始まる連番）
            logdf.insert(0, "LineId", logdf.index + 1)

            # 総データ数を出力
            print("Total size after reading CSV:", len(logdf))

            return logdf
        except Exception as e:
            # エラー発生時の処理
            print(f"Failed to read CSV file {log_file}: {str(e)}")
            return pd.DataFrame(columns=headers)  # 空のDataFrameを返す


def sliding_window(raw_data, para):
    """
    split logs into sliding windows/session
    :param raw_data: dataframe columns=[timestamp, label, eventid, time duration]
    :param para:{window_size: seconds, step_size: seconds}
    :return: dataframe columns=[eventids, time durations, label]
    """
    log_size = raw_data.shape[0]
    label_data, time_data = raw_data.iloc[:, 1], raw_data.iloc[:, 0]
    logkey_data, deltaT_data = raw_data.iloc[:, 2], raw_data.iloc[:, 3]
    new_data = []
    start_end_index_pair = set()

    start_time = time_data[0]
    end_time = start_time + para["window_size"]
    start_index = 0
    end_index = 0

    # get the first start, end index, end time
    for cur_time in time_data:
        if cur_time < end_time:
            end_index += 1
        else:
            break

    start_end_index_pair.add(tuple([start_index, end_index]))

    # move the start and end index until next sliding window
    num_session = 1
    while end_index < log_size:
        start_time = start_time + para["step_size"]
        end_time = start_time + para["window_size"]
        for i in range(start_index, log_size):
            if time_data[i] < start_time:
                i += 1
            else:
                break
        for j in range(end_index, log_size):
            if time_data[j] < end_time:
                j += 1
            else:
                break
        start_index = i
        end_index = j

        # when start_index == end_index, there is no value in the window
        if start_index != end_index:
            start_end_index_pair.add(tuple([start_index, end_index]))

        num_session += 1
        if num_session % 1000 == 0:
            print("process {} time window".format(num_session), end="\r")

    for start_index, end_index in start_end_index_pair:
        dt = deltaT_data[start_index:end_index].values
        dt[0] = 0
        new_data.append(
            [
                time_data[start_index:end_index].values,
                max(label_data[start_index:end_index]),
                logkey_data[start_index:end_index].values,
                dt,
            ]
        )

    assert len(start_end_index_pair) == len(new_data)
    print(
        "there are %d instances (sliding windows) in this dataset\n"
        % len(start_end_index_pair)
    )
    return pd.DataFrame(new_data, columns=raw_data.columns)


tqdm.pandas()
pd.options.mode.chained_assignment = None

PAD = 0
UNK = 1
START = 2
data_dir = os.path.expanduser("/home/ssakurai/project/log/dataset/amiya/WEB2/")
output_dir = "/home/ssakurai/project/log/dataset/preprocessed/WEB2_3/"
log_file = "WEB2_3.csv"


# In the first column of the log, "-" indicates non-alert messages while others are alert messages.
def count_anomaly():
    total_size = 0
    normal_size = 0
    with open(data_dir + log_file, encoding="utf8") as f:
        for line in f:
            total_size += 1
            if line.split(" ", 1)[0] == "-":
                normal_size += 1
    print(
        "total size {}, abnormal size {}".format(total_size, total_size - normal_size)
    )


def deeplog_file_generator(filename, df, features):
    with open(filename, "w") as f:
        for _, row in df.iterrows():
            for val in zip(*row[features]):
                f.write(",".join([str(v) for v in val]) + " ")
            f.write("\n")


def parse_log(input_dir, output_dir, log_file, parser_type):
    log_format = "<Content> <TimeCreated> <EventID> <Label>"
    regex = [
        r"(0x)[0-9a-fA-F]+",  # hexadecimal
        r"\d+.\d+.\d+.\d+",
        # r'/\w+( )$'
        r"\d+",
    ]
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
        parser.parse(log_file)
    elif parser_type == "spell":
        tau = 0.55
        parser = Spell(
            indir=data_dir,
            outdir=output_dir,
            log_format=log_format,
            tau=tau,
            rex=regex,
            keep_para=keep_para,
        )
        parser.parse(log_file)


def prepre(csv_path):

    # 複数の検索対象行をリストとして定義

    data = pd.read_csv(
        "/home/ssakurai/project/log/dataset/amiya/WEB2/filtered_output.csv"
    )

    # 一致した行の行番号を保存するリスト
    matched_indices = [
        333111,
        333946,
        333950,
        333952,
        334496,
        334498,
        334499,
        335004,
        335124,
        335155,
        335281,
        335285,
    ]

    # 行番号を表示
    data["Label"] = "-"
    data.loc[matched_indices, "Label"] = "anomaly"
    data = data.sort_values(by="TimeCreated")
    data["Content"] = "Content" + data["Content"].astype("str")

    data.to_csv(csv_path)


if __name__ == "__main__":
    # csv_path = f"{data_dir}{log_file}"

    # prepre(csv_path)

    # parse_log(data_dir, output_dir, log_file, "drain")

    # #########
    # # Count #
    # #########
    # count_anomaly()

    # ##################
    # # Transformation #
    # ##################
    # #     # mins
    window_size = 5
    step_size = 1
    train_ratio = 0.4

    df = pd.read_csv(f"{output_dir}{log_file}_structured.csv")
    df["TimeCreated"] = df["TimeCreated"].str.pad(width=26, side="right", fillchar="0")

    #     # data preprocess
    df["datetime"] = pd.to_datetime(
        df["TimeCreated"], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce"
    )
    df["Label"] = df["Label"].apply(lambda x: int(x != "-"))
    df["timestamp"] = df["datetime"].values.astype(np.int64) // 10**9
    df["deltaT"] = df["datetime"].diff() / np.timedelta64(1, "s")
    df["deltaT"] = df["deltaT"].fillna(0)
    #     # convert time to UTC timestamp
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize("UTC")
    df["deltaT"] = df["datetime"].apply(
        lambda t: (t - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1s")
    )

    # sampling with fixed window
    # features = ["EventId", "deltaT"]
    # target = "Label"
    # deeplog_df = deeplog_df_transfer(df, features, target, "datetime", window_size=args.w)
    # deeplog_df.dropna(subset=[target], inplace=True)

    # sampling with sliding window
    deeplog_df = sliding_window(
        df[["timestamp", "Label", "EventId", "deltaT"]],
        para={"window_size": int(window_size) * 60, "step_size": int(step_size) * 60},
    )

    #     #########
    #     # Train #
    #     #########
    df_normal = deeplog_df[deeplog_df["Label"] == 0]
    df_normal = df_normal.sample(frac=1, random_state=12).reset_index(
        drop=True
    )  # shuffle
    normal_len = len(df_normal)

    #     #########
    #     # Train #
    #     #########

    train_ratio = 0.4
    train_len = int(normal_len * train_ratio)

    train = df_normal[:train_len]
    deeplog_file_generator(os.path.join(output_dir, "train4"), train, ["EventId"])

    print("training size {}".format(train_len))

    #     # ###############
    #     # # Test Normal #
    #     # ###############
    test_normal = df_normal[train_len:]
    deeplog_file_generator(
        os.path.join(output_dir, "test_normal6"), test_normal, ["EventId"]
    )

    print("test normal size {}".format(normal_len - train_len))

    #     #########
    #     # Train #
    #     #########

    train_ratio = 0.6
    train_len = int(normal_len * train_ratio)

    train = df_normal[:train_len]
    deeplog_file_generator(os.path.join(output_dir, "train6"), train, ["EventId"])

    print("training size {}".format(train_len))

    #     # ###############
    #     # # Test Normal #
    #     # ###############
    test_normal = df_normal[train_len:]
    deeplog_file_generator(
        os.path.join(output_dir, "test_normal4"), test_normal, ["EventId"]
    )

    print("test normal size {}".format(normal_len - train_len))

    #     #########
    #     # Train #
    #     #########

    train_ratio = 0.8
    train_len = int(normal_len * train_ratio)

    train = df_normal[:train_len]
    deeplog_file_generator(os.path.join(output_dir, "train8"), train, ["EventId"])

    print("training size {}".format(train_len))

    #     # ###############
    #     # # Test Normal #
    #     # ###############
    test_normal = df_normal[train_len:]
    deeplog_file_generator(
        os.path.join(output_dir, "test_normal2"), test_normal, ["EventId"]
    )

    print("test normal size {}".format(normal_len - train_len))

    train_ratio = 1
    train_len = int(normal_len * train_ratio)

    train = df_normal[:train_len]
    deeplog_file_generator(os.path.join(output_dir, "train"), train, ["EventId"])

    print("training size {}".format(train_len))

    del df_normal
    del train
    # del test_normal
    gc.collect()

    # #################
    # # Test Abnormal #
    # #################
    df_abnormal = deeplog_df[deeplog_df["Label"] == 1]
    # df_abnormal["EventId"] = df_abnormal["EventId"].progress_apply(lambda e: event_index_map[e] if event_index_map.get(e) else UNK)
    deeplog_file_generator(
        os.path.join(output_dir, "test_abnormal"), df_abnormal, ["EventId"]
    )
    print("test abnormal size {}".format(len(df_abnormal)))
