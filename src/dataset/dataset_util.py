from torch.utils.data import DataLoader
import numpy as np
import gc
from vocab import WordVocab, Vocab
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from .logdataset import LogDataset
from pathlib import Path
from omegaconf import DictConfig

INTERIM_DIR = Path('../data/interim')
PROCESSED_DIR = Path('../data/processed')
RAW_DIR = Path('../data/raw')

def set_dataset_path(
    cfg: DictConfig,
    what: str,
)-> Path:
    """
    whatに応じたデータセットのパスを返す：
    what が "train" のとき、学習用データのパスを返す
    what が "test_normal" のとき、正常ログの検証用データのパスを返す
    what が "test_abnormal" のとき、異常ログの検証用データのパスを返す
    what が "vocab" のとき、辞書のパスを返す
    """
    # データセットのベースパス
    # another_methodがなければ既存の方式
    if "another_method" in cfg.dataset and cfg.dataset.another_method:
        another_method = True # 識別子
        dataset_dir = Path(cfg.dataset.dataset_dir)
        dataset_dir = dataset_dir / f"ratio_{cfg.dataset.train_ratio}"
    else:
        another_method = False
        dataset_dir = PROCESSED_DIR / cfg.dataset.name / f"ver_{cfg.dataset.version}" / f"ratio_{cfg.dataset.train_ratio}"
    
    # reverse が存在していて True のときだけ「test_normal」を train として扱う
    use_reverse = ("reverse" in cfg.dataset) and cfg.dataset.reverse
    
    # 読み込むファイルの決定
    if what == "train":
        if use_reverse:
            if another_method:
                data_path = dataset_dir / cfg.dataset.name / "test_normal"
            else:
                data_path = dataset_dir / "test_normal"
        else:
            if another_method:
                data_path = dataset_dir / "train"
            else:
                data_path = dataset_dir / "train"
    elif what == "test_normal":
        if use_reverse:
            if another_method:
                data_path = dataset_dir / "train"
            else:
                data_path = dataset_dir / "train"
        else:
            if another_method:
                data_path = dataset_dir / cfg.dataset.name / "test_normal"
            else:
                data_path = dataset_dir / "test_normal"

    elif what == "test_abnormal":
        if another_method:
            data_path = dataset_dir / cfg.dataset.name / "test_abnormal"
        else:
            data_path = dataset_dir / "test_abnormal"
    elif what == "vocab":
        if another_method:
            data_path = Path(cfg.dataset.dataset_dir) / "vocab" /"train"
        else:
            data_path = PROCESSED_DIR / cfg.dataset.name / f"ver_{cfg.dataset.version}" / "vocab" /"train"
    
    return data_path

def generate_train_valid(cfg):
    """
    学習および検証用データを読み込み、固定長ウィンドウに分割したシーケンスを
    train / valid に分割して返す。

    Returns
    -------
    logkey_trainset : np.ndarray
        学習用のログキーシーケンス（長さ降順にソート）
    logkey_validset : np.ndarray
        検証用のログキーシーケンス（長さ降順にソート）
    time_trainset : np.ndarray
        学習用のタイムスタンプシーケンス（長さ降順にソート）
    time_validset : np.ndarray
        検証用のタイムスタンプシーケンス（長さ降順にソート）
    """
    
    data_path = set_dataset_path(cfg, "train")
    
    # データのロード
    with data_path.open("r") as f:
        data_iter = f.readlines()

    # 使用するセッション数
    num_session = int(len(data_iter) * cfg.dataset.sample.sample_ratio)
    num_session = min(num_session, len(data_iter))

    # valid に回すセッション数（ここではセッション数ベースの割合）
    test_size = int(num_session * cfg.dataset.sample.valid_size)

    print("before filtering short session")
    print("train size ", int(num_session - test_size))
    print("valid size ", int(test_size))
    print("=" * 40)

    logkey_seq_pairs = []
    time_seq_pairs = []
    session = 0
    for line in tqdm(data_iter):
        if session >= num_session:
            break
        session += 1

        logkeys, times = fixed_window(
            line,
            cfg.dataset.sample.window_size,
            cfg.dataset.sample.adaptive_window,
            cfg.dataset.sample.seq_len,
            cfg.dataset.sample.min_len,
        )

        # 空セッションはそのままスキップ
        if len(logkeys) == 0:
            continue
        
        logkey_seq_pairs += logkeys
        time_seq_pairs += times

    logkey_seq_pairs = np.array(logkey_seq_pairs)
    time_seq_pairs = np.array(time_seq_pairs)

    # データを　train と val に分割
    logkey_trainset, logkey_validset, time_trainset, time_validset = train_test_split(
        logkey_seq_pairs,
        time_seq_pairs,
        test_size=test_size,
        random_state=cfg.default.r_seed,
    )

    # シーケンスを長さ順にソート
    train_len = list(map(len, logkey_trainset))
    valid_len = list(map(len, logkey_validset))

    train_sort_index = np.argsort(-1 * np.array(train_len))
    valid_sort_index = np.argsort(-1 * np.array(valid_len))

    logkey_trainset = logkey_trainset[train_sort_index]
    logkey_validset = logkey_validset[valid_sort_index]

    time_trainset = time_trainset[train_sort_index]
    time_validset = time_validset[valid_sort_index]

    print("=" * 40)
    print("Num of train seqs", len(logkey_trainset))
    print("Num of valid seqs", len(logkey_validset))
    print("=" * 40)

    return logkey_trainset, logkey_validset, time_trainset, time_validset


def generate_test(cfg):
    """
    テスト用データを読み込み、固定長ウィンドウに分割したシーケンスを返す。

    Returns
    -------
    normal_logkey_seqs : np.ndarray
        正常ログのシーケンス（長さ降順にソート）
    normal_time_seqs : np.ndarray
        正常ログのタイムスタンプシーケンス（長さ降順にソート）
    abnormal_logkey_seqs : np.ndarray
        異常ログのシーケンス（長さ降順にソート）
    abnormal_time_seqs : np.ndarray
        異常ログのタイムスタンプシーケンス（長さ降順にソート）
    """
    # 出力用のリストを用意
    normal_logkey_seqs = []
    normal_time_seqs = []
    abnormal_logkey_seqs = []
    abnormal_time_seqs = []
    
    # test_normal のファイルパス
    normal_path = set_dataset_path(cfg, "test_normal")

    # test_abnormal のファイルパス
    abnormal_path = set_dataset_path(cfg, "test_abnormal")

    # データのロード
    with normal_path.open("r") as f:
        normal_iter = f.readlines()

    with abnormal_path.open("r") as f:
        abnormal_iter = f.readlines()
    
    # 正常ログのシーケンス生成
    for line in tqdm(normal_iter, desc="Processing normal test data"):
        normal_logkey_seq, normal_time_seq = fixed_window(
            line,
            cfg.dataset.sample.window_size,
            cfg.dataset.sample.adaptive_window,
            cfg.dataset.sample.seq_len,
            cfg.dataset.sample.min_len,
        )
        if len(normal_logkey_seq) == 0: # 空シーケンスはskip
            continue

        normal_logkey_seqs += normal_logkey_seq
        normal_time_seqs += normal_time_seq
    
    # 異常ログのシーケンス生成
    for line in tqdm(abnormal_iter, desc="Processing abnormal test data"):
        abnormal_logkey_seq, abnormal_time_seq = fixed_window(
            line,
            cfg.dataset.sample.window_size,
            cfg.dataset.sample.adaptive_window,
            cfg.dataset.sample.seq_len,
            cfg.dataset.sample.min_len,
        )
        if len(abnormal_logkey_seq) == 0: # 空シーケンスはskip
            continue

        abnormal_logkey_seqs += abnormal_logkey_seq
        abnormal_time_seqs += abnormal_time_seq

    # ===== 正常ログを長さ順にソート =====
    normal_logkey_seqs = np.array(normal_logkey_seqs, dtype=object)
    normal_time_seqs = np.array(normal_time_seqs, dtype=object)

    normal_len = list(map(len, normal_logkey_seqs))
    normal_sort_index = np.argsort(-1 * np.array(normal_len))

    normal_logkey_seqs = normal_logkey_seqs[normal_sort_index]
    normal_time_seqs = normal_time_seqs[normal_sort_index]

    # ===== 異常ログも長さ順にソート =====
    abnormal_logkey_seqs = np.array(abnormal_logkey_seqs, dtype=object)
    abnormal_time_seqs = np.array(abnormal_time_seqs, dtype=object)

    abnormal_len = list(map(len, abnormal_logkey_seqs))
    abnormal_sort_index = np.argsort(-1 * np.array(abnormal_len))

    abnormal_logkey_seqs = abnormal_logkey_seqs[abnormal_sort_index]
    abnormal_time_seqs = abnormal_time_seqs[abnormal_sort_index]

    return normal_logkey_seqs, normal_time_seqs, abnormal_logkey_seqs, abnormal_time_seqs


# １シーケンスをlog情報とtime情報に分ける関数
# おそらく、[EventID, deltaT]のような入力を想定
def fixed_window(line, window_size, adaptive_window, seq_len=512, min_len=0):
    line = [ln.split(",") for ln in line.split()]

    # filter the line/session shorter than 0
    if len(line) < min_len:
        return [], []

    # max seq len
    if seq_len is not None:
        line = line[:seq_len]

    if adaptive_window:
        window_size = len(line)

    line = np.array(line)

    # if time duration exists in data
    if line.shape[1] == 2:
        tim = line[:, 1].astype(float)
        line = line[:, 0]

        # the first time duration of a session should be 0, so max is window_size(mins) * 60
        tim[0] = 0
    else:
        line = line.squeeze()
        # if time duration doesn't exist, then create a zero array for time
        # 時間情報が無ければ、timはゼロ埋め
        tim = np.zeros(line.shape)

    logkey_seqs = []
    time_seq = []
    for i in range(0, len(line), window_size):
        logkey_seqs.append(line[i : i + window_size])
        time_seq.append(tim[i : i + window_size])

    padded_logkey_seqs = [
        (
            np.pad(seq, (0, seq_len - len(seq)), constant_values="<pad>")
            if len(seq) < seq_len
            else seq[:seq_len]
        )
        for seq in logkey_seqs
    ]

    padded_time_seqs = [
        (
            np.pad(seq, (0, seq_len - len(seq)), constant_values="0")
            if len(seq) < seq_len
            else seq[:seq_len]
        )
        for seq in time_seq
    ]

    return padded_logkey_seqs, padded_time_seqs


def suggest_vocab(cfg):

    # データセットのパスを設定
    data_path = set_dataset_path(cfg, "vocab")
    
    with data_path.open("r", encoding="utf-8") as f:
        texts = f.readlines()
    
    if cfg.dataset.vocab.name == "wordvocab":
        vocab = WordVocab(
            texts,
            max_size=cfg.dataset.vocab.vocab_size,
            min_freq=cfg.dataset.vocab.min_freq,
        )
    elif cfg.dataset.vocab.name == "vocab":
        vocab = Vocab(texts)
    print("VOCAB SIZE:", len(vocab))
    # vocab.save_vocab(f"{cfg.out_dir}vocab.pkl")
    return vocab


def suggest_testloader(cfg, vocab, eval_batchsize):
    # パーサーを使うverしか実装無し
    if cfg.dataset.parser:
        logkey_normal, time_normal, logkey_abnormal, time_abnormal  = generate_test(cfg)

        normal_data = LogDataset(
            logkey_normal,
            time_normal,
            vocab,
            seq_len=cfg.dataset.sample.seq_len,
            mask_ratio=cfg.dataset.sample.mask_ratio,
            predict_mode=True,
        )

        abnormal_data = LogDataset(
            logkey_abnormal,
            time_abnormal,
            vocab,
            seq_len=cfg.dataset.sample.seq_len,
            mask_ratio=cfg.dataset.sample.mask_ratio,
            predict_mode=True,
        )

        test_normal_loader = DataLoader(
            normal_data,
            batch_size=eval_batchsize,
            num_workers=cfg.default.num_workers,
            collate_fn=normal_data.collate_fn,
            drop_last=True,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
        )

        test_abnormal_loader = DataLoader(
            abnormal_data,
            batch_size=1,
            num_workers=cfg.default.num_workers,
            collate_fn=abnormal_data.collate_fn,
            drop_last=True,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
        )
        del normal_data
        del abnormal_data
        del logkey_normal
        del logkey_abnormal
        del time_abnormal
        del time_normal
        gc.collect()
    else:
        raise ValueError("to do: No Parser version")

    return {
        "test_abnormal": test_abnormal_loader,
        "test_normal": test_normal_loader,
    }


def suggest_dataloader(cfg):
    vocab = suggest_vocab(cfg)
    data = {}

    if cfg.dataset.parser:
        logkey_train, logkey_valid, time_train, time_valid = generate_train_valid(cfg)

        train_data = LogDataset(
            logkey_train,
            time_train,
            vocab,
            seq_len=cfg.dataset.sample.seq_len,
            mask_ratio=cfg.dataset.sample.mask_ratio,
        )
        val_data = LogDataset(
            logkey_valid,
            time_valid,
            vocab,
            seq_len=cfg.dataset.sample.seq_len,
            mask_ratio=cfg.dataset.sample.mask_ratio,
        )

        data["train"] = DataLoader(
            train_data,
            batch_size=cfg.optimizer.hp.batch_size,
            num_workers=cfg.default.num_workers,
            collate_fn=train_data.collate_fn,
            drop_last=True,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True,
        )

        data["val"] = DataLoader(
            val_data,
            batch_size=cfg.optimizer.hp.batch_size,
            num_workers=cfg.default.num_workers,
            collate_fn=val_data.collate_fn,
            drop_last=True,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
        )
        del train_data
        del val_data
        del logkey_train
        del logkey_valid
        del time_train
        del time_valid
        gc.collect()
    else:
        raise ValueError("to do: No Parser version")

    return data
