from torch.utils.data import DataLoader
import numpy as np
import gc
from vocab import WordVocab, Vocab
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from .logdataset import LogDataset


def generate_train_valid(cfg):
    if (cfg.dataset.reverse) & ("reverse" in cfg.dataset):
        with open(
            f"./dataset/preprocessed/{cfg.dataset.name}/test_normal{str(10-cfg.dataset.train_ratio)}",
            "r",
        ) as f:
            data_iter = f.readlines()
    else:
        with open(
            f"./dataset/preprocessed/{cfg.dataset.name}/train{str(cfg.dataset.train_ratio)}",
            "r",
        ) as f:
            data_iter = f.readlines()

    num_session = int(len(data_iter) * cfg.dataset.sample.sample_ratio)

    test_size = int(min(num_session, len(data_iter)) * cfg.dataset.sample.valid_size)

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

        logkey_seq_pairs += logkeys
        time_seq_pairs += times

    logkey_seq_pairs = np.array(logkey_seq_pairs)
    time_seq_pairs = np.array(time_seq_pairs)

    logkey_trainset, logkey_validset, time_trainset, time_validset = train_test_split(
        logkey_seq_pairs,
        time_seq_pairs,
        test_size=test_size,
        random_state=cfg.default.r_seed,
    )

    # sort seq_pairs by seq len
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


def generate_test(cfg, file_name):
    """
    :return: log_seqs: num_samples x session(seq)_length, tim_seqs: num_samples x session_length
    """
    log_seqs = []
    tim_seqs = []
    with open(f"./dataset/preprocessed/{cfg.dataset.name}/{file_name}", "r") as f:
        for _, line in tqdm(enumerate(f.readlines())):
            log_seq, tim_seq = fixed_window(
                line,
                cfg.dataset.sample.window_size,
                cfg.dataset.sample.adaptive_window,
                cfg.dataset.sample.seq_len,
                cfg.dataset.sample.min_len,
            )
            if len(log_seq) == 0:
                continue

            log_seqs += log_seq
            tim_seqs += tim_seq

    # sort seq_pairs by seq len
    log_seqs = np.array(log_seqs)
    tim_seqs = np.array(tim_seqs)

    test_len = list(map(len, log_seqs))
    test_sort_index = np.argsort(-1 * np.array(test_len))

    log_seqs = log_seqs[test_sort_index]
    tim_seqs = tim_seqs[test_sort_index]

    print(f"{file_name} size: {len(log_seqs)}")
    return log_seqs, tim_seqs


def fixed_window(line, window_size, adaptive_window, seq_len=512, min_len=0):
    line = [ln.split(",") for ln in line.split()]

    # filter the line/session shorter than 10
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
    with open(
        f"./dataset/preprocessed/{cfg.dataset.name}/train_", "r", encoding="utf-8"
    ) as f:
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
    if cfg.dataset.parser:
        if (cfg.dataset.reverse) & ("reverse" in cfg.dataset):
            logkey_normal, time_normal = generate_test(
                cfg, f"train{str(cfg.dataset.train_ratio)}"
            )
        else:
            logkey_normal, time_normal = generate_test(
                cfg, f"test_normal{str(10-cfg.dataset.train_ratio)}"
            )
        logkey_abnormal, time_abnormal = generate_test(cfg, "test_abnormal")

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
