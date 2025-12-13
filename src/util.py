import os
from pathlib import Path
from omegaconf import OmegaConf
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from timm.scheduler import CosineLRScheduler
from model.logbert.log_model import LogBertModel
from model.logbert.bert import BERT
from model.logbert.bert_v2 import BERTv2
from model.logbert.rnn import LSTM, GRU
from model.logbert.mlp import MLP

INTERIM_DIR = Path('../data/interim')
PROCESSED_DIR = Path('../data/processed')
RAW_DIR = Path('../data/raw')

def setup_config(
    config_file_name: str,
    override_args: list[str]
):
    """
    yamlからconfigをロードする関数。
    arg で部分的に書き換えも可。
    生成した output_dir に最終的な config.yaml を残す。
    """

    config_file_path = f"conf/{config_file_name}.yaml"
    if os.path.exists(config_file_path):
        cfg = OmegaConf.load(config_file_path)
    else:
        raise FileNotFoundError(f"No YAML file !!! (path={config_file_path})")

    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args_list=override_args))
    
    #{default.dir_name}/
    #  {network.ver}/
    #    {network.encoder.name}/
    #      {dataset_path}/
    #        seq_len_{dataset.sample.seq_len}/
    #          r_seed_{default.r_seed}/
    # 出力ディレクトリのパス生成
    if "out_dir" not in cfg:
        if "reverse" in cfg.dataset and cfg.dataset.reverse:
            if hasattr(cfg.dataset, 'another_method') and cfg.dataset.another_method:
                dataset_path = f"{cfg.dataset.name}/ratio_{1.0 - cfg.dataset.train_ratio}/"
            else:
                dataset_path = f"{cfg.dataset.name}/ratio_{cfg.dataset.train_ratio}/"
        else:
            if "reverse" in cfg.dataset and cfg.dataset.reverse:
                dataset_path = f"{cfg.dataset.name}/ver_{cfg.dataset.version}/ratio_{1.0 - cfg.dataset.train_ratio}/"
                #dataset_path = f"{cfg.dataset.name}_test{str(cfg.dataset.train_ratio)}train{str(10-cfg.dataset.train_ratio)}/"
            else:
                dataset_path = f"{cfg.dataset.name}/ver_{cfg.dataset.version}/ratio_{cfg.dataset.train_ratio}/"
            #dataset_path = f"{cfg.dataset.name}_train{str(cfg.dataset.train_ratio)}test{str(10-cfg.dataset.train_ratio)}/"
        output_dir_path = (
            f"{cfg.default.dir_name}/"
            + f"{cfg.network.ver}/"
            + f"{cfg.network.encoder.name}/"
            + dataset_path
            + f"seq_len_{cfg.dataset.sample.seq_len}/"
            + f"r_seed_{cfg.default.r_seed}/"
        )
    else:
        output_dir_path = f"{cfg.out_dir}"
    os.makedirs(output_dir_path, exist_ok=True)

    out_dir_comp = {"out_dir": output_dir_path}
    cfg = OmegaConf.merge(cfg, out_dir_comp)

    config_name_comp = {"execute_config_name": config_file_name}
    cfg = OmegaConf.merge(cfg, config_name_comp)

    config_name_comp = {"override_cmd": override_args}
    cfg = OmegaConf.merge(cfg, config_name_comp)

    with open(output_dir_path + "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)
    return cfg


def suggest_network(cfg):
    if cfg.network.ver == "logbert":
        if cfg.network.encoder.name == "bert":
            encoder = BERT(
                cfg.dataset.vocab.vocab_size,
                cfg.dataset.sample.seq_len,
                cfg.network.encoder.hidden_size,
                cfg.network.encoder.layer_num,
                cfg.network.encoder.attn_heads,
                cfg.network.encoder.dropout,
                cfg.network.is_logkey,
                cfg.network.is_time,
            )
        elif cfg.network.encoder.name == "bert_v2":
            encoder = BERTv2(
                cfg.dataset.vocab.vocab_size,
                cfg.dataset.sample.seq_len,
                cfg.network.encoder.hidden_size,
                cfg.network.encoder.layer_num,
                cfg.network.encoder.attn_heads,
                cfg.network.encoder.dropout,
                cfg.network.is_logkey,
                cfg.network.is_time,
            )
        elif cfg.network.encoder.name == "lstm_bi":
            encoder = LSTM(
                cfg.dataset.vocab.vocab_size,
                cfg.dataset.sample.seq_len,
                cfg.network.encoder.hidden_size,
                cfg.network.encoder.layer_num,
                cfg.network.encoder.dropout,
                True,
                cfg.network.is_logkey,
                cfg.network.is_time,
            )
        elif cfg.network.encoder.name == "gru_bi":
            encoder = GRU(
                cfg.dataset.vocab.vocab_size,
                cfg.dataset.sample.seq_len,
                cfg.network.encoder.hidden_size,
                cfg.network.encoder.layer_num,
                cfg.network.encoder.dropout,
                True,
                cfg.network.is_logkey,
                cfg.network.is_time,
            )
        elif cfg.network.encoder.name == "lstm_uni":
            encoder = LSTM(
                cfg.dataset.vocab.vocab_size,
                cfg.dataset.sample.seq_len,
                cfg.network.encoder.hidden_size,
                cfg.network.encoder.layer_num,
                cfg.network.encoder.dropout,
                False,
                cfg.network.is_logkey,
                cfg.network.is_time,
            )
        elif cfg.network.encoder.name == "gru_uni":
            encoder = GRU(
                cfg.dataset.vocab.vocab_size,
                cfg.dataset.sample.seq_len,
                cfg.network.encoder.hidden_size,
                cfg.network.encoder.layer_num,
                cfg.network.encoder.dropout,
                False,
                cfg.network.is_logkey,
                cfg.network.is_time,
            )
        elif cfg.network.encoder.name == "mlp":
            encoder = MLP(
                cfg.dataset.vocab.vocab_size,
                cfg.dataset.sample.seq_len,
                cfg.network.encoder.hidden_size,
                cfg.network.encoder.layer_num,
                cfg.network.encoder.dropout,
                cfg.network.is_logkey,
                cfg.network.is_time,
            )

        model = LogBertModel(encoder, cfg)
    elif cfg.network.ver == "deeplog":
        raise ValueError("to do : deeplog")
    else:
        raise ValueError("Choose Model Ver")
    return model


def setup_device(cfg):
    if torch.cuda.is_available():
        device = cfg.default.device_id
        # Request device check
        try:
            torch.tensor([0]).to(device)
        except RuntimeError as e:
            print(f"Warning: Failed to use {device}: {e}")
            found = False
            # Find available device
            for i in range(torch.cuda.device_count()):
                candidate = f"cuda:{i}"
                try:
                    torch.tensor([0]).to(candidate)
                    print(f"Switching to available device: {candidate}")
                    device = candidate
                    cfg.default.device_id = device # Update config
                    found = True
                    break
                except RuntimeError:
                    continue
            
            if not found:
                 print("Warning: No working CUDA device found.")

        if not cfg.default.deterministic:
            torch.backends.cudnn.benchmark = True
    else:
        device = "cpu"
    return device


def fixed_r_seed(cfg):
    random.seed(cfg.default.r_seed)
    np.random.seed(cfg.default.r_seed)
    torch.manual_seed(cfg.default.r_seed)
    torch.cuda.manual_seed(cfg.default.r_seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def suggest_optimizer(cfg, model):
    if cfg.optimizer.name == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.optimizer.hp.lr,
            momentum=cfg.optimizer.hp.momentum,
            weight_decay=cfg.optimizer.hp.weight_decay,
            nesterov=True,
        )
    elif cfg.optimizer.name == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),  # parameters() メソッドを呼び出す（12/04修正）
            lr=cfg.optimizer.hp.lr,
            weight_decay=cfg.optimizer.hp.weight_decay,
        )
    else:
        raise ValueError("Please select from [SGD or Adam]")
    return optimizer


def suggest_scheduler(cfg, optimizer):
    if cfg.optimizer.scheduler.name == "cosine":
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=cfg.default.epochs,
            lr_min=cfg.optimizer.hp.lr * cfg.optimizer.hp.lr_decay,
            warmup_t=cfg.optimizer.hp.lr_warmup_step,
            warmup_lr_init=cfg.optimizer.hp.lr_warmup_init,
            warmup_prefix=cfg.optimizer.hp.warmup_prefix,
        )
    else:
        raise ValueError("Please select from [cosine]")
    return scheduler


def save_learner(cfg, model, device, BEST=False):
    weight_dir_path = cfg.out_dir + "weights/"
    os.makedirs(weight_dir_path, exist_ok=True)
    if BEST:
        save_file_path = weight_dir_path + f"{cfg.default.monitor.name}best.pth"
    else:
        save_file_path = weight_dir_path + "latest.pth"

    torch.save(
        model.to("cpu").state_dict(),
        save_file_path,
    )
    model.to(device)
    

def plot_log(cfg, result, loss_name):
    fig, axs = plt.subplots(
        figsize=(16, 9), nrows=2, ncols=cfg.loss.num + 1, sharex=True
    )
    for i in range(2):
        for j in range(cfg.loss.num + 1):
            axs[i][j].plot(result[f"{loss_name[(cfg.loss.num + 1) * i + j]}"])
            axs[i][j].set_title(loss_name[(cfg.loss.num + 1) * i + j], fontsize=20)
            axs[i][j].grid()

    fig.supxlabel("Epochs", fontsize=20)
    plt.savefig(cfg.out_dir + "result_graph.png")
    plt.savefig(cfg.out_dir + "result_graph.pdf")
    plt.close()
    
    
    
# ----------------------- do_test で利用 ----------------------------#

def compute_anomaly(results, cfg, seq_threshold=0.5):
    # 各シーケンスの条件に基づき異常としてカウント

    # 1. logkeyの異常判定
    logkey_anomalies = (
        results["undetected_tokens"] > results["masked_tokens"] * seq_threshold
        if cfg.network.is_logkey
        else torch.zeros_like(results["undetected_tokens"], dtype=torch.bool)
    )

    # 2. 時間情報の異常判定
    time_anomalies = (
        results["num_error"] > results["masked_tokens"] * seq_threshold
        if cfg.network.is_time
        else torch.zeros_like(results["num_error"], dtype=torch.bool)
    )

    # 3. ハイパースフィア判定
    hypersphere_anomalies = (
        results["deepSVDD_label"] > 0
        if cfg.eval.hypersphere_loss_test
        else torch.zeros_like(results["deepSVDD_label"], dtype=torch.bool)
    )

    total_anomalies = logkey_anomalies | time_anomalies | hypersphere_anomalies # OR判定
    total_errors = total_anomalies.sum(dim=0).item() # エラー件数

    return total_errors


def cal_eval_matrix(test_normal_results, test_abnormal_results, cfg, seq_range):
    # best_result = [0] * 8
    full_result = []

    for seq_th in seq_range:
        FP = compute_anomaly(test_normal_results, cfg, seq_th)
        TP = compute_anomaly(test_abnormal_results, cfg, seq_th)

        TN = len(test_normal_results["masked_tokens"]) - FP
        FN = len(test_abnormal_results["masked_tokens"]) - TP

        Precision = 100 * TP / (TP + FP) if (TP + FP) != 0 else 0
        Recall = 100 * TP / (TP + FN) if (TP + FN) != 0 else 0
        F1 = 2 * Precision * Recall / (Precision + Recall) if (Precision + Recall) != 0 else 0

        # total = FP + TP + FN + TN

        FPR = FP * 100 / (TN + FP)
        TPR = TP * 100 / (FN + TP)

        full_result.append(
            [
                FP,
                TP,
                TN,
                FN,
                Precision,
                Recall,
                F1,
                TPR,
                FPR
            ]
        )

        # if F1 > best_result[-1]:
        #     best_result = [
        #         seq_th,
        #         FP,
        #         TP,
        #         TN,
        #         FN,
        #         P,
        #         R,
        #         F1,
        #     ]

    return full_result

