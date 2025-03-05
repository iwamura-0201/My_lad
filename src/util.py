import os
import sys
from omegaconf import OmegaConf
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from timm.scheduler import CosineLRScheduler
from model.logbert.log_model import LogBertModel
from model.logbert.bert import BERT
from model.logbert.rnn import LSTM, GRU
from model.logbert.mlp import MLP


def setup_config():
    args = sys.argv

    config_file_name = args[1]
    config_file_path = f"./src/conf/{config_file_name}.yaml"
    if os.path.exists(config_file_path):
        cfg = OmegaConf.load(config_file_path)
    else:
        raise "No YAML file !!!"

    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args_list=args[2:]))

    if "out_dir" not in cfg:
        if (cfg.dataset.reverse) & ("reverse" in cfg.dataset):
            dataset_path = f"{cfg.dataset.name}_test{str(cfg.dataset.train_ratio)}train{str(10-cfg.dataset.train_ratio)}/"
        else:
            dataset_path = f"{cfg.dataset.name}_train{str(cfg.dataset.train_ratio)}test{str(10-cfg.dataset.train_ratio)}/"
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

    config_name_comp = {"override_cmd": args[2:]}
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
            model.parameters,
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
