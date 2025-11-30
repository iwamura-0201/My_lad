import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import sys
from tqdm import tqdm
import pickle
from omegaconf import OmegaConf
import click
from loss.loss import suggest_loss
from train_val import train, val

from dataset.dataset_util import (
    suggest_dataloader, 
    suggest_testloader, 
    suggest_vocab,
)

from util import (
    setup_config,
    fixed_r_seed,
    setup_device,
    suggest_network,
    suggest_optimizer,
    suggest_scheduler,
    plot_log,
    save_learner,
    cal_eval_matrix,
    
)


def do_train(cfg):
    """
    ＜すべて cfg.out_dir 以下に出力＞
    
    実行時の最終config：config.yaml
    各epochでの train/val のloss：output.csv
    loss の推移：result_graph.png & result_graph.pdf
    モデルの重み（最良only）：weights/<monitor名>best.pth
    
    例）
    outputs/logbert/bert/T1105/8/seq_len_512/r_seed_42/
    ├─ config.yaml
    ├─ output.csv
    ├─ result_graph.png
    ├─ result_graph.pdf
    └─ weights/
        ├─ latest.pth
        └─ lossbest.pth  （monitor.name が "loss" なら）
    """
    device = setup_device(cfg)
    fixed_r_seed(cfg)
    data_dict = suggest_dataloader(cfg)
    model = suggest_network(cfg)
    model.to(device)
    optimizer = suggest_optimizer(cfg, model)
    scheduler = suggest_scheduler(cfg, optimizer)
    save_file_path = cfg.out_dir + "output.csv"
    criterion, column_name = suggest_loss(cfg=cfg, device=device, phase="train")
    if cfg.default.monitor.mode == "min":
        best_model_score = 100
    elif cfg.default.monitor.mode == "max":
        best_model_score = 0
    else:
        raise ValueError("choose min or max mode")

    result = []
    no_improve_count = 0
    best_epoch = 0
    for epoch in range(1, cfg.default.epochs + 1):
        for phase in ["train", "val"]:
            if phase == "train":
                train_loss = train(cfg, model, device, data_dict, optimizer, criterion)
            elif phase == "val":
                val_loss = val(cfg, model, device, data_dict, criterion)
        local_result = (
            torch.cat(
                [
                    train_loss.sum().unsqueeze(0),
                    train_loss,
                    val_loss.sum().unsqueeze(0),
                    val_loss,
                ]
            )
            .cpu()
            .tolist()
        )

        result.append(local_result)

        result_df = pd.DataFrame(result, columns=column_name)
        result_df.to_csv(save_file_path, index=False)
        plot_log(cfg, result_df, column_name)
        scheduler.step(epoch)
        print(
            f"epoch {epoch} || TRAIN_Loss:{train_loss.sum():.4f} ||VAL_Loss:{val_loss.sum():.4f}"
        )
        
        current_score = result_df[f"{cfg.default.monitor.name}"].iloc[-1]
        
        improved = False
        if cfg.default.monitor.mode == "min":
            if current_score < best_model_score:
                improved = True
        else:  # mode == "max"
            if current_score > best_model_score:
                improved = True

        if improved:
            best_model_score = current_score
            best_epoch = epoch
            no_improve_count = 0
            save_learner(cfg, model, device, True)
        else:
            no_improve_count += 1

        if no_improve_count >= cfg.default.early_stop_patience:
            print(f"Early stopping at epoch {epoch} (best = {best_epoch})")
            break
        print(f"Best epoch = {best_epoch}")



def test(cfg, model, device, data_dict, output_dir_path):
    """
    テストデータに対して異常検知評価を行う関数。

    < output_dir_path 以下に出力 >
    test_normal_results   ：test_normal の集計結果（pickle）
    test_abnormal_results ：test_abnormal の集計結果（pickle）

    戻り値：
    full_result ：各シーケンス閾値 seq_th ごとの評価指標一覧
        1 行あたり [FP, TP, TN, FN, Precision, Recall, F1, TPR, FPR]
        を要素とする list / ndarray（呼び出し元で CSV 保存などに利用）
    """
    # 評価モードにする
    model.eval()

    normal_abnormal_total_results_list = []

    # with no_grad：勾配変動なしの意
    with torch.no_grad():
        for name in ["test_normal", "test_abnormal"]:
            # data_dict に test_normal と test_abnormal の DataLoader が入っている想定
            dataloader = data_dict[name]

            all_results = {
                "num_error": [],
                "undetected_tokens": [],
                "masked_tokens": [],
                "total_logkey": [],
                "deepSVDD_label": [],
            }

            # DataLoaderをバッチで回す
            for data in tqdm(dataloader):

                # バッチをGPUに乗せる
                data = {key: value.to(device) for key, value in data.items()}

                # モデルに通して出力を得る
                # data["bert_input"]：[B, L]
                # data["bert_label"]：[B, L]
                output = model(data)


                mask_index = data["bert_label"] > 0
                num_masked_tokens = mask_index.sum(dim=-1)

                # top-k予測候補の取得（output["logkey_output"]：[B, L, V]）
                # top_candidates：[B, L, k]
                top_candidates = torch.argsort(-output["logkey_output"], dim=-1)[
                    :, :, : cfg.eval.num_candidates
                ]

                # top-kに正解が含まれない箇所をカウント
                # num_undetected_tokens：[B]
                num_undetected_tokens = (
                    ~(
                        data["bert_label"]
                        .unsqueeze(-1)
                        .expand(-1, -1, top_candidates.size(-1))
                        == top_candidates
                    ).any(dim=-1)
                    & mask_index
                ).sum(dim=-1)

                # リザルトのパッケージング
                batch_results = {
                    "num_error": torch.zeros_like(num_masked_tokens),       #[B]
                    "undetected_tokens": num_undetected_tokens,             #[B]
                    "masked_tokens": num_masked_tokens,                     #[B]
                    "total_logkey": (data["bert_input"] > 0).sum(dim=-1),   #[B]
                    "deepSVDD_label": torch.zeros_like(num_masked_tokens),  #[B]
                }

                for key in all_results:
                    all_results[key].append(batch_results[key])

            # Concatenate all results
            for key in all_results:
                all_results[key] = torch.cat(all_results[key], dim=0)

            normal_abnormal_total_results_list.append(all_results)

        print("Saving test normal results")
        with open(output_dir_path + "/test_normal_results", "wb") as f:
            pickle.dump(normal_abnormal_total_results_list[0], f)

        print("Saving test abnormal results")
        with open(output_dir_path + "/test_abnormal_results", "wb") as f:
            pickle.dump(normal_abnormal_total_results_list[1], f)
            
        seq_start = getattr(cfg.eval, "seq_range_start", 0.0)
        seq_end   = getattr(cfg.eval, "seq_range_end",   1.0)
        seq_step  = getattr(cfg.eval, "seq_range_step",  0.1)

        seq_range = np.arange(seq_start, seq_end + 1e-8, seq_step)

        full_result = cal_eval_matrix(
            normal_abnormal_total_results_list[0],
            normal_abnormal_total_results_list[1],
            cfg=cfg,
            seq_range=seq_range,
        )
        
        return full_result, seq_range
    

def do_test(weight_file_name, eval_batchsize, gpu):
    """
    学習済みモデルの重みファイルを指定してテスト一式を実行するラッパー関数。

    < 入力 >
      weight_file_name ：weights フォルダ配下の .pth ファイルへのパス（train時に生成）
      eval_batchsize  ：テスト時のバッチサイズ
      gpu             ：使用する GPU ID（例: "0"。CPU の場合は実装側の仕様に依存）

    < 出力 >
      test_normal_results   ：test_normal の集計結果（pickle）
      test_abnormal_results ：test_abnormal の集計結果（pickle）
      test_result.csv       ：各 seq_th ごとの [FP, TP, TN, FN, P, R, F1, TPR, FPR]

    例）
    outputs/logbert/bert/T1105/8/seq_len_512/r_seed_42/
    ├─ config.yaml          （学習時に保存済み）
    ├─ output.csv           （学習ログ）
    ├─ result_graph.png
    ├─ result_graph.pdf
    ├─ test_normal_results  （本関数で追加）
    ├─ test_abnormal_results
    ├─ test_result.csv
    └─ weights/
        └─ lossbest.pth など（model_file_name で指定）
    """
    # 実験ディレクトリの導出
    output_dir_path = weight_file_name.split("/")
    output_dir_path = "/".join(output_dir_path[:-2])

    print("=" * 50)
    print(output_dir_path)
    print("=" * 50)

    # cfgの読み込み
    cfg = OmegaConf.load(output_dir_path + "/config.yaml")

    # デバイスのセットアップ・reverseオプションの設定
    # GPU_IDだけは実行時引数の指定を優先
    cfg.default.device_id = gpu
    device = setup_device(cfg)
    if "reverse" not in cfg.dataset:
        cfg.dataset.reverse = False
    print(device, cfg.default.device_id)

    fixed_r_seed(cfg)   # シード値の固定
    vocab = suggest_vocab(cfg)  # vocabのロード

    # modelとdataのロード
    data_dict = suggest_testloader(cfg, vocab, eval_batchsize)
    model = suggest_network(cfg)
    model_weight = torch.load(weight_file_name, map_location={device: "cpu"})
    model.load_state_dict(model_weight)
    model.to(device)

    # テスト実行
    full_result, seq_range = test(cfg, model, device, data_dict, output_dir_path)

    # 結果の表示
    print("[seq_th, FP, TP, TN, FN, Precision, Recall, F1, TPR, FPR]")
    for th, row in zip(seq_range, full_result):
        print(f"threshold={th:.3f}{[f'{num:.4f}' for num in row]}")
    # FP = [x[0] for x in full_result]
    # TP = [x[1] for x in full_result]
    # TN = [x[2] for x in full_result]
    # FN = [x[3] for x in full_result]
    # P = [x[4] for x in full_result]
    # R = [x[5] for x in full_result]
    # F1 = [x[6] for x in full_result]
    # TPR = [x[7] for x in full_result]
    # FPR = [x[8] for x in full_result]

    # csvへの結果の保存
    df = pd.DataFrame(
        full_result, columns=["FP", "TP", "TN", "FN", "P", "R", "F1", "TPR", "FPR"]
    )
    df.to_csv(output_dir_path + "/test_result.csv")



def main_cli(args):
    """
    コマンドライン相当の引数リストを解釈して train / test を実行する関数。
    args には sys.argv[1:] 相当を渡す想定。
    例:
      ["train", "config_name", "default.device_id=0"]
      ["test", "path/to/weights/lossbest.pth", "256", "0"]
    """
    # 引数が何もなければエラー
    if len(args) < 1:
        raise ValueError(
            "Usage:\n"
            "  python main.py train <config_name> [key=value ...]\n"
            "  python main.py test  <model_file_name> <eval_batchsize> <gpu>"
            "  python main.py <config_name> [key=value ...]"
        )

    # 各引数の解釈
    if args[0] in ("train", "test"):
        run_mode = args[0]
        if run_mode == "train":
            if len(args) < 2:
                raise ValueError(
                    "train モードでは config_name を指定してください。\n"
                    "Usage: python main.py train <config_name> [key=value ...]"
                )
            config_file_name = args[1]
            override_args = args[2:]
        else:
            if len(args) < 4:
                raise ValueError(
                    "test モードでは model_file_name, eval_batchsize, gpu を指定してください。\n"
                    "Usage: python main.py test <model_file_name> <eval_batchsize> <gpu>"
                )
            # mode == "test"
            weight_file_name = args[1]
            eval_batchsize = int(args[2])
            gpu = args[3]
    else:
        # モード省略時は train 扱い
        run_mode = "train"
        config_file_name = args[0]
        override_args = args[1:]

    # 実行
    if run_mode == "train":
        cfg = setup_config(config_file_name, override_args) 
        print("setup")
        do_train(cfg)
    else:
        do_test(weight_file_name, eval_batchsize, gpu)



if __name__ == "__main__":
    main_cli(sys.argv[1:])
