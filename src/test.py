import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
import click
from dataset.dataset_util import suggest_testloader, suggest_vocab

from util import fixed_r_seed, setup_device, suggest_network
import pickle


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
                FPR,
                TPR,
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


def test(cfg, model, device, data_dict, output_dir_path):

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

        full_result = cal_eval_matrix(
            normal_abnormal_total_results_list[0],
            normal_abnormal_total_results_list[1],
            cfg=cfg,
            seq_range=np.arange(0, 1, 0.1),
        )

    return full_result


# def plot_test_result(out_dir, FP, TP, TN, FN, P, R, F1):
#     thresholds = np.arange(0, 1, 0.1)

#     # Figure and Subplot設定
#     plt.figure(figsize=(16, 9))

#     # 積み上げ棒グラフ (FP, TP, TN, FN)
#     plt.bar(thresholds, FP, label="FP", color="red", width=0.05)
#     plt.bar(thresholds, TP, bottom=FP, label="TP", color="green", width=0.05)
#     plt.bar(
#         thresholds,
#         TN,
#         bottom=np.array(FP) + np.array(TP),
#         label="TN",
#         color="blue",
#         width=0.05,
#     )
#     plt.bar(
#         thresholds,
#         FN,
#         bottom=np.array(FP) + np.array(TP) + np.array(TN),
#         label="FN",
#         color="orange",
#         width=0.05,
#     )
#     plt.xlabel("Threshold")
#     plt.ylabel("Counts")
#     plt.title("FP, TP, TN, FN Stacked Bar Graph")
#     plt.ylim(0, 100)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(out_dir + "/test1.png")
#     plt.savefig(out_dir + "/test1.pdf")
#     plt.close()

#     # 遷移グラフ (FP, TP, TN, FN)
#     plt.figure(figsize=(16, 9))
#     plt.plot(thresholds, FP, label="FP", marker="o", color="red")
#     plt.plot(thresholds, TP, label="TP", marker="s", color="green")
#     plt.plot(thresholds, TN, label="TN", marker="^", color="blue")
#     plt.plot(thresholds, FN, label="FN", marker="x", color="orange")
#     plt.xlabel("Threshold")
#     plt.ylabel("Counts")
#     plt.title("FP, TP, TN, FN Transition Graph")
#     plt.legend()
#     plt.grid()
#     plt.ylim(0, 100)
#     plt.tight_layout()
#     plt.savefig(out_dir + "/test2.png")
#     plt.savefig(out_dir + "/test2.pdf")
#     plt.close()

#     # P, R, F1 の遷移グラフ
#     plt.figure(figsize=(16, 9))
#     plt.plot(thresholds, P, label="Precision (P)", marker="o", color="purple")
#     plt.plot(thresholds, R, label="Recall (R)", marker="s", color="orange")
#     plt.plot(thresholds, F1, label="F1 Score", marker="^", color="green")
#     plt.xlabel("Threshold")
#     plt.ylabel("Metrics")
#     plt.title("P, R, F1 Transition Graph")
#     plt.legend()
#     plt.grid()
#     plt.ylim(0, 100)
#     plt.tight_layout()
#     plt.savefig(out_dir + "/test3.png")
#     plt.savefig(out_dir + "/test3.pdf")
#     plt.close()


@click.command()
@click.argument("model_file_name", type=str)
@click.argument("eval_batchsize", type=int)
@click.argument("gpu", type=str)
def main(model_file_name, eval_batchsize, gpu):
    # 実験ディレクトリの導出
    output_dir_path = model_file_name.split("/")
    output_dir_path = "/".join(output_dir_path[:-2])

    print("=" * 50)
    print(output_dir_path)
    print("=" * 50)

    # cfgの読み込み
    cfg = OmegaConf.load(output_dir_path + "/config.yaml")

    # デバイスのセットアップ・reverseオプションの設定
    cfg.default.device_id = gpu
    if "reverse" not in cfg.dataset:
        cfg.dataset.reverse = False
    device = setup_device(cfg)
    print(device, cfg.default.device_id)

    fixed_r_seed(cfg)   # シード値の固定
    vocab = suggest_vocab(cfg)  # vocabのロード

    # modelとdataのロード
    data_dict = suggest_testloader(cfg, vocab, eval_batchsize)
    model = suggest_network(cfg)
    model_weight = torch.load(model_file_name, map_location={device: "cpu"})
    model.load_state_dict(model_weight)
    model.to(device)

    # テスト実行
    full_result = test(cfg, model, device, data_dict, output_dir_path)

    # 結果の表示
    print("[seq_th, FP, TP, TN, FN, Precision, Recall, F1, TPR, FPR]")
    for i in range(len(full_result)):
        print(f"thresholds={i*0.1:.1f}{[f'{num:.4f}' for num in full_result[i]]}")
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


if __name__ == "__main__":
    main()
