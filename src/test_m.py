import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
import click
import pickle
from dataset.dataset_util import suggest_testloader, suggest_vocab
from util import fixed_r_seed, setup_device, suggest_network


def compute_anomaly(results_list, cfg, seq_threshold=0.5):
    num_models = len(results_list)

    # 各モデルの異常判定を格納するリスト
    logkey_anomalies_list = []
    time_anomalies_list = []
    hypersphere_anomalies_list = []

    for results in results_list:
        logkey_anomalies = (
            results["undetected_tokens"] > results["masked_tokens"] * seq_threshold
            if cfg.network.is_logkey
            else torch.zeros_like(results["undetected_tokens"], dtype=torch.bool)
        )

        time_anomalies = (
            results["num_error"] > results["masked_tokens"] * seq_threshold
            if cfg.network.is_time
            else torch.zeros_like(results["num_error"], dtype=torch.bool)
        )

        hypersphere_anomalies = (
            results["deepSVDD_label"] > 0
            if cfg.eval.hypersphere_loss_test
            else torch.zeros_like(results["deepSVDD_label"], dtype=torch.bool)
        )

        logkey_anomalies_list.append(logkey_anomalies)
        time_anomalies_list.append(time_anomalies)
        hypersphere_anomalies_list.append(hypersphere_anomalies)

    # 各異常判定の合計
    logkey_sum = torch.stack(logkey_anomalies_list).sum(dim=0)
    time_sum = torch.stack(time_anomalies_list).sum(dim=0)
    hypersphere_sum = torch.stack(hypersphere_anomalies_list).sum(dim=0)

    # 過半数のモデルが異常と判断した場合のみ異常にする
    majority_vote = 0  # 過半数の閾値
    total_anomalies = (
        (logkey_sum > majority_vote)
        | (time_sum > majority_vote)
        | (hypersphere_sum > majority_vote)
    )

    total_errors = total_anomalies.sum(dim=0).item()
    return total_errors


def cal_eval_matrix(
    test_normal_results_list, test_abnormal_results_list, cfg, seq_range
):
    full_result = []

    for seq_th in seq_range:
        FP = compute_anomaly(test_normal_results_list, cfg, seq_th)
        TP = compute_anomaly(test_abnormal_results_list, cfg, seq_th)

        TN = len(test_normal_results_list[0]["masked_tokens"]) - FP
        FN = len(test_abnormal_results_list[0]["masked_tokens"]) - TP

        P = 100 * TP / (TP + FP) if (TP + FP) != 0 else 0
        R = 100 * TP / (TP + FN) if (TP + FN) != 0 else 0
        F1 = 2 * P * R / (P + R) if (P + R) != 0 else 0

        FPR = FP * 100 / (TN + FP)
        TPR = TP * 100 / (FN + TP)

        full_result.append([FP, TP, TN, FN, P, R, F1, FPR, TPR])

    return full_result


def test(cfg, models, device, data_dict, output_dir_path):
    for model in models:
        model.eval()

    normal_abnormal_total_results_list = [[] for _ in range(len(models))]

    with torch.no_grad():
        for name in ["test_normal", "test_abnormal"]:
            dataloader = data_dict[name]

            all_results_list = []
            for _ in models:
                all_results_list.append(
                    {
                        "num_error": [],
                        "undetected_tokens": [],
                        "masked_tokens": [],
                        "total_logkey": [],
                        "deepSVDD_label": [],
                    }
                )

            for data in tqdm(dataloader):
                data = {key: value.to(device) for key, value in data.items()}

                outputs = [model(data) for model in models]
                mask_index = data["bert_label"] > 0
                num_masked_tokens = mask_index.sum(dim=-1)

                for i, output in enumerate(outputs):
                    top_candidates = torch.argsort(-output["logkey_output"], dim=-1)[
                        :, :, : cfg.eval.num_candidates
                    ]
                    num_undetected_tokens = (
                        ~(
                            data["bert_label"]
                            .unsqueeze(-1)
                            .expand(-1, -1, top_candidates.size(-1))
                            == top_candidates
                        ).any(dim=-1)
                        & mask_index
                    ).sum(dim=-1)

                    batch_results = {
                        "num_error": torch.zeros_like(num_masked_tokens),
                        "undetected_tokens": num_undetected_tokens,
                        "masked_tokens": num_masked_tokens,
                        "total_logkey": (data["bert_input"] > 0).sum(dim=-1),
                        "deepSVDD_label": torch.zeros_like(num_masked_tokens),
                    }

                    for key in all_results_list[i]:
                        all_results_list[i][key].append(batch_results[key])

            # Concatenate all results
            for i in range(len(models)):
                for key in all_results_list[i]:
                    all_results_list[i][key] = torch.cat(
                        all_results_list[i][key], dim=0
                    )
                normal_abnormal_total_results_list[i].append(all_results_list[i])

        full_result = cal_eval_matrix(
            [
                results[0] for results in normal_abnormal_total_results_list
            ],  # test_normal
            [
                results[1] for results in normal_abnormal_total_results_list
            ],  # test_abnormal
            cfg=cfg,
            seq_range=np.arange(0, 1, 0.1),
        )

    return full_result


@click.command()
@click.argument("model_file_paths", nargs=-1, type=str)
@click.argument("eval_batchsize", type=int)
@click.argument("gpu", type=str)
def main(model_file_paths, eval_batchsize, gpu):
    output_dir_path = model_file_paths[0].split("/")
    output_dir_path = "/".join(output_dir_path[:-2])
    cfg = OmegaConf.load(output_dir_path + "/config.yaml")
    cfg.default.device_id = gpu
    device = setup_device(cfg)
    fixed_r_seed(cfg)
    vocab = suggest_vocab(cfg)
    data_dict = suggest_testloader(cfg, vocab, eval_batchsize)

    models = []
    for model_path in model_file_paths:
        output_dir_path = model_path.split("/")
        output_dir_path = "/".join(output_dir_path[:-2])
        cfg = OmegaConf.load(output_dir_path + "/config.yaml")
        model = suggest_network(cfg)
        model_weight = torch.load(model_path, map_location={device: "cpu"})
        model.load_state_dict(model_weight)
        model.to(device)
        models.append(model)

    full_result = test(
        cfg, models, device, data_dict, model_file_paths[0].split("/")[0]
    )

    df = pd.DataFrame(
        full_result, columns=["FP", "TP", "TN", "FN", "P", "R", "F1", "FPR", "TPR"]
    )
    df.to_csv(model_file_paths[0].split("/")[0] + "/test_result.csv")
    print("[seq_th, FP, TP, TN, FN, P, R, F1, TPR, FPR]")
    for i in range(len(full_result)):
        print(f"thresholds={i*0.1:.1f}{[f'{num:.4f}' for num in full_result[i]]}")


if __name__ == "__main__":
    main()
