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


def compute_anomaly(resultsA, resultsB, cfg, seq_threshold=0.5):
    # modelAとmodelBの両方が異常と判断した場合のみ異常とする
    logkey_anomalies_A = (
        resultsA["undetected_tokens"] > resultsA["masked_tokens"] * seq_threshold
        if cfg.network.is_logkey
        else torch.zeros_like(resultsA["undetected_tokens"], dtype=torch.bool)
    )

    logkey_anomalies_B = (
        resultsB["undetected_tokens"] > resultsB["masked_tokens"] * seq_threshold
        if cfg.network.is_logkey
        else torch.zeros_like(resultsB["undetected_tokens"], dtype=torch.bool)
    )

    time_anomalies_A = (
        resultsA["num_error"] > resultsA["masked_tokens"] * seq_threshold
        if cfg.network.is_time
        else torch.zeros_like(resultsA["num_error"], dtype=torch.bool)
    )

    time_anomalies_B = (
        resultsB["num_error"] > resultsB["masked_tokens"] * seq_threshold
        if cfg.network.is_time
        else torch.zeros_like(resultsB["num_error"], dtype=torch.bool)
    )

    hypersphere_anomalies_A = (
        resultsA["deepSVDD_label"] > 0
        if cfg.eval.hypersphere_loss_test
        else torch.zeros_like(resultsA["deepSVDD_label"], dtype=torch.bool)
    )

    hypersphere_anomalies_B = (
        resultsB["deepSVDD_label"] > 0
        if cfg.eval.hypersphere_loss_test
        else torch.zeros_like(resultsB["deepSVDD_label"], dtype=torch.bool)
    )

    # 両方のモデルが異常と判断した場合のみ異常とする
    total_anomalies = (
        (logkey_anomalies_A & logkey_anomalies_B)
        | (time_anomalies_A & time_anomalies_B)
        | (hypersphere_anomalies_A & hypersphere_anomalies_B)
    )

    total_errors = total_anomalies.sum(dim=0).item()
    return total_errors


def cal_eval_matrix(
    test_normal_resultsA,
    test_normal_resultsB,
    test_abnormal_resultsA,
    test_abnormal_resultsB,
    cfg,
    seq_range,
):
    full_result = []

    for seq_th in seq_range:
        FP = compute_anomaly(test_normal_resultsA, test_normal_resultsB, cfg, seq_th)
        TP = compute_anomaly(
            test_abnormal_resultsA, test_abnormal_resultsB, cfg, seq_th
        )

        TN = len(test_normal_resultsA["masked_tokens"]) - FP
        FN = len(test_abnormal_resultsA["masked_tokens"]) - TP

        P = 100 * TP / (TP + FP) if (TP + FP) != 0 else 0
        R = 100 * TP / (TP + FN) if (TP + FN) != 0 else 0
        F1 = 2 * P * R / (P + R) if (P + R) != 0 else 0

        FPR = FP * 100 / (TN + FP)
        TPR = TP * 100 / (FN + TP)

        full_result.append(
            [
                FP,
                TP,
                TN,
                FN,
                P,
                R,
                F1,
                FPR,
                TPR,
            ]
        )

    return full_result


def test(cfg, modelA, modelB, device, data_dict, output_dir_path):
    modelA.eval()
    modelB.eval()

    normal_abnormal_total_results_listA = []
    normal_abnormal_total_results_listB = []

    with torch.no_grad():
        for name in ["test_normal", "test_abnormal"]:
            dataloader = data_dict[name]

            all_resultsA = {
                "num_error": [],
                "undetected_tokens": [],
                "masked_tokens": [],
                "total_logkey": [],
                "deepSVDD_label": [],
            }

            all_resultsB = {
                "num_error": [],
                "undetected_tokens": [],
                "masked_tokens": [],
                "total_logkey": [],
                "deepSVDD_label": [],
            }

            for data in tqdm(dataloader):
                data = {key: value.to(device) for key, value in data.items()}
                outputA = modelA(data)
                outputB = modelB(data)

                mask_index = data["bert_label"] > 0
                num_masked_tokens = mask_index.sum(dim=-1)

                top_candidatesA = torch.argsort(-outputA["logkey_output"], dim=-1)[
                    :, :, : cfg.eval.num_candidates
                ]
                top_candidatesB = torch.argsort(-outputB["logkey_output"], dim=-1)[
                    :, :, : cfg.eval.num_candidates
                ]

                num_undetected_tokensA = (
                    ~(
                        data["bert_label"]
                        .unsqueeze(-1)
                        .expand(-1, -1, top_candidatesA.size(-1))
                        == top_candidatesA
                    ).any(dim=-1)
                    & mask_index
                ).sum(dim=-1)

                num_undetected_tokensB = (
                    ~(
                        data["bert_label"]
                        .unsqueeze(-1)
                        .expand(-1, -1, top_candidatesB.size(-1))
                        == top_candidatesB
                    ).any(dim=-1)
                    & mask_index
                ).sum(dim=-1)

                batch_resultsA = {
                    "num_error": torch.zeros_like(num_masked_tokens),
                    "undetected_tokens": num_undetected_tokensA,
                    "masked_tokens": num_masked_tokens,
                    "total_logkey": (data["bert_input"] > 0).sum(dim=-1),
                    "deepSVDD_label": torch.zeros_like(num_masked_tokens),
                }

                batch_resultsB = {
                    "num_error": torch.zeros_like(num_masked_tokens),
                    "undetected_tokens": num_undetected_tokensB,
                    "masked_tokens": num_masked_tokens,
                    "total_logkey": (data["bert_input"] > 0).sum(dim=-1),
                    "deepSVDD_label": torch.zeros_like(num_masked_tokens),
                }

                for key in all_resultsA:
                    all_resultsA[key].append(batch_resultsA[key])
                    all_resultsB[key].append(batch_resultsB[key])

            # Concatenate all results
            for key in all_resultsA:
                all_resultsA[key] = torch.cat(all_resultsA[key], dim=0)
                all_resultsB[key] = torch.cat(all_resultsB[key], dim=0)

            normal_abnormal_total_results_listA.append(all_resultsA)
            normal_abnormal_total_results_listB.append(all_resultsB)

        print("Saving test normal results")
        with open(output_dir_path + "/test_normal_resultsA", "wb") as f:
            pickle.dump(normal_abnormal_total_results_listA[0], f)
        with open(output_dir_path + "/test_normal_resultsB", "wb") as f:
            pickle.dump(normal_abnormal_total_results_listB[0], f)

        print("Saving test abnormal results")
        with open(output_dir_path + "/test_abnormal_resultsA", "wb") as f:
            pickle.dump(normal_abnormal_total_results_listA[1], f)
        with open(output_dir_path + "/test_abnormal_resultsB", "wb") as f:
            pickle.dump(normal_abnormal_total_results_listB[1], f)

        full_result = cal_eval_matrix(
            normal_abnormal_total_results_listA[0],
            normal_abnormal_total_results_listB[0],
            normal_abnormal_total_results_listA[1],
            normal_abnormal_total_results_listB[1],
            cfg=cfg,
            seq_range=np.arange(0, 1, 0.1),
        )

    return full_result


@click.command()
@click.argument("model_file_name", type=str)
@click.argument("model_file_name2", type=str)
@click.argument("eval_batchsize", type=int)
@click.argument("gpu", type=str)
def main(model_file_name, model_file_name2, eval_batchsize, gpu):
    output_dir_path = model_file_name.split("/")
    output_dir_path = "/".join(output_dir_path[:-2])
    cfg = OmegaConf.load(output_dir_path + "/config.yaml")
    output_dir_path2 = model_file_name2.split("/")
    output_dir_path2 = "/".join(output_dir_path2[:-2])
    cfg2 = OmegaConf.load(output_dir_path2 + "/config.yaml")
    print("=" * 50)
    print(output_dir_path)
    print(output_dir_path2)
    print("=" * 50)
    cfg.default.device_id = gpu
    if "reverse" not in cfg.dataset:
        cfg.dataset.reverse = False
    device = setup_device(cfg)
    print(device, cfg.default.device_id)
    fixed_r_seed(cfg)
    vocab = suggest_vocab(cfg)
    data_dict = suggest_testloader(cfg, vocab, eval_batchsize)
    model = suggest_network(cfg)
    model2 = suggest_network(cfg2)
    model_weight = torch.load(model_file_name, map_location={device: "cpu"})
    model_weight2 = torch.load(model_file_name2, map_location={device: "cpu"})
    model.load_state_dict(model_weight)
    model2.load_state_dict(model_weight2)
    model.to(device)
    model2.to(device)
    full_result = test(cfg, model, model2, device, data_dict, output_dir_path)
    print("[seq_th, FP, TP, TN, FN, P, R, F1, FPR, TPR]")
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

    df = pd.DataFrame(
        full_result, columns=["FP", "TP", "TN", "FN", "P", "R", "F1", "FPR", "TPR"]
    )
    df.to_csv(output_dir_path + "/test_result.csv")


if __name__ == "__main__":
    main()
