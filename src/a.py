import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_roc(
    file_path: Path,
)-> None:
    # CSV 読み込み
    df = pd.read_csv(file_path)

    # FPR が % の場合は [0,1] に正規化
    fpr = df["FPR"] / 100.0
    tpr = df["TPR"] / 100.0

    # ROC 曲線
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, marker="o", label="ROC curve")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve:{file_path}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    #plt.savefig(file_path/"roc")
    
def plot_threshold_metrics(file_path: Path) -> None:
    df = pd.read_csv(file_path)

    thresholds = df.index  # seq_threshold
    f1 = df["F1"]
    tpr = df["TPR"]
    fpr = df["FPR"]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(thresholds, f1, marker="o", label="F1")
    ax1.plot(thresholds, tpr, marker="s", label="TPR (Recall)")
    ax1.set_xlabel("seq_threshold")
    ax1.set_ylabel("Score (%)")
    ax1.set_ylim(0, 100)
    ax1.legend(loc="upper right")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(thresholds, fpr, marker="^", linestyle="--", color="black", label="FPR")
    ax2.set_ylabel("FPR (%)")
    ax2.set_ylim(0, 100)

    ax2.legend(loc="upper left")

    plt.title(f"Threshold vs Metrics\n{file_path.name}")
    plt.tight_layout()
    plt.show()