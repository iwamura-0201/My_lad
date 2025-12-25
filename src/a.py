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
    