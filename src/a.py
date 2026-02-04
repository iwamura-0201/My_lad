import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# def plot_roc(
#     file_path: Path,
# )-> None:
#     # CSV 読み込み
#     df = pd.read_csv(file_path)

#     # FPR が % の場合は [0,1] に正規化
#     fpr = df["FPR"] / 100.0
#     tpr = df["TPR"] / 100.0

#     # ROC 曲線
#     plt.figure(figsize=(6, 6))
#     plt.plot(fpr, tpr, marker="o", label="ROC curve")
#     plt.plot([0, 1], [0, 1], linestyle="--", label="Random")

#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.title(f"ROC Curve:{file_path}")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
    
    #plt.savefig(file_path/"roc")


def plot_roc(file_path: Path, add_endpoint: bool = False) -> None:
    """
    ROC曲線を描画する関数
    
    Parameters
    ----------
    file_path : Path
        CSVファイルのパス（FPR, TPRカラムを含む）
    add_endpoint : bool, optional
        Trueの場合、(1,1)の点が存在しない場合に追加する（デフォルト: False）
    """
    # CSV 読み込み
    df = pd.read_csv(file_path)

    # FPR/TPR が % の場合は [0,1] に正規化
    fpr = df["FPR"] / 100.0
    tpr = df["TPR"] / 100.0
    
    # add_endpointがTrueの場合、(1,1)の点が存在しない場合に追加
    if add_endpoint:
        # (1,1)付近の点が存在するかチェック (浮動小数点誤差を考慮)
        has_endpoint = ((fpr >= 0.99) & (tpr >= 0.99)).any()
        if not has_endpoint:
            # (1,1)の点を先頭に追加
            fpr = pd.concat([pd.Series([1.0]), fpr], ignore_index=True)
            tpr = pd.concat([pd.Series([1.0]), tpr], ignore_index=True)

    # --- 論文向けスタイル設定（配色中心）---
    # 色覚多様性に比較的強い代表色（Okabe-Ito系）
    roc_color = "#0072B2"   # 青
    rand_color = "#7F7F7F"  # グレー（主張を抑える）

    plt.figure(figsize=(6, 6))

    # ROC 曲線：青 + 実線 + 小さめマーカー（白抜きで視認性UP）
    plt.plot(
        fpr, tpr,
        color=roc_color,
        linewidth=2.2,
        linestyle="-",
        marker="o",
        markersize=4.5,
        markerfacecolor="white",
        markeredgecolor=roc_color,
        markeredgewidth=1.2,
        label="ROC"
    )

    # Random：控えめなグレー + 破線
    plt.plot(
        [0, 1], [0, 1],
        color=rand_color,
        linewidth=1.8,
        linestyle=(0, (4, 3)),  # ほどよい破線
        label="Random"
    )

    plt.xlabel("FPR (%)")
    plt.ylabel("TPR (%)")
    # plt.title(f"ROC Curve: {file_path}")

    # グリッドは薄く（論文で邪魔になりがちなので）
    plt.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.5)

    # 余白・枠
    ax = plt.gca()
    # 端で欠けないように少し余白を作る（重要）
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.02)

    ax.set_aspect("equal", adjustable="box")

     # 上・右の罫線（spine）を削除
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 凡例：枠付き・背景白で印刷でも読みやすく
    # plt.legend(
    #     loc="lower right",
    #     frameon=True,
    #     framealpha=1.0,
    #     facecolor="white",
    #     edgecolor="#CCCCCC"
    # )

    plt.tight_layout()
    plt.show()
    

def plot_pr(file_path: Path) -> None:
    """
    PR曲線（Precision-Recall curve）を描画する関数
    """
    # CSV 読み込み
    df = pd.read_csv(file_path)

    # Precision/Recall が % の場合は [0,1] に正規化
    precision = df["P"] / 100.0
    recall = df["R"] / 100.0

    # --- 論文向けスタイル設定（配色中心）---
    # 色覚多様性に比較的強い代表色（Okabe-Ito系）
    pr_color = "#E69F00"   # オレンジ
    baseline_color = "#7F7F7F"  # グレー（主張を抑える）

    plt.figure(figsize=(6, 6))

    # PR 曲線：オレンジ + 実線 + 小さめマーカー（白抜きで視認性UP）
    plt.plot(
        recall, precision,
        color=pr_color,
        linewidth=2.2,
        linestyle="-",
        marker="o",
        markersize=4.5,
        markerfacecolor="white",
        markeredgecolor=pr_color,
        markeredgewidth=1.2,
        label="PR"
    )

    # Baseline（完全ランダムの場合のPrecision）：控えめなグレー + 破線
    # ※正例率が不明な場合は baseline を引かないことも多い
    # ここでは視覚的参考として y=0.5 のラインを引く例
    plt.axhline(
        y=0.5,
        color=baseline_color,
        linewidth=1.8,
        linestyle=(0, (4, 3)),  # ほどよい破線
        label="Baseline"
    )

    plt.xlabel("Recall (%)")
    plt.ylabel("Precision (%)")
    # plt.title(f"PR Curve: {file_path}")

    # グリッドは薄く（論文で邪魔になりがちなので）
    plt.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.5)

    # 余白・枠
    ax = plt.gca()
    # 端で欠けないように少し余白を作る（重要）
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.02)

    ax.set_aspect("equal", adjustable="box")

    # 上・右の罫線（spine）を削除
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 凡例：枠付き・背景白で印刷でも読みやすく
    # plt.legend(
    #     loc="lower left",
    #     frameon=True,
    #     framealpha=1.0,
    #     facecolor="white",
    #     edgecolor="#CCCCCC"
    # )

    plt.tight_layout()
    plt.show()
    

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


def plot_roc_compare(file_path1: Path, file_path2: Path, label1: str = "Model 1", label2: str = "Model 2", add_endpoint: bool = False) -> None:
    """
    2つのtest_result.csvファイルからROC曲線を同じグラフにプロットする関数
    
    Parameters
    ----------
    file_path1 : Path
        1つ目のCSVファイルのパス（FPR, TPRカラムを含む）
    file_path2 : Path
        2つ目のCSVファイルのパス（FPR, TPRカラムを含む）
    label1 : str, optional
        1つ目の曲線のラベル（デフォルト: "Model 1"）
    label2 : str, optional
        2つ目の曲線のラベル（デフォルト: "Model 2"）
    add_endpoint : bool, optional
        Trueの場合、(1,1)の点が存在しない場合に追加する（デフォルト: False）
    """
    # CSV 読み込み
    df1 = pd.read_csv(file_path1)
    df2 = pd.read_csv(file_path2)

    # FPR/TPR が % の場合は [0,1] に正規化
    fpr1 = df1["FPR"] / 100.0
    tpr1 = df1["TPR"] / 100.0
    fpr2 = df2["FPR"] / 100.0
    tpr2 = df2["TPR"] / 100.0

    # add_endpointがTrueの場合、(1,1)の点が存在しない場合に追加
    if add_endpoint:
        # Data 1
        has_endpoint1 = ((fpr1 >= 0.99) & (tpr1 >= 0.99)).any()
        if not has_endpoint1:
            fpr1 = pd.concat([pd.Series([1.0]), fpr1], ignore_index=True)
            tpr1 = pd.concat([pd.Series([1.0]), tpr1], ignore_index=True)
        # Data 2
        has_endpoint2 = ((fpr2 >= 0.99) & (tpr2 >= 0.99)).any()
        if not has_endpoint2:
            fpr2 = pd.concat([pd.Series([1.0]), fpr2], ignore_index=True)
            tpr2 = pd.concat([pd.Series([1.0]), tpr2], ignore_index=True)

    # --- 論文向けスタイル設定（配色中心）---
    # 色覚多様性に比較的強い代表色（Okabe-Ito系）
    color1 = "#0072B2"   # 青
    color2 = "#D55E00"   # オレンジ
    rand_color = "#7F7F7F"  # グレー（主張を抑える）

    plt.figure(figsize=(6, 6))

    # ROC 曲線1：青 + 実線 + 小さめマーカー（白抜きで視認性UP）
    plt.plot(
        fpr1, tpr1,
        color=color1,
        linewidth=2.2,
        linestyle="-",
        marker="o",
        markersize=4.5,
        markerfacecolor="white",
        markeredgecolor=color1,
        markeredgewidth=1.2,
        label=label1
    )

    # ROC 曲線2：オレンジ + 実線 + 小さめマーカー（白抜きで視認性UP）
    plt.plot(
        fpr2, tpr2,
        color=color2,
        linewidth=2.2,
        linestyle="-",
        marker="s",  # 四角形マーカー
        markersize=4.5,
        markerfacecolor="white",
        markeredgecolor=color2,
        markeredgewidth=1.2,
        label=label2
    )

    # Random：控えめなグレー + 破線
    plt.plot(
        [0, 1], [0, 1],
        color=rand_color,
        linewidth=1.8,
        linestyle=(0, (4, 3)),  # ほどよい破線
        label="Random"
    )

    plt.xlabel("FPR")
    plt.ylabel("TPR")

    # グリッドは薄く（論文で邪魔になりがちなので）
    plt.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.5)

    # 余白・枠
    ax = plt.gca()
    # 端で欠けないように少し余白を作る（重要）
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.02)

    ax.set_aspect("equal", adjustable="box")

    # 上・右の罫線（spine）を削除
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 凡例：枠付き・背景白で印刷でも読みやすく
    plt.legend(
        loc="lower right",
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="#CCCCCC"
    )

    plt.tight_layout()
    plt.show()