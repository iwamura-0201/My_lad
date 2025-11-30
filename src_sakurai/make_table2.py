import pandas as pd
import click


def make_table(encoder_names, dataset_names, train_ratios, seq_lens):
    # LaTeXの表用リスト
    latex_table = []

    # 動的に固定値またはリスト内の要素が1つのパラメータを検出
    caption_parts = []
    param_axes = []
    direction_needed = any("lstm" in enc or "gru" in enc for enc in encoder_names)

    if len(dataset_names) == 1:
        caption_parts.append(f"Dataset: {dataset_names[0]}")
    else:
        param_axes.append("Dataset")
    if len(seq_lens) == 1:
        caption_parts.append(f"Sequence Length: {seq_lens[0]}")
    else:
        param_axes.append("Seq Len")

    if len(encoder_names) == 1:
        if direction_needed:
            caption_parts.append(f"Encoder: {'LSTM/GRU'}")
        elif encoder_names[0] not in ["lstm_uni", "lstm_bi", "gru_uni", "gru_bi"]:
            caption_parts.append(f"Encoder: {encoder_names[0].upper()}")
    else:
        param_axes.append("Encoder")
        if direction_needed:
            param_axes.append("Direction")

    if len(train_ratios) == 1:
        caption_parts.append(
            f"Test:Train Ratio: {train_ratios[0]}:{10-train_ratios[0]}"
        )
    else:
        param_axes.append("Test:Train")

    # 各パラメータの組み合わせを処理
    previous_row = {}
    for dataset_name in dataset_names:
        for encoder_name in encoder_names:
            for train_ratio in train_ratios:
                for seq_len in seq_lens:
                    file_path = f"./reverse/logbert/{encoder_name}/{dataset_name}_test{train_ratio}train{10-train_ratio}/seq_len_{seq_len}/r_seed_31/test_result.csv"
                    try:
                        # CSVファイルを読み込む
                        data = pd.read_csv(file_path)

                        # F1スコアが最も高い行を抽出
                        max_f1_row = data.loc[data["F1"].idxmax()]

                        # Encoder名の変換
                        if "lstm" in encoder_name:
                            base_encoder = "LSTM"
                        elif "gru" in encoder_name:
                            base_encoder = "GRU"
                        else:
                            base_encoder = encoder_name.upper()

                        # LaTeX行を作成
                        direction = (
                            "Bi"
                            if "bi" in encoder_name
                            else ("Uni" if "uni" in encoder_name else "")
                        )
                        latex_row = []

                        if "Dataset" in param_axes:
                            latex_row.append(
                                dataset_name
                                if previous_row.get("Dataset") != dataset_name
                                else ""
                            )
                            previous_row["Dataset"] = dataset_name
                        if "Encoder" in param_axes:
                            latex_row.append(
                                base_encoder
                                if previous_row.get("Encoder") != base_encoder
                                else ""
                            )
                            previous_row["Encoder"] = base_encoder
                        if "Direction" in param_axes:
                            latex_row.append(
                                direction
                                if previous_row.get("Direction") != direction
                                else ""
                            )
                            previous_row["Direction"] = direction
                        if "Test:Train" in param_axes:
                            train_test = f"{train_ratio}:{10-train_ratio}"
                            latex_row.append(
                                train_test
                                if previous_row.get("Test:Train") != train_test
                                else ""
                            )
                            previous_row["Test:Train"] = train_test
                        if "Seq Len" in param_axes:
                            latex_row.append(
                                str(seq_len)
                                if previous_row.get("Seq Len") != seq_len
                                else ""
                            )
                            previous_row["Seq Len"] = seq_len

                        latex_row.extend(
                            [
                                f"{max_f1_row['FP']:.2f}",
                                f"{max_f1_row['TP']:.2f}",
                                f"{max_f1_row['TN']:.2f}",
                                f"{max_f1_row['FN']:.2f}",
                                f"{max_f1_row['P']:.2f}",
                                f"{max_f1_row['R']:.2f}",
                                f"{max_f1_row['F1']:.2f}",
                            ]
                        )
                        latex_table.append(latex_row)
                    except FileNotFoundError:
                        print(f"File not found: {file_path}")

    # LaTeX表のフォーマット
    columns = []
    if "Dataset" in param_axes:
        columns.append("Dataset")
    if "Encoder" in param_axes:
        columns.append("Encoder")
    if "Direction" in param_axes:
        columns.append("Direction")
    if "Test:Train" in param_axes:
        columns.append("Test:Train")
    if "Seq Len" in param_axes:
        columns.append("Seq Len")
    columns.extend(["FP", "TP", "TN", "FN", "Precision", "Recall", "F1"])

    latex_output = (
        "\n\\begin{table}[]\n\\begin{tabular}{"
        + "l" * len(columns)
        + "}\n\\hline\n"
        + " & ".join([f"\\multicolumn{{1}}{{c}}{{{col}}}" for col in columns])
        + " \\\\ \hline\n"
    )

    for row in latex_table:
        latex_output += " & ".join(row) + " \\\\ \n"

    latex_output += "\\hline\n\\end{tabular}\n"
    if caption_parts:
        latex_output += "\\caption{" + ", ".join(caption_parts) + "}\n"
    else:
        latex_output += "\\caption{Results for multiple parameters}\n"
    latex_output += "\\end{table}"

    # 結果を保存
    with open("./results_table_r.tex", "w") as tex_file:
        tex_file.write(latex_output)


@click.command()
@click.argument(
    "encoder_names",
    callback=lambda ctx, param, value: value.split(","),
)
@click.argument(
    "dataset_names",
    callback=lambda ctx, param, value: value.split(","),
)
@click.argument(
    "train_ratios",
    callback=lambda ctx, param, value: [int(r) for r in value.split(",")],
)
@click.argument(
    "seq_lens",
    callback=lambda ctx, param, value: [int(l) for l in value.split(",")],
)
def main(encoder_names, dataset_names, train_ratios, seq_lens):
    make_table(encoder_names, dataset_names, train_ratios, seq_lens)


if __name__ == "__main__":
    main()
