import pandas as pd
import torch
from loss.loss import suggest_loss

from dataset.dataset_util import suggest_dataloader
from train_val import train, val

from util import (
    setup_config,
    fixed_r_seed,
    setup_device,
    suggest_network,
    suggest_optimizer,
    suggest_scheduler,
    plot_log,
    save_learner,
)


def main(cfg):
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
        if (epoch) % cfg.default.check_val_every_n_epoch == 0:
            save_learner(cfg, model, device)
        if (
            best_model_score > result_df[f"{cfg.default.monitor.name}"].iloc[-1]
            and cfg.default.monitor.mode == "min"
        ):
            save_learner(cfg, model, device, True)
            best_model_score = result_df[f"{cfg.default.monitor.name}"].iloc[-1]
            best_epoch = epoch
        if (
            best_model_score < result_df[f"{cfg.default.monitor.name}"].iloc[-1]
            and cfg.default.monitor.mode == "max"
        ):
            save_learner(cfg, model, device, True)
            best_model_score = result_df[f"{cfg.default.monitor.name}"].iloc[-1]
            best_epoch = epoch
        print(f"Best epoch = {best_epoch}")


if __name__ == "__main__":
    cfg = setup_config()
    print("setup")
    main(cfg)
