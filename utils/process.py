import copy
import logging
from typing import Optional, NoReturn
import os
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
from pathlib import Path

from .hand_visualize import Hand, save_animation
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def save_checkpoint_custom(state, best_model_path):
    """
    Save checkpoint based on state information. Save model.state_dict() weight of models.
    Parameters:
    state: torch dict weights
        model.state_dict()
    best_model_path: str
        path to save best model( copy from checkpoint_path)
    """

    best_check = os.path.split(best_model_path)[0]
    if not os.path.exists(best_check):
        os.makedirs(best_check)

    torch.save(state, best_model_path)


def aggregate_batch_results(
    previous_aggregation: dict, new_results: dict, inplace: bool = True
) -> dict:
    if inplace:
        next_aggregation = previous_aggregation
    else:
        next_aggregation = copy.copy(
            previous_aggregation
        )  # Deepcopy for more complex aggregations in future

    for key in new_results:
        next_aggregation[key] += new_results[key]

    return next_aggregation


def train_epoch(
    model,
    dataloader,
    optimizer,
    loss_func,
    device,
    scheduler: Optional = None,
    is_batch_scheduler: bool = False,
) -> dict:

    model.train()

    batches_pbar = tqdm(enumerate(dataloader), leave=True)
    batches_pbar.set_description("Train process through batches")
    for step, data in batches_pbar:
        batch_results = train_batch(
            model=model,
            optimizer=optimizer,
            loss_func=loss_func,
            device=device,
            data=data,
        )

        if scheduler is not None and is_batch_scheduler:
            scheduler.step()

        # Aggregate with previous batches
        if not step:
            epoch_results = copy.copy(batch_results)
        else:
            epoch_results = aggregate_batch_results(
                epoch_results, batch_results, inplace=False
            )

    if scheduler is not None and not is_batch_scheduler:
        scheduler.step()

    epoch_results = {k: v / step for (k, v) in epoch_results.items()}

    return epoch_results


def train_batch(model, optimizer, loss_func, device, data) -> dict:
    """
    model: must be already on device
    """
    model.train()

    optimizer.zero_grad()

    model_input, target = data
    model_input, target = model_input.to(device), target.to(device)

    model_output = model(model_input)

    losses: dict = loss_func(target, model_output)
    losses["total_loss"].backward()

    optimizer.step()

    losses = {k: v.item() for (k, v) in losses.items()}

    return losses


@torch.no_grad()
def val_epoch(model, dataloader, loss_func, device) -> dict:
    """
    model: must be already on device
    """
    model.eval()

    batches_pbar = tqdm(enumerate(dataloader), leave=True)
    batches_pbar.set_description("Val process through batches")
    for step, data in batches_pbar:
        batch_results = val_batch(
            model=model, loss_func=loss_func, device=device, data=data
        )

        # Aggregate with previous batches
        if not step:
            epoch_results = copy.copy(batch_results)
        else:
            epoch_results = aggregate_batch_results(
                epoch_results, batch_results, inplace=False
            )

    epoch_results = {k: v / step for (k, v) in epoch_results.items()}

    return epoch_results


@torch.no_grad()
def val_batch(model, loss_func, device, data) -> dict:
    """
    model: must be already on device
    """
    model.eval()

    model_input, target = data
    model_input, target = model_input.to(device), target.to(device)

    model_output = model(model_input)
    losses: dict = loss_func(target, model_output)

    losses = {k: v.item() for (k, v) in losses.items()}

    return losses


def log_train_results(results, epoch):
    # wandb logging
    for key in results.keys():
        wandb.log({"train/" + str(key): results[key]}, epoch)


def visualize_val_moves(model, val_exps_data, epoch, device):
    old_fps = 200
    new_fps = 20
    step = old_fps // new_fps

    for n, raw_data in enumerate(val_exps_data):

        x, y = raw_data["data_myo"][:256], raw_data["data_vr"][:256]

        y_pred = model.inference(x, device)

        hand_gt = Hand(y[::step])
        hand_pred = Hand(y_pred[::step])

        gt_path = f"{wandb.run.dir}/videos/{n}_move/true_sample_{epoch}.gif"
        pred_path = f"{wandb.run.dir}/videos/{n}_move/pred_sample_{epoch}.gif"
        Path(gt_path).parent.mkdir(parents=True, exist_ok=True)

        plt.close()
        ani_gt = hand_gt.visualize_all_frames()
        save_animation(ani_gt, gt_path, fps=new_fps)

        plt.close()
        ani_pred = hand_pred.visualize_all_frames()
        save_animation(ani_pred, pred_path, fps=new_fps)

        wandb.log(
            {
                f"visualization/{n}_move": [
                    wandb.Video(gt_path, fps=new_fps),
                    wandb.Video(pred_path, fps=new_fps),
                ]
            },
            epoch,
        )
        del hand_gt
        del hand_pred


def log_val_results(results, epoch):
    # wandb logging
    for key in results.keys():
        wandb.log({"val/" + str(key): results[key]}, epoch)


def train_epochs(
    model: nn.Module,
    device,
    max_epochs: int,
    loss_func,
    # train-specific
    train_dataloader: torch.utils.data.DataLoader,
    scheduler,
    optimizer,
    is_batch_scheduler: bool,
    # val-specific
    val_dataloader: torch.utils.data.DataLoader,
    val_every: int = 1,
) -> NoReturn:

    val_preproc_data = val_dataloader.dataset.exps_data  # data for visualization.
    min_val_loss = 100000

    for epoch in range(1, max_epochs + 1):
        model.to(device)
        train_epoch_results = train_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            loss_func=loss_func,
            device=device,
            scheduler=scheduler,
            is_batch_scheduler=is_batch_scheduler,
        )
        log_train_results(train_epoch_results, epoch)

        val_epoch_results = val_epoch(
            model=model, dataloader=val_dataloader, loss_func=loss_func, device=device
        )
        log_val_results(val_epoch_results, epoch)

        # criterion on saving model and visualization.

        val_loss = val_epoch_results["total_loss"]

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            # save animation
            visualize_val_moves(model, val_preproc_data, epoch, device)

            # save model weights
            filename = "epoch_{}_val_loss{:.2}.pt".format(epoch, val_loss)
            filepath_name = os.path.join(wandb.run.dir, filename)
            save_checkpoint_custom(model.state_dict(), filepath_name)
