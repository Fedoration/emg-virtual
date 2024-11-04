import torch
import torch.nn as nn


def make_loss_function():
    criterion = nn.L1Loss()

    def loss_func(y_hat, y_batch):
        """
        [batch, n_bones, 4, time]
        """
        batch, n_bones, n_quat, time = y_hat.shape

        # batch, 15, 4, time -> batch*15*time, 4
        y_hat = torch.permute(y_hat, (0, 1, 3, 2))
        y_batch = torch.permute(y_batch, (0, 1, 3, 2))
        y_hat, y_batch = y_hat.reshape(-1, 4), y_batch.reshape(-1, 4)

        mult = torch.sum(y_hat * y_batch, dim=-1) ** 2
        angle_degree = torch.mean(
            torch.arccos(torch.clip((2 * mult - 1), -1, 1)) / torch.pi * 180
        )
        q_dist = torch.mean(1 - mult)  # (0, 1) range

        mae = criterion(y_batch, y_hat)

        total_loss = mae
        return {
            "total_loss": total_loss,
            "min_mae": mae,
            "angle_degree": angle_degree,
            "q_dist": q_dist,
        }

    return loss_func
