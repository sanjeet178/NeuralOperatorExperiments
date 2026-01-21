import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging

def plotAndSaveScatter(
    selectedYComp,
    selectedYTest,
    saveDir,
    baseName,
    prefix="comparison"
):
    """
    X, Y, selectedYComp, selectedYTest:
        torch.Tensor of shape (B, 1, H, W) or (B, 1, ...)
    """

    os.makedirs(saveDir, exist_ok=True)

    # Squeeze dim=1 and convert to numpy
    YComp_np = selectedYComp.squeeze(1).detach().cpu().numpy()
    YTest_np = selectedYTest.squeeze(1).detach().cpu().numpy()

    batch_size = YComp_np.shape[0]

    for i in range(batch_size):

        # Shared color limits
        vmin = min(YComp_np[i].min(), YTest_np[i].min())
        vmax = max(YComp_np[i].max(), YTest_np[i].max())

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        im1 = axes[0].imshow(
            YComp_np[i],
            cmap="jet",
            origin="lower",
            aspect="auto",
            vmin=vmin,
            vmax=vmax
        )
        axes[0].set_title("YComp")
        plt.colorbar(im1, ax=axes[0])

        im2 = axes[1].imshow(
            YTest_np[i],
            cmap="jet",
            origin="lower",
            aspect="auto",
            vmin=vmin,
            vmax=vmax
        )
        axes[1].set_title("YTest")
        plt.colorbar(im2, ax=axes[1])

        plt.tight_layout()
        plt.savefig(
            os.path.join(saveDir, f"{prefix}_{baseName}_{i}.png"),
            dpi=300
        )
        plt.close(fig)