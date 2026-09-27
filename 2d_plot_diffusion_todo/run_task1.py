"""
Task 1: Swiss Roll 2D DDPM Runner & Figure Generator
Trains DDPM on 2D Swiss Roll, generates evaluation figures for report:
1. q_sample.png: Visualizing forward process q(x_t)
2. loss_curve.png: Training loss curve
3. swiss_roll_result.png: Generated samples overlaid with target distribution
4. Computes Chamfer Distance (< 20)
"""
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from dataset import TwoDimDataClass, get_data_iterator
from network import SimpleNet
from ddpm import BaseScheduler, DiffusionModule
from chamferdist import chamfer_distance

def main():
    device = "cuda:0" if torch.cuda.is_available() and False else "cpu"
    print(f"Running Task 1 on: {device}")

    out_dir = Path("task1_results")
    out_dir.mkdir(exist_ok=True, parents=True)

    # 1. Dataset
    target_ds = TwoDimDataClass(dataset_type='swiss_roll', N=1000000, batch_size=256)
    prior_ds = TwoDimDataClass(dataset_type='gaussian_centered', N=1000000, batch_size=256)

    # 2. Config & Model
    config = {
        "num_diffusion_steps": 1000,
        "dim_hids": [128, 128, 128],
        "lr": 1e-3,
        "batch_size": 128,
        "num_train_iters": 5000,
        "device": device,
    }

    network = SimpleNet(
        dim_in=2,
        dim_out=2,
        dim_hids=config["dim_hids"],
        num_timesteps=config["num_diffusion_steps"],
    )
    var_scheduler = BaseScheduler(config["num_diffusion_steps"])
    ddpm = DiffusionModule(network, var_scheduler).to(device)

    # 3. Figure of q_sample implementation (Report Item: 10pts)
    print("Generating forward diffusion figure q(x_t)...")
    num_vis = 500
    vis_x0 = target_ds[:num_vis].to(device)
    fig, axs = plt.subplots(1, 10, figsize=(28, 3))
    for i, t in enumerate(range(0, 500, 50)):
        t_tensor = (torch.ones(num_vis) * t).to(device)
        x_t = ddpm.q_sample(vis_x0, t_tensor).cpu().numpy()
        axs[i].scatter(x_t[:, 0], x_t[:, 1], color='steelblue', edgecolor='gray', s=5, alpha=0.7)
        axs[i].set_axis_off()
        axs[i].set_title(f'$q(\\mathbf{{x}}_{{{t}}})$', fontsize=12)
    plt.tight_layout()
    q_sample_path = out_dir / "task1_q_sample.png"
    plt.savefig(q_sample_path, dpi=200)
    plt.close()
    print(f"Saved q_sample figure to: {q_sample_path}")

    # 4. Training
    print("Starting training (5000 iterations)...")
    optimizer = torch.optim.Adam(ddpm.parameters(), lr=config["lr"])
    train_dl = torch.utils.data.DataLoader(target_ds, batch_size=config["batch_size"])
    train_iter = get_data_iterator(train_dl)

    losses = []
    pbar = tqdm(range(config["num_train_iters"]))
    for step in pbar:
        optimizer.zero_grad()
        batch_x = next(train_iter).to(device)
        loss = ddpm.compute_loss(batch_x)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)
        pbar.set_description(f"Loss: {loss_val:.4f}")

    # Save model
    ddpm.save(str(out_dir / "ddpm_swiss_roll.ckpt"))
    print(f"Model saved to {out_dir / 'ddpm_swiss_roll.ckpt'}")

    # 5. Loss Curve (Report Item)
    plt.figure(figsize=(8, 4))
    plt.plot(losses, color='tab:blue', alpha=0.8, linewidth=0.8)
    plt.title("Task 1: Training Loss Curve", fontsize=14)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("MSE Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    loss_path = out_dir / "task1_loss_curve.png"
    plt.savefig(loss_path, dpi=200)
    plt.close()
    print(f"Saved loss curve to: {loss_path}")

    # 6. Evaluation & Chamfer Distance (Report Item: 10pts)
    print("Sampling 2048 particles from trained DDPM for Chamfer Distance evaluation...")
    num_eval = 2048
    pc_ref = target_ds[:num_eval]
    with torch.no_grad():
        pc_gen = ddpm.p_sample_loop(shape=(num_eval, 2)).cpu()
        cd = chamfer_distance(
            pc_gen.reshape(-1, 2).numpy(),
            pc_ref.reshape(-1, 2).numpy(),
        )
    print(f"==========================================")
    print(f"DDPM Chamfer Distance: {cd.item():.4f} (Goal: < 20.0)")
    print(f"==========================================")

    # 7. Distribution Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pc_ref[:, 0], pc_ref[:, 1], alpha=0.3, label="Target distribution", color="tab:blue", s=10)
    ax.scatter(pc_gen[:, 0], pc_gen[:, 1], alpha=0.3, label="DDPM samples", color="tab:orange", s=10)
    ax.set_aspect("equal")
    ax.set_title(f"Target vs Generated Samples (CD = {cd.item():.2f})", fontsize=13)
    ax.legend(loc="upper right")
    dist_path = out_dir / "task1_distribution.png"
    plt.savefig(dist_path, dpi=200)
    plt.close()
    print(f"Saved distribution plot to: {dist_path}")

if __name__ == "__main__":
    main()
