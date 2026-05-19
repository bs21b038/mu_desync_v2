import os
import random
import yaml

import mlflow
import mlflow.pytorch

import numpy as np

import torch
import torch.nn as nn

from tqdm import tqdm

from src.models.cnn_lstm import CNNLSTM
from src.data.dataset import build_dataloaders
from src.data.renderer import TensorRenderer


# --------------------------------------------------
# load params
# --------------------------------------------------

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)


# --------------------------------------------------
# reproducibility
# --------------------------------------------------

seed = params["seed"]

random.seed(seed)
np.random.seed(seed)

torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------
# device
# --------------------------------------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    and params["device"]["use_cuda"]
    else "cpu"
)

print("Using device:", device)


# --------------------------------------------------
# dataloaders
# --------------------------------------------------

train_loader, val_loader, _ = build_dataloaders(params)


# --------------------------------------------------
# renderer
# --------------------------------------------------

renderer = TensorRenderer(

    image_size=params["dataset"]["image_size"],

    xlim=tuple(params["renderer"]["xlim"]),

    ylim=tuple(params["renderer"]["ylim"]),

    arm_radius=params["renderer"]["arm_radius"],

    target_size=params["renderer"]["target_size"]
)


# --------------------------------------------------
# model
# --------------------------------------------------

model = CNNLSTM(

    feature_dim=params["model"]["feature_dim"],

    hidden_dim=params["model"]["hidden_dim"],

    dropout=params["model"]["dropout"]

).to(device)


# --------------------------------------------------
# optimizer
# --------------------------------------------------

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=params["training"]["learning_rate"]
)


# --------------------------------------------------
# loss
# --------------------------------------------------

loss_fn = nn.MSELoss()


# --------------------------------------------------
# teacher forcing schedule
# --------------------------------------------------

def teacher_forcing_probability(epoch):

    start = params["training"]["teacher_forcing"]["start"]

    end = params["training"]["teacher_forcing"]["end"]

    total_epochs = params["training"]["epochs"]

    frac = epoch / total_epochs

    tf_prob = start + (end - start) * frac

    return tf_prob


# --------------------------------------------------
# checkpoint directory
# --------------------------------------------------

os.makedirs("checkpoints", exist_ok=True)


# --------------------------------------------------
# mlflow setup
# --------------------------------------------------

mlflow.set_experiment(
    params["logging"]["experiment_name"]
)


# --------------------------------------------------
# training
# --------------------------------------------------

best_val_loss = float("inf")

epochs = params["training"]["epochs"]

with mlflow.start_run():

    # ----------------------------------------------
    # log params
    # ----------------------------------------------

    mlflow.log_params({

        "epochs": epochs,

        "batch_size":
        params["training"]["batch_size"],

        "learning_rate":
        params["training"]["learning_rate"],

        "feature_dim":
        params["model"]["feature_dim"],

        "hidden_dim":
        params["model"]["hidden_dim"],

        "dropout":
        params["model"]["dropout"]
    })

    # ----------------------------------------------
    # epoch loop
    # ----------------------------------------------

    for epoch in range(1, epochs + 1):

        # ==========================================
        # TRAINING
        # ==========================================

        model.train()

        train_loss = 0.0

        tf_prob = teacher_forcing_probability(epoch)

        train_bar = tqdm(
            train_loader,
            desc=f"Train Epoch {epoch}"
        )

        for imgs, coords, targets in train_bar:

            imgs = imgs.to(device)

            coords = coords.to(device)

            targets = targets.to(device)

            B, T = coords.shape[:2]

            hidden = model.init_hidden(
                batch_size=B,
                device=device
            )

            # --------------------------------------
            # first frame input
            # --------------------------------------

            prev_img = imgs[:, 0]

            total_loss = 0.0

            # --------------------------------------
            # rollout loop
            # --------------------------------------

            for t in range(1, T):

                # ----------------------------------
                # predict next coords
                # ----------------------------------

                pred_coords, hidden = model.forward_step(
                    prev_img,
                    hidden
                )

                # ----------------------------------
                # ground truth coords
                # ----------------------------------

                gt_coords = coords[:, t]

                # ----------------------------------
                # mse loss
                # ----------------------------------

                loss = loss_fn(
                    pred_coords,
                    gt_coords
                )

                total_loss += loss

                # ----------------------------------
                # teacher forcing decision
                # ----------------------------------

                use_teacher_forcing = (
                    random.random() < tf_prob
                )

                # ----------------------------------
                # next input image
                # ----------------------------------

                if use_teacher_forcing:

                    prev_img = imgs[:, t]

                else:

                    rendered_imgs = []

                    for b in range(B):

                        pred_np = (
                            pred_coords[b]
                            .detach()
                            .cpu()
                            .numpy()
                        )

                        target_np = (
                            targets[b]
                            .detach()
                            .cpu()
                            .numpy()
                        )

                        rendered_img = renderer.render(
                            pred_np,
                            target_np
                        )

                        rendered_imgs.append(
                            rendered_img
                        )

                    prev_img = torch.stack(
                        rendered_imgs
                    ).to(device)

            # --------------------------------------
            # average rollout loss
            # --------------------------------------

            total_loss = total_loss / (T - 1)

            # --------------------------------------
            # backward
            # --------------------------------------

            optimizer.zero_grad()

            total_loss.backward()

            optimizer.step()

            train_loss += total_loss.item()

            train_bar.set_postfix({
                "loss": total_loss.item(),
                "tf": round(tf_prob, 3)
            })

        train_loss /= len(train_loader)

        # ==========================================
        # VALIDATION
        # ==========================================

        model.eval()

        val_loss = 0.0

        with torch.no_grad():

            val_bar = tqdm(
                val_loader,
                desc=f"Val Epoch {epoch}"
            )

            for imgs, coords, targets in val_bar:

                imgs = imgs.to(device)

                coords = coords.to(device)

                targets = targets.to(device)

                B, T = coords.shape[:2]

                hidden = model.init_hidden(
                    batch_size=B,
                    device=device
                )

                prev_img = imgs[:, 0]

                total_loss = 0.0

                # ----------------------------------
                # FULL autoregressive rollout
                # NO teacher forcing
                # ----------------------------------

                for t in range(1, T):

                    pred_coords, hidden = (
                        model.forward_step(
                            prev_img,
                            hidden
                        )
                    )

                    gt_coords = coords[:, t]

                    loss = loss_fn(
                        pred_coords,
                        gt_coords
                    )

                    total_loss += loss

                    # ----------------------------------
                    # render predicted frame
                    # ----------------------------------

                    rendered_imgs = []

                    for b in range(B):

                        pred_np = (
                            pred_coords[b]
                            .cpu()
                            .numpy()
                        )

                        target_np = (
                            targets[b]
                            .cpu()
                            .numpy()
                        )

                        rendered_img = renderer.render(
                            pred_np,
                            target_np
                        )

                        rendered_imgs.append(
                            rendered_img
                        )

                    prev_img = torch.stack(
                        rendered_imgs
                    ).to(device)

                total_loss = total_loss / (T - 1)

                val_loss += total_loss.item()

                val_bar.set_postfix({
                    "loss": total_loss.item()
                })

        val_loss /= len(val_loader)

        # ==========================================
        # logging
        # ==========================================

        print(
            f"\nEpoch {epoch}"
            f" | Train {train_loss:.6f}"
            f" | Val {val_loss:.6f}"
            f" | TF {tf_prob:.3f}"
        )

        mlflow.log_metric(
            "train_loss",
            train_loss,
            step=epoch
        )

        mlflow.log_metric(
            "val_loss",
            val_loss,
            step=epoch
        )

        mlflow.log_metric(
            "teacher_forcing_probability",
            tf_prob,
            step=epoch
        )

        # ==========================================
        # save best model
        # ==========================================

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            checkpoint_path = (
                "checkpoints/best_model.pt"
            )

            torch.save({

                "model_state_dict":
                model.state_dict(),

                "optimizer_state_dict":
                optimizer.state_dict(),

                "epoch":
                epoch,

                "val_loss":
                val_loss

            }, checkpoint_path)

            mlflow.log_artifact(
                checkpoint_path
            )

            print(
                f"Saved best model "
                f"(val={val_loss:.6f})"
            )

    # --------------------------------------------------
    # save final model
    # --------------------------------------------------

    mlflow.pytorch.log_model(
        model,
        artifact_path="final_model"
    )

print("\nTraining complete.")