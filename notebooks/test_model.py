import yaml
import torch

from src.models.cnn_lstm import CNNLSTM
from src.data.dataset import build_dataloaders


# --------------------------------------------------
# load params
# --------------------------------------------------

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)


# --------------------------------------------------
# device
# --------------------------------------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    and params["device"]["use_cuda"]
    else "cpu"
)

print("Device:", device)


# --------------------------------------------------
# dataloaders
# --------------------------------------------------

train_loader, _, _ = build_dataloaders(params)

imgs, coords, targets = next(iter(train_loader))

print("Images:", imgs.shape)
print("Coords:", coords.shape)
print("Targets:", targets.shape)


# --------------------------------------------------
# build model
# --------------------------------------------------

model = CNNLSTM(

    feature_dim=params["model"]["feature_dim"],

    hidden_dim=params["model"]["hidden_dim"],

    dropout=params["model"]["dropout"]

).to(device)

print(model)

# --------------------------------------------------
# move tensors to device
# --------------------------------------------------

imgs = imgs.to(device)
coords = coords.to(device)


# --------------------------------------------------
# test forward_sequence
# --------------------------------------------------

preds = model.forward_sequence(imgs)

print("Sequence predictions:", preds.shape)


# --------------------------------------------------
# test forward_step
# --------------------------------------------------

B = imgs.shape[0]

hidden = model.init_hidden(
    batch_size=B,
    device=device
)

pred_step, hidden = model.forward_step(
    imgs[:,0],
    hidden
)

print("Step prediction:", pred_step.shape)

print("Hidden h:", hidden[0].shape)
print("Hidden c:", hidden[1].shape)


# --------------------------------------------------
# test backward pass
# --------------------------------------------------

loss = ((pred_step - coords[:,1]) ** 2).mean()

loss.backward()

print("Backward pass successful")
print("Loss:", loss.item())