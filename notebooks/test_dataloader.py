import yaml

from src.data.dataset import build_dataloaders

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

train_loader, val_loader, test_loader = build_dataloaders(params)

imgs, coords, targets = next(iter(train_loader))

print("Images:", imgs.shape)
print("Coords:", coords.shape)
print("Targets:", targets.shape)