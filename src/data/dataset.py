import torch
from torch.utils.data import Dataset, DataLoader
import os


# ---------------------------------------------------
# trajectory dataset
# ---------------------------------------------------
class TrajectoryDataset(Dataset):

    def __init__(self, data_path):

        data = torch.load(data_path)

        self.images = data["images"]
        self.coords = data["coords"]
        self.targets = data["targets"]

    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):

        return (
            self.images[idx],
            self.coords[idx],
            self.targets[idx]
        )


# ---------------------------------------------------
# dataloader builder
# ---------------------------------------------------
def build_dataloaders(params):

    batch_size = params["training"]["batch_size"]
    num_workers = params["training"]["num_workers"]
    processed_dir = params["dataset"]["processed_dir"]

    train_dataset = TrajectoryDataset(
        os.path.join(processed_dir, params["dataset"]["train_file"])
    )

    val_dataset = TrajectoryDataset(
        os.path.join(processed_dir, params["dataset"]["val_file"])
    )

    test_dataset = TrajectoryDataset(
        os.path.join(processed_dir, params["dataset"]["test_file"])
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader