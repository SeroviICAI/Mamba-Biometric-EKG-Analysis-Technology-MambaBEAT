# deep learning libraries
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MultiLabelBinarizer

# plotting libraries
import matplotlib.pyplot as plt

# other libraries
import ast
import os
import random
import requests  # type: ignore
import shutil
import wfdb
import zipfile


class EKGDataset(Dataset):
    """
    This class represents the PTB-XL dataset.
    """

    def __init__(self, X, y) -> None:
        """
        Constructor of EKGDataset.

        Args:
            X: The input data.
            y: The labels corresponding to the input data.
        """
        self.X = torch.from_numpy(X).float()

        # Create a MultiLabelBinarizer object
        mlb = MultiLabelBinarizer()

        # Fit the MultiLabelBinarizer to the labels and transform the labels to binary vectors
        self.y = torch.tensor(mlb.fit_transform(y), dtype=torch.float)

        # Store the classes (unique labels)
        self.classes_ = mlb.classes_

    def __len__(self) -> int:
        """
        This method returns the length of the dataset.

        Returns:
            length of dataset.
        """
        return len(self.X)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This method loads an item based on the index.

        Args:
            index: index of the element in the dataset.

        Returns:
            tuple with input data and label.
        """
        return self.X[index], self.y[index]


def load_raw_data(df, sampling_rate, path):
    filenames = df.filename_lr if sampling_rate == 100 else df.filename_hr
    data = [wfdb.rdsamp(path + f)[0] for f in filenames]
    return np.array(data)


def aggregate_diagnostic(y_dic, agg_df):
    return list(
        set(
            agg_df.loc[key].diagnostic_class
            for key in y_dic.keys()
            if key in agg_df.index
        )
    )


def plot_ekg(dataloader, sampling_rate=100, num_plots=5):
    # Get a batch of data
    ekg_signals, labels = next(iter(dataloader))

    # Define the grid and colors
    color_major = (1, 0, 0)
    color_minor = (1, 0.7, 0.7)
    color_line = (0, 0, 0.7)

    # Plot the first `num_plots` EKG signals
    for i in range(num_plots):
        # Convert tensor to numpy array and select all leads
        signal = ekg_signals[i].numpy()

        fig, axes = plt.subplots(signal.shape[1], 1, figsize=(10, 10), sharex=True)

        for c in np.arange(signal.shape[1]):
            # Set grid
            axes[c].grid(
                True, which="both", color=color_major, linestyle="-", linewidth=0.5
            )
            axes[c].minorticks_on()
            axes[c].grid(which="minor", linestyle=":", linewidth=0.5, color=color_minor)

            # Plot EKG signal in blue
            axes[c].plot(signal[:, c], color=color_line)

            # If it's not the last subplot, remove the x-axis label
            if c < signal.shape[1] - 1:
                axes[c].set_xticklabels([])
            else:
                # Set x-ticks for the last subplot
                axes[c].set_xticks(np.arange(0, len(signal[:, c]), step=sampling_rate))
                axes[c].set_xticklabels(
                    np.arange(0, len(signal[:, c]) / sampling_rate, step=1)
                )

        # Reduce the vertical distance between subplots
        plt.subplots_adjust(hspace=0.5)

        # Set y label in the middle left
        fig.text(0.04, 0.5, "Amplitude", va="center", rotation="vertical")

        # Set title for the entire figure
        axes[0].set_title(f"EKG Signal {i+1}, Label: {labels[i]}")

        # Set x label
        plt.xlabel("Time (seconds)")
        plt.tight_layout(pad=4, w_pad=1.0, h_pad=0.1)
        plt.show()


def load_ekg_data(
    path: str,
    sampling_rate: int = 100,
    batch_size: int = 128,
    shuffle: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
):
    if not os.path.isdir(f"{path}"):
        os.makedirs(f"{path}")
        download_data(path)

    Y = pd.read_csv(path + "ptbxl_database.csv", index_col="ecg_id")
    Y.scp_codes = Y.scp_codes.apply(ast.literal_eval)
    X = load_raw_data(Y, sampling_rate, path)

    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(path + "scp_statements.csv", index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    Y["diagnostic_superclass"] = Y.scp_codes.apply(
        lambda x: aggregate_diagnostic(x, agg_df)
    )

    # Split the data into train, validation, and test sets
    test_fold = 10

    # Train + Val
    X_train_val = X[np.where(Y.strat_fold != test_fold)]
    y_train_val = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
    combined_dataset = EKGDataset(X_train_val, y_train_val)

    # Determine the lengths of the splits
    train_len = int(len(combined_dataset) * 0.8)  # 80% for training
    val_len = len(combined_dataset) - train_len  # remaining for validation

    # Use random_split to split the data
    train_dataset, val_dataset = random_split(combined_dataset, [train_len, val_len])

    # Test
    X_test = X[np.where(Y.strat_fold == test_fold)]
    y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass
    test_dataset = EKGDataset(X_test, y_test)

    # Create dataloaders
    train_dataloader: DataLoader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )
    val_dataloader: DataLoader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )
    test_dataloader: DataLoader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )

    return train_dataloader, val_dataloader, test_dataloader


def download_data(path: str) -> None:
    url: str = (
        "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
    )
    target_path: str = path + "/ptb-xl.zip"
    unnecessary_folder_path: str = (
        path + "/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    )

    response: requests.Response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(target_path, "wb") as f:
            f.write(response.raw.read())

    with zipfile.ZipFile(target_path, "r") as zip_ref:
        zip_ref.extractall(path)

    # Move important directories out of the unnecessary folder
    for directory in os.listdir(unnecessary_folder_path):
        shutil.move(os.path.join(unnecessary_folder_path, directory), path)

    # Remove the unnecessary folder
    shutil.rmtree(unnecessary_folder_path)

    os.remove(target_path)


def set_seed(seed: int) -> None:
    """
    This function sets a seed and ensure a deterministic behavior.

    Args:
        seed: seed number to fix radomness.
    """

    # set seed in numpy and random
    np.random.seed(seed)
    random.seed(seed)

    # set seed and deterministic algorithms for torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Ensure all operations are deterministic on GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # for deterministic behavior on cuda >= 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    return None
