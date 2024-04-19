# deep learning libraries
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder

# plotting libraries
import matplotlib.pyplot as plt

# other libraries
import ast
import os
import requests  # type: ignore
import shutil
import wfdb
import zipfile
from typing import List, Dict


class EKGDataset(Dataset):
    """
    This class represents the PTB-XL dataset, a large publicly available
    electrocardiography dataset.

    When using this dataset in your work, please cite the following:

    - The PTB-XL dataset itself:
        Wagner, P., Strodthoff, N., Bousseljot, R., Samek, W., & Schaeffter, T. (2022). PTB-XL,
        a large publicly available electrocardiography dataset (version 1.0.3).
        PhysioNet. https://doi.org/10.13026/kfzx-aw45.

    - The original publication of the dataset:
        Wagner, P., Strodthoff, N., Bousseljot, R.-D., Kreiseler, D., Lunze, F.I., Samek, W.,
        Schaeffter, T. (2020), PTB-XL: A Large Publicly Available ECG Dataset.
        Scientific Data. https://doi.org/10.1038/s41597-020-0495-6

    - The PhysioNet resource:
        Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... &
        Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new
        research resource for complex physiologic signals. Circulation [Online].
        101 (23), pp. e215-e220.
    """

    def __init__(self, X: List[str], y: List[List[str]], path: str) -> None:
        """
        Constructor of EKGDataset.

        Args:
            X (List[str]): The input data. Each row corresponds to the filename where the raw
            data is stored.

            y (List[List[str]]): The labels corresponding to the input data. Each element in the
            list is a list of strings, where each string is a diagnostic superclass for the
            corresponding EKG recording. The labels are binarized using a MultiLabelBinarizer to
            create a binary matrix indicating the presence of each diagnostic superclass
            for each EKG recording.

            path (str): the path where the data is stored.
        """

        self._path = path
        self.X = X

        # Create a LabelEncoder object
        self._le = LabelEncoder()

        # Fit the LabelEncoder to the labels and transform the labels to integers
        self.y = torch.tensor(self._le.fit_transform(y), dtype=torch.long)

    def __len__(self) -> int:
        """
        This method returns the length of the dataset.

        Returns:
            int: The number of EKG recordings in the dataset.
        """

        return len(self.X)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This method loads an item based on the index.

        Args:
            index (int): The index of the element in the dataset.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the EKG recording and its corresponding labels.
            The EKG recording is a 1D tensor where each element is a lead of the EKG, and the labels are a 1D
            tensor of binary values indicating the presence of each diagnostic superclass for the EKG recording.
        """

        return self.load_raw_data(index), self.y[index]

    def load_raw_data(self, index: int):
        """
        Load raw data from a specified index.

        Args:
            index (int): The index where the data is stored.

        Returns:
            np.ndarray: The loaded raw data.
        """
        ekg_data = wfdb.rdsamp(self._path + self.X[index])[0]
        return np.array(ekg_data)


def aggregate_diagnostic(y_dic: Dict, agg_df: pd.DataFrame) -> List[str]:
    """
    Aggregate diagnostics from a dictionary.

    Args:
        y_dic (Dict): The dictionary containing diagnostic data.
        agg_df (pd.DataFrame): DataFrame for diagnostic aggregation.

    Returns:
        List[str]: The aggregated diagnostics.
    """

    return list(
        set(
            agg_df.loc[key].diagnostic_class
            for key in y_dic.keys()
            if key in agg_df.index
        )
    )


def load_ekg_data(
    path: str,
    sampling_rate: int = 100,
    batch_size: int = 128,
    shuffle: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
):
    """
    Load EKG data, split it into train, validation, and test sets, and return dataloaders for each set.

    Args:
        path (str): The path where the data is stored.
        sampling_rate (int, optional): The sampling rate of the data. Defaults to 100.
        batch_size (int, optional): The batch size for the dataloaders. Defaults to 128.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        drop_last (bool, optional): Whether to drop the last incomplete batch. Defaults to False.
        num_workers (int, optional): The number of worker processes for data loading. Defaults to 0.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Dataloaders for the train, validation, and test sets.
    """

    if not os.path.isdir(f"{path}"):
        os.makedirs(f"{path}")
        download_data(path)

    Y = pd.read_csv(path + "ptbxl_database.csv", index_col="ecg_id")
    Y.scp_codes = Y.scp_codes.apply(ast.literal_eval)
    X = Y.filename_lr if sampling_rate == 100 else Y.filename_hr
    X = X.to_numpy()

    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(path + "scp_statements.csv", index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    Y["diagnostic_superclass"] = Y.scp_codes.apply(
        lambda x: aggregate_diagnostic(x, agg_df)
    )

    # Repeat the EKG signals for each label
    X_repeat = np.repeat(X, [len(labels) for labels in Y.diagnostic_superclass], axis=0)

    # Flatten the list of labels
    y_flatten = [label for sublist in Y.diagnostic_superclass for label in sublist]

    # Define the test fold
    test_fold = 10

    # Create a mask for the train + val set
    train_val_mask = np.repeat(
        (Y.strat_fold != test_fold).values,
        [len(labels) for labels in Y.diagnostic_superclass],
    )

    # Train + Val
    X_train_val = X_repeat[train_val_mask]
    y_train_val = [y_flatten[i] for i in np.where(train_val_mask)[0]]
    combined_dataset = EKGDataset(X_train_val, y_train_val, path)

    # Use random_split to split the data
    train_dataset, val_dataset = random_split(combined_dataset, [0.8, 0.2])

    # Test
    test_mask = np.repeat(
        (Y.strat_fold == test_fold).values,
        [len(labels) for labels in Y.diagnostic_superclass],
    )
    X_test = X_repeat[test_mask]
    y_test = [y_flatten[i] for i in np.where(test_mask)[0]]
    test_dataset = EKGDataset(X_test, y_test, path)
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


def plot_ekg(
    dataloader: DataLoader, sampling_rate: int = 100, num_plots: int = 5
) -> None:
    """
    Plot EKG signals from a dataloader.

    Args:
        dataloader (DataLoader): The dataloader containing the EKG signals and labels.
        sampling_rate (int, optional): The sampling rate of the EKG signals. Defaults to 100.
        num_plots (int, optional): The number of EKG signals to plot. Defaults to 5.
    """

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


def download_data(path: str) -> None:
    """
    Download and extract data from a specified URL, and remove an unnecessary folder.

    Args:
        path (str): The path where the data will be downloaded and extracted.
    """

    url: str = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
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


# train_loader, val_loader, _ = load_ekg_data("./data/")
# plot_ekg(train_loader)
