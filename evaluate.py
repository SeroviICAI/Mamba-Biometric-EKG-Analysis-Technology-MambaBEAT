import torch
from src.binary_classification.data import load_ekg_data
from src.utils.torchutils import set_seed, load_model
from src.utils.metrics import Accuracy, BinaryAccuracy

# static variables
DATA_PATH: str = "./data/"
NUM_CLASSES: int = 5

# set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
set_seed(42)


def main(name, accuracy: Accuracy = Accuracy()) -> None:
    """
    This function is the main program.
    """

    # Load test data
    _, _, test_data = load_ekg_data(DATA_PATH, num_workers=4)

    # Load mode
    model = load_model(name).to(device).double()

    with torch.no_grad():
        for inputs, _, targets in test_data:
            inputs, targets = inputs.to(device), targets.to(device)

            # forward
            outputs = model(inputs)

            # Compute accuracy
            accuracy.update(outputs, targets)
        # show the accuracy
        print(f"Accuracy: {accuracy.compute()}")
        accuracy.reset()


if __name__ == "__main__":
    path = "path_to_model"
    # change to BinaryAccuracy() if model has a binary approach
    accuracy = Accuracy()
    main(path, accuracy)
