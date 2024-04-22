import torch
from src.binary_classification.data import load_ekg_data
from src.utils.torchutils import set_seed, load_model
from src.utils.metrics import Accuracy

# static variables
DATA_PATH: str = "./data/"
NUM_CLASSES: int = 5

# set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
set_seed(42)


def main(name) -> None:
    """
    This function is the main program.
    """

    # Load test data
    _, _, test_data = load_ekg_data(DATA_PATH, num_workers=4)

    # Load mode
    model = load_model(name).to(device).double()

    # evaluate
    accuracy = Accuracy()

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
    name = "your_model.name"
    main(f"./models/{name}.pth")
