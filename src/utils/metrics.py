# deep learning libraries
import torch


class Accuracy:
    """
    This class is the accuracy object.

    Attributes:
        correct: number of correct predictions.
        total: number of total examples to classify.
    """

    correct: int
    total: int

    def __init__(self) -> None:
        """
        This is the constructor of Accuracy class. It should
        initialize correct and total to zero.
        """

        self.correct = 0
        self.total = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """
        This method update the value of correct and total counts.

        Args:
            logits: outputs of the model.
                Dimensions: [batch, number of classes]
            labels: labels of the examples. Dimensions: [batch].
        """

        # compute predictions
        predictions = logits.argmax(1).type_as(labels)

        # update counts
        self.correct += int(predictions.eq(labels).sum().item())
        self.total += labels.shape[0]

        return None

    def compute(self) -> float:
        """
        This method returns the accuracy value.

        Returns:
            accuracy value.
        """

        return self.correct / self.total

    def reset(self) -> None:
        """
        This method resets to zero the count of correct and total number of
        examples.
        """

        # init to zero the counts
        self.correct = 0
        self.total = 0

        return None


class BinaryAccuracy(Accuracy):
    """
    This class is the accuracy object.

    Attributes:
        correct: number of correct predictions.
        total: number of total examples to classify.
    """

    correct: int
    total: int

    def __init__(self, threshold: float = 0.5) -> None:
        """
        This is the constructor of Accuracy class. It should
        initialize correct and total to zero.
        """

        super().__init__()
        self.treshold = threshold

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """
        This method update the value of correct and total counts.

        Args:
            logits: outputs of the model.
                Dimensions: [batch, number of classes]
            labels: labels of the examples. Dimensions: [batch].
        """

        # compute predictions
        predictions = torch.where(logits > self.treshold, 1, 0)

        # update counts
        self.correct += int(predictions.eq(labels).sum().item())
        self.total += labels.nelement()

        return None
