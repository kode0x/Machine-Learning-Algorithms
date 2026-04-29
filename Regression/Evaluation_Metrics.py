import numpy as np

class MeanAbsoluteError:
    """
    A Class To Compute Mean Absolute Error (MAE) Between True And Predicted Values
    """

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Initialize With True And Predicted Values

        Args:
            y_true (np.ndarray): Ground Truth Values
            y_pred (np.ndarray): Predicted Values
        """
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)

        self._validate_inputs()

    def _validate_inputs(self):
        """Ensure Inputs Are Valid And Compatible."""
        if self.y_true.shape != self.y_pred.shape:
            raise ValueError("y_true And y_pred Must Have The Same Shape")

        if self.y_true.size == 0:
            raise ValueError("Input Arrays Must Not Be Empty")

    def calculate(self) -> float:
        """
        Compute The Mean Absolute Error

        Returns:
            float: The MAE Value
        """
        absolute_errors = np.abs(self.y_true - self.y_pred)
        return float(np.mean(absolute_errors))

    def __call__(self) -> float:
        """Allow The Instance To Be Called Like A Function"""
        return self.calculate()

class MeanSquaredError:
    """
    A Class To Compute Mean Squared Error (MSE) Between True And Predicted Values
    """

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Initialize With True And Predicted Values

        Args:
            y_true (np.ndarray): Ground Truth Values
            y_pred (np.ndarray): Predicted Values
        """
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)

        self._validate_inputs()

    def _validate_inputs(self):
        """Ensure Inputs Are Valid And Compatible."""
        if self.y_true.shape != self.y_pred.shape:
            raise ValueError("y_true And y_pred Must Have The Same Shape")

        if self.y_true.size == 0:
            raise ValueError("Input Arrays Must Not Be Empty")

    def calculate(self) -> float:
        """
        Compute The Mean Squared Error

        Returns:
            float: The MSE Value
        """
        squared_errors = np.square(self.y_true - self.y_pred)
        return float(np.mean(squared_errors))

    def __call__(self) -> float:
        """Allow The Instance To Be Called Like A Function"""
        return self.calculate()