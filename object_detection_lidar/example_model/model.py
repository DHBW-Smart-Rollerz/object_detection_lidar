from pathlib import Path

import numpy as np
import yaml


class ExampleModel:
    """Example model class."""

    def __init__(self, root: str, model_config_path: str):
        """Initializes the ExampleModel.

        Args:
            root: Root path of the package
            model_config_path: Path to the model configuration
        """
        cfg = self._load_config(Path(root) / model_config_path)
        self._load_model(Path(root) / cfg["model_path"])

    def predict(self, image: np.ndarray) -> float:
        """Calculates the average of an image and returns the result with a bias.

        Args:
            image (np.ndarray): input image

        Returns:
            float: average of image with bias
        """
        # Just some example calculation with the image
        return np.mean(image) + 42

    def _load_config(self, path: Path) -> dict:
        """Loads the model configuration.

        Args:
            path (Path): path to the configuration

        Returns:
            dict: configuration
        """
        with path.open() as file:
            return yaml.safe_load(file)

    def _load_model(self, path: Path) -> None:
        """Loads the model.

        Args:
            path (Path): path to the model
        """
