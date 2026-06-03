from typing import Optional
import torch
import tensorflow as tf
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoader:
    def __init__(self):
        self.tf_model: Optional[tf.keras.Model] = None
        self.pt_model: Optional[torch.nn.Module] = None
        self.model_type: Optional[str] = None

    def load_tensorflow_model(self, path: str) -> bool:
        """
        Load a TensorFlow model from a file path.
        """
        try:
            self.tf_model = tf.keras.models.load_model(path)
            self.model_type = "tensorflow"
            logger.info(f"TensorFlow model loaded successfully from {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load TensorFlow model: {e}")
            return False

    def load_pytorch_model(self, path: str, model_class=None) -> bool:
        """
        Load a PyTorch model from a .pt or .pth file.
        """
        try:
            if model_class is not None:
                model = model_class()
                model.load_state_dict(torch.load(path))
                model.eval()
                self.pt_model = model
            else:
                self.pt_model = torch.load(path)

            self.model_type = "pytorch"

            logger.info(f"PyTorch model loaded successfully from {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            return False

    def get_model_info(self) -> dict:
        """
        Return information about the currently loaded model.
        """
        return {
            "model_type": self.model_type,
            "tensorflow_loaded": self.tf_model is not None,
            "pytorch_loaded": self.pt_model is not None,
            "status": "loaded" if self.model_type else "not_loaded"
        } 
    
if __name__ == "__main__":
    loader = ModelLoader()
    print(loader.get_model_info())