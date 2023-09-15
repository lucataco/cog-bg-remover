# Prediction interface for Cog

from cog import BasePredictor, Input, Path
import os
import torch
from PIL import Image
from carvekit.web.schemas.config import MLConfig
from carvekit.web.utils.init_utils import init_interface

MODEL_CACHE = "model-cache/"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Make cache folders and copy over models
        os.makedirs("/root/.cache/carvekit/checkpoints/tracer_b7/")
        os.makedirs("/root/.cache/carvekit/checkpoints/fba/")
        os.system("cp "+MODEL_CACHE+"/tracer_b7.pth /root/.cache/carvekit/checkpoints/tracer_b7/")
        os.system("cp "+MODEL_CACHE+"/fba_matting.pth /root/.cache/carvekit/checkpoints/fba/")
        # Load the interface
        self.interface = init_interface(
            MLConfig(
                segmentation_network="tracer_b7",
                preprocessing_method="none",
                postprocessing_method="fba",
                seg_mask_size=640,
                trimap_dilation=30,
                trimap_erosion=5,
                device='cuda'
            )
        )

    def predict(
        self,
        image: Path = Input(description="Remove background from this image"),
    ) -> Path:
        """Run a single prediction on the model"""
        pil_image = Image.open(image)
        processed_bg = self.interface([pil_image])[0]
        save_path = "/tmp/output.png"
        processed_bg.save(save_path)
        return Path(save_path)
