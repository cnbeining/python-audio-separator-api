import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

import litserve as ls
import torch
from fastapi import HTTPException, UploadFile, Request
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

logger = logging.getLogger(__name__)

temp_dir = Path("/tmp")


class AudioSeparatorLitAPI(ls.LitAPI):
    def setup(self, device) -> None:
        try:
            from audio_separator.separator import Separator

            # Use the provided device
            print('Using device:', device)
            use_autocast = device == "cuda"
            if "cuda" in device:
                use_autocast = True
            self.separator = Separator(use_autocast=use_autocast, 
                      output_format="mp3", 
                      output_dir=temp_dir, 
                      output_single_stem="Vocals",
                      )
            self.separator.load_model(
                model_filename="model_bs_roformer_ep_368_sdr_12.9628.ckpt"
            )
            logger.info(f"Model loaded successfully on device: {device}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def decode_request(self, request: UploadFile) -> str:
        if not request.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = request.filename.split(".")
        if len(filename) < 2:
            raise HTTPException(status_code=400, detail="Invalid file format")

        path = temp_dir / f"{filename[0]}_{timestamp}.{filename[-1]}"

        try:
            with path.open("wb") as f:
                shutil.copyfileobj(request.file, f)
            logger.info(f"File saved successfully at {path}")
            return str(path)
        except Exception as e:
            logger.error(f"Failed to save file: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to save uploaded file")

    def predict(self, x: str) -> str:
        try:
            # Generate unique output filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            input_filename = Path(x).stem
            output_filename = f"{input_filename}_{timestamp}_vocals.mp3"
            
            output_names = {
                "Vocals": output_filename.replace('.mp3', ''),  # Separator will add .mp3 extension
            }
            self.separator.separate(x, output_names)

            # Get the output file path
            output_path = temp_dir / output_filename

            # Clean up the original input file
            try:
                os.remove(x)
            except Exception as e:
                logger.warning(f"Failed to remove input file {x}: {str(e)}")

            return str(output_path)
        except Exception as e:
            logger.error(f"Separation failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Audio separation failed")

    @staticmethod
    def cleanup_file(output_path: str):
        try:
            os.remove(output_path)
            logger.info(f"Cleaned up output file: {output_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up file {output_path}: {str(e)}")

    def encode_response(self, output: str) -> FileResponse:
        if not os.path.exists(output):
            raise HTTPException(status_code=404, detail="Output file not found")

        response = FileResponse(output, background=BackgroundTask(AudioSeparatorLitAPI.cleanup_file, output))
        return response


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Determine the appropriate accelerator
    if torch.cuda.is_available():
        accelerator = "cuda"
        logger.info("Using CUDA accelerator")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        accelerator = "mps"
        logger.info("Using MPS accelerator")
    else:
        accelerator = "cpu"
        logger.info("Using CPU accelerator")

    api = AudioSeparatorLitAPI()
    server = ls.LitServer(
        api, accelerator=accelerator, timeout=1000, workers_per_device=2
    )
    
    server.run(port=8000)
