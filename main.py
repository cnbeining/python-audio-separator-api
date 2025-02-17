import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

import litserve as ls
import torch
from fastapi import HTTPException, UploadFile, Request
from fastapi.responses import FileResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

logger = logging.getLogger(__name__)

# Load API key from environment variable
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY environment variable must be set")

class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            raise HTTPException(
                status_code=401,
                detail="API key missing. Please provide it in the X-API-Key header",
            )
        if api_key != API_KEY:
            raise HTTPException(
                status_code=403,
                detail="Invalid API key",
            )
        return await call_next(request)

class AudioSeparatorLitAPI(ls.LitAPI):
    def setup(self, device: str = None) -> None:
        try:
            from audio_separator.separator import Separator

            # Pass the device to use_autocast and configure device-specific settings
            use_autocast = device == "cuda"
            self.separator = Separator(use_autocast=use_autocast, output_format="mp3")
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

        temp_dir = Path("/tmp")
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
            output_names = {
                "Vocals": f"{x}_vocals.wav",
            }
            self.separator.separate(x, output_names=output_names)

            # Clean up the original input file
            try:
                os.remove(x)
            except Exception as e:
                logger.warning(f"Failed to remove input file {x}: {str(e)}")

            return f"{x}_vocals.wav"
        except Exception as e:
            logger.error(f"Separation failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Audio separation failed")

    def encode_response(self, output: str) -> FileResponse:
        if not os.path.exists(output):
            raise HTTPException(status_code=404, detail="Output file not found")

        response = FileResponse(output)

        # Clean up the output file after sending
        def cleanup(output_path: str = output):
            try:
                os.remove(output_path)
                logger.info(f"Cleaned up output file: {output_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up file {output_path}: {str(e)}")

        response.background = cleanup
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
    
    # Add the API key middleware
    server.app.add_middleware(APIKeyMiddleware)
    
    server.run(port=8000)
