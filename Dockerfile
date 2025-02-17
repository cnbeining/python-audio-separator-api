FROM beveradb/audio-separator:gpu

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

RUN pip install fastapi litserve python-multipart

COPY . /app

WORKDIR /app

# force the model to be downloaded
RUN audio-separator --download_model_only

CMD ["python", "main.py"]