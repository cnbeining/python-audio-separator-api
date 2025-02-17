FROM beveradb/audio-separator:gpu

RUN pip install fastapi litserve python-multipart

COPY . /app

WORKDIR /app

# force the model to be downloaded
RUN audio-separator --download_model_only

CMD ["python3", "main.py"]