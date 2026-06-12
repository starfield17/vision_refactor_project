FROM python:3.13-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY common ./common
COPY core ./core
COPY control_plane ./control_plane
COPY train_worker ./train_worker
COPY autolabel_worker ./autolabel_worker
COPY edge_agent ./edge_agent
COPY remote_worker ./remote_worker
COPY stats_service ./stats_service

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir \
      ultralytics>=8.4.0 \
      opencv-python>=4.9.0 \
      onnx>=1.16.0 \
      onnxruntime>=1.18.0 \
      onnxconverter-common>=1.14.0 \
      fastapi>=0.115.0 \
      uvicorn>=0.30.0 \
      PySide6>=6.8.0 \
      lmdb>=1.7.5 \
      decord>=0.6.0 \
      peft \
      Pillow>=11.1.0 \
      transformers>=4.57.1

CMD ["python", "-m", "control_plane.api", "--config", "control_plane/config/config.example.toml"]
