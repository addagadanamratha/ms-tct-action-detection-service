FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch (the Space has no GPU)
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Official MS-TCT architecture code (Dai et al., CVPR 2022)
RUN git clone --depth 1 https://github.com/dairui01/MS-TCT.git /opt/MS-TCT

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1 HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app

# Download the CLIP weights at build time, so the first upload is not slowed down
RUN python -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='openai')"

COPY --chown=user . /home/user/app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
