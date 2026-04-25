FROM 10.200.99.202:15080/zero2x002/competition-base:pytorch2.5.1-cuda12.1-cudnn9

WORKDIR /workspace
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

ARG PIP_CACHE_DIR
COPY requirements.txt .
# PyTorch 2.5.1 with CUDA 12.1 is pre-installed in the base image.
# Only install lightweight project dependencies (rasterio, pandas, etc.).
RUN pip install --cache-dir /tmp/pip-cache -r requirements.txt -i https://repo.huaweicloud.com/repository/pypi/simple

COPY . .
RUN chmod +x /workspace/run.sh

CMD ["./run.sh"]
