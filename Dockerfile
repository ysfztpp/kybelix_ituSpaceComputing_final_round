FROM 10.200.99.202:15080/zero2x002/competition-base:ubuntu22.04-py310.19

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

ARG PIP_CACHE_DIR
COPY requirements.txt .
RUN pip install --cache-dir /tmp/pip-cache -r requirements.txt -i https://repo.huaweicloud.com/repository/pypi/simple && \
    pip install --cache-dir /tmp/pip-cache torch==2.5.1+cpu \
      -i https://repo.huaweicloud.com/repository/pypi/simple \
      --extra-index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN chmod +x /app/run.sh

CMD ["./run.sh"]
