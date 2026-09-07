# Base image and layout follow the platform's own Build Image Guide, shown under
# Image Environment Configuration on the model details page:
#   base image : base/base:base   (Satellite SDK, Python 3.9, scheduling runtime)
#   workdir    : /app
#   build      : docker build -t app/c03_fulldatatrained:v1.0.0 .
#   push       : docker push app/c03_fulldatatrained:v1.0.0
# In the spaceborne dev environment, build_delta.sh re-reads base.image and
# rewrites the FROM line if the device declares a different base.
FROM base/base:base

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# numpy + onnxruntime only. No torch (an aarch64 CUDA wheel alone would exceed
# the 300 MB delta cap) and no rasterio/GDAL. The model ships as ONNX.
# Both resolve to Python 3.9-compatible wheels, matching the base runtime.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY kybelix_orbit.py /app/kybelix_orbit.py
COPY model /app/model
COPY sample /app/sample

# Fail the build, not the satellite pass, if the runtime is not importable.
RUN python3 -c "import numpy, onnxruntime; print('numpy', numpy.__version__, 'ort', onnxruntime.__version__)" \
 && python3 -c "import onnxruntime as o; o.InferenceSession('/app/model/c03.onnx', providers=['CPUExecutionProvider']); print('onnx graph loads')"

# The platform starts the application from the YAML cmd; this is the local default.
CMD ["python3", "/app/kybelix_orbit.py", "--output-dir", "/output"]
