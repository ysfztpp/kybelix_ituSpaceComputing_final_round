# Base image and layout follow the platform's own Build Image Guide, shown under
# Image Environment Configuration on the model details page:
#   base image : base/base:base   (Satellite SDK, Python 3.9, scheduling runtime)
#   workdir    : /app
# In the spaceborne dev environment, build_delta.sh re-reads base.image and
# rewrites the FROM line if the device declares a different base.
FROM base/base:base

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Dependencies go into an explicit vendor directory, NOT site-packages.
#
# image_tool.sh drives kaniko with --single-snapshot: one filesystem snapshot at
# the end of the build. A plain `pip install` is therefore invisible to the
# delta package whenever the requirement is already satisfied in the build
# environment - which is exactly what happens on a dev pod where the package was
# installed by hand while debugging. The result builds and reports success, then
# fails on the satellite with ModuleNotFoundError.
#
# --target writes into a path that cannot pre-exist, so the snapshot always
# captures it, and PYTHONPATH makes it importable at runtime.
ENV PYTHONPATH=/app/vendor
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir --no-deps --target /app/vendor -r /app/requirements.txt

COPY kybelix_orbit.py /app/kybelix_orbit.py
COPY model /app/model
COPY sample /app/sample

# Guards. The first two fail the build if the vendor directory did not
# materialise; the third proves the graph actually loads on this architecture.
# onnxruntime must be vendored; numpy must NOT be, since the base image already
# provides it at /usr/local/lib/python3.10/dist-packages (verified by probing
# base/base:base directly). Assert on the filesystem, which is unambiguous,
# rather than on substrings of a resolved module path.
RUN test -d /app/vendor/onnxruntime && test ! -d /app/vendor/numpy
RUN python3 -c "import onnxruntime, numpy; assert '/app/vendor' in onnxruntime.__file__, onnxruntime.__file__; print('vendored ort', onnxruntime.__version__, 'at', onnxruntime.__file__); print('numpy', numpy.__version__, 'at', numpy.__file__)"
RUN python3 -c "import onnxruntime as o; s=o.SessionOptions(); s.intra_op_num_threads=2; s.inter_op_num_threads=1; o.InferenceSession('/app/model/c03.onnx', s, providers=['CPUExecutionProvider']); print('onnx session OK')"

# The platform starts the application from the YAML cmd; this is the local default.
CMD ["python3", "/app/kybelix_orbit.py", "--output-dir", "/output"]
