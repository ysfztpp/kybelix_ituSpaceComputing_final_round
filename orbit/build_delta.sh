#!/usr/bin/env bash
# Build the image delta package inside the spaceborne development environment.
#
#   cd /home/spaceapp/project/kybelix/orbit && ./build_delta.sh
#
# Wraps the platform's /usr/local/bin/image_tool.sh with the checks that catch
# the three failures the manual lists most often: a FROM that does not match the
# device's base image, a delta package over the 300 MB cap, and a runtime that
# cannot actually import its dependencies.
set -euo pipefail

APP_NAME="${APP_NAME:-app/c03_fulldatatrained}"
APP_VERSION="${APP_VERSION:-v1.0.0}"
DELTA_NAME="${DELTA_NAME:-c03_fulldatatrained-${APP_VERSION}.tar}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/spaceapp/project/data}"
BASE_IMAGE_FILE="${BASE_IMAGE_FILE:-/home/spaceapp/project/base.image}"
IMAGE_TOOL="${IMAGE_TOOL:-/usr/local/bin/image_tool.sh}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_BYTES=$((300 * 1000 * 1000))

echo "[build] app=${APP_NAME} version=${APP_VERSION} delta=${DELTA_NAME}"

# 1. The Dockerfile FROM must equal the device's declared base image.
if [ -f "${BASE_IMAGE_FILE}" ]; then
  BASE_IMAGE="$(tr -d '[:space:]' < "${BASE_IMAGE_FILE}" | sed 's/^FROM//')"
  echo "[build] base.image declares: ${BASE_IMAGE}"
  DOCKERFILE_FROM="$(awk '/^FROM /{print $2; exit}' "${HERE}/Dockerfile")"
  echo "[build] Dockerfile FROM:     ${DOCKERFILE_FROM}"
  if [ "${BASE_IMAGE}" != "${DOCKERFILE_FROM}" ]; then
    echo "[build] rewriting Dockerfile FROM to match base.image"
    sed -i "s|^FROM .*|FROM ${BASE_IMAGE}|" "${HERE}/Dockerfile"
  fi
else
  echo "[build] WARNING ${BASE_IMAGE_FILE} not found; leaving Dockerfile FROM unchanged." >&2
  echo "[build] WARNING an image built on the wrong base will fail the container-start stage." >&2
fi

# 1b. Remove stale build state from any previous run.
#
# image_tool.sh drives kaniko unprivileged, so the Dockerfile's COPY and RUN
# steps write into this pod's REAL filesystem at /app rather than an isolated
# rootfs. Anything a previous build left there is still present for the next
# one: pip reports "Target directory /app/vendor/... already exists" and skips
# it, the guards see packages nobody asked for, and --single-snapshot can sweep
# the leftovers into the delta. A stale /app is how a 22 MB dependency set
# turns back into a 142 MB one.
STALE_DIR="${STALE_DIR:-/app}"
if [ -d "${STALE_DIR}" ]; then
  echo "[build] removing stale build state: ${STALE_DIR} ($(du -sh "${STALE_DIR}" 2>/dev/null | cut -f1))"
  rm -rf "${STALE_DIR}"
fi

# 2. Required payload must be present before we spend build time.
for required in kybelix_orbit.py requirements.txt model/c03.onnx model/band_stats.json sample/demo_input.npz; do
  if [ ! -f "${HERE}/${required}" ]; then
    echo "[build] ERROR missing ${required}" >&2
    exit 1
  fi
done
echo "[build] payload size: $(du -sh "${HERE}/model" "${HERE}/sample" | tr '\n' ' ')"

# 3. Build the delta package.
mkdir -p "${OUTPUT_DIR}"
echo "[build] running ${IMAGE_TOOL}"
"${IMAGE_TOOL}" "${HERE}/Dockerfile" "${APP_NAME}" "${APP_VERSION}" "${DELTA_NAME}" "${OUTPUT_DIR}"

# 4. Enforce the platform's 300 MB delta cap before upload, not after.
DELTA_PATH="${OUTPUT_DIR}/${DELTA_NAME}"
if [ ! -f "${DELTA_PATH}" ]; then
  echo "[build] ERROR expected delta package at ${DELTA_PATH}" >&2
  exit 1
fi
SIZE_BYTES="$(wc -c < "${DELTA_PATH}" | tr -d ' ')"
echo "[build] delta package: ${DELTA_PATH} ($((SIZE_BYTES / 1000 / 1000)) MB)"
if [ "${SIZE_BYTES}" -gt "${MAX_BYTES}" ]; then
  echo "[build] ERROR delta package exceeds the 300 MB platform limit" >&2
  exit 1
fi

# 5. Prove the delta actually contains the runtime, not just the app files.
#
# image_tool.sh drives kaniko with --single-snapshot, so anything pip considers
# "already satisfied" in the build environment silently produces no layer
# content. That builds green and then dies on the satellite with
# ModuleNotFoundError, which costs a verification round to discover. Check here
# instead.
echo "[build] verifying delta contents"
LAYER="$(tar -tf "${DELTA_PATH}" | grep '\.tar$' | head -1)"
if [ -z "${LAYER}" ]; then
  echo "[build] ERROR no layer tar inside ${DELTA_PATH}" >&2
  exit 1
fi
# Extract the layer to a file first. Piping into `tar -t` without an explicit
# `-f -` does not reliably read stdin, which silently yields an empty listing and
# reports every path as missing.
LAYER_TMP="$(mktemp -t layer.XXXXXX.tar)"
trap 'rm -f "${LAYER_TMP}"' EXIT
tar -xOf "${DELTA_PATH}" "${LAYER}" > "${LAYER_TMP}"
CONTENTS="$(tar -tf "${LAYER_TMP}" 2>/dev/null | sed 's#^/##')"
missing=0
for required in "app/kybelix_orbit.py" "app/model/c03.onnx" "app/sample/demo_input.npz" "app/vendor/onnxruntime"; do
  if printf '%s\n' "${CONTENTS}" | grep -q "^${required}"; then
    echo "[build]   present: ${required}"
  else
    echo "[build]   MISSING: ${required}" >&2
    missing=1
  fi
done
if [ "${missing}" -ne 0 ]; then
  echo "[build] ERROR delta package is incomplete; do not upload it." >&2
  echo "[build] The runtime is installed with pip --target /app/vendor precisely so" >&2
  echo "[build] the kaniko snapshot captures it. Check the RUN pip step in the build log." >&2
  exit 1
fi

echo "[build] OK. Download ${DELTA_NAME} from the environment's Output File List,"
echo "[build] then upload it with app.yaml at Start Model Verification."
