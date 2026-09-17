FROM nvidia/cuda:13.0.2-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn \
    nnUNet_raw=/data2/home/wcq/nnUNet/nnUNet_raw \
    nnUNet_preprocessed=/data2/home/wcq/nnUNet/nnUNet_preprocessed \
    nnUNet_results=/data2/home/wcq/nnUNet/nnUNet_results \
    API_DATA_DIR=/data

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3.10-dev python3.10-venv build-essential ninja-build ca-certificates \
      libglib2.0-0 libgl1 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /data0/home/wcq/.conda/envs/perm_test && \
    python3.10 -m venv /data0/home/wcq/.conda/envs/nodulenet_h100

COPY requirements-nnunet.txt requirements-api.txt /tmp/
RUN /data0/home/wcq/.conda/envs/perm_test/bin/pip install --upgrade pip && \
    /data0/home/wcq/.conda/envs/perm_test/bin/pip install -r /tmp/requirements-nnunet.txt -r /tmp/requirements-api.txt

COPY requirements-manet.txt /tmp/
RUN /data0/home/wcq/.conda/envs/nodulenet_h100/bin/pip install --upgrade pip && \
    /data0/home/wcq/.conda/envs/nodulenet_h100/bin/pip install -r /tmp/requirements-manet.txt

COPY build_context/engineering_demo/ /data2/home/wcq/nnUNet/nav_project/engineering_demo/
COPY build_context/manet/ /data2/home/wcq/projects/MANet-main/
COPY build_context/nnUNet_results/ /data2/home/wcq/nnUNet/nnUNet_results/
COPY backend/ /opt/airway_gpu_backend/backend/

RUN cd /data2/home/wcq/projects/MANet-main/scripts/build/box && \
    /data0/home/wcq/.conda/envs/nodulenet_h100/bin/pip install wheel && \
    /data0/home/wcq/.conda/envs/nodulenet_h100/bin/pip install --no-build-isolation . && \
    mkdir -p /data /data2/home/wcq/nnUNet/nnUNet_raw /data2/home/wcq/nnUNet/nnUNet_preprocessed

WORKDIR /opt/airway_gpu_backend
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD /data0/home/wcq/.conda/envs/perm_test/bin/python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)" || exit 1
CMD ["/data0/home/wcq/.conda/envs/perm_test/bin/uvicorn", "backend.gpu_api_server:app", "--host", "0.0.0.0", "--port", "8000"]
