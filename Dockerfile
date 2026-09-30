FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv python3-pip git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /workspace/kk-ru

COPY requirements.txt pyproject.toml README.md ./
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install --index-url https://download.pytorch.org/whl/cu128 torch \
    && python3 -m pip install -r requirements.txt

COPY . .

RUN python3 -m pip install -e . --no-deps

CMD ["python3", "-m", "kk_ru.train", "--config", "configs/p0.yaml"]
