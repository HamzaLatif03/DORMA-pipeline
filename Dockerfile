FROM --platform=linux/amd64 ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ---------------------------------------------------------------------------
# System packages + build tools for the C++ SDK
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ninja-build git \
    gpg curl ca-certificates lsb-release wget \
    libcurl4-openssl-dev libssl-dev pkg-config \
    libv4l-dev libgles2-mesa-dev libunwind-dev \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    ffmpeg libsm6 libxext6 libgl1 \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

# CMake 3.27+ required by SmartSpectra SDK
RUN wget -q https://github.com/Kitware/CMake/releases/download/v3.27.0/cmake-3.27.0-linux-x86_64.sh \
    && chmod +x cmake-3.27.0-linux-x86_64.sh \
    && ./cmake-3.27.0-linux-x86_64.sh --skip-license --prefix=/usr/local \
    && rm cmake-3.27.0-linux-x86_64.sh

# ---------------------------------------------------------------------------
# Presage SmartSpectra C++ SDK from their PPA
# ---------------------------------------------------------------------------
RUN curl -s "https://presage-security.github.io/PPA/KEY.gpg" \
        | gpg --dearmor \
        | tee /etc/apt/trusted.gpg.d/presage-technologies.gpg >/dev/null \
    && curl -s --compressed \
        -o /etc/apt/sources.list.d/presage-technologies.list \
        "https://presage-security.github.io/PPA/presage-technologies.list" \
    && apt-get update \
    && apt-get install -y --no-install-recommends libsmartspectra-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Build the presage_processor C++ binary
# ---------------------------------------------------------------------------
COPY presage_processor/ /build/presage_processor/
WORKDIR /build/presage_processor
RUN mkdir build && cd build \
    && cmake -G Ninja -DCMAKE_BUILD_TYPE=Release .. \
    && ninja \
    && cp presage_processor /usr/local/bin/presage_processor \
    && cd / && rm -rf /build

# ---------------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------------
WORKDIR /app
COPY requirements.txt /app/
# dlib needs single-threaded build under QEMU emulation
RUN CMAKE_BUILD_PARALLEL_LEVEL=1 pip3 install --no-cache-dir dlib
RUN pip3 install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Application code
# ---------------------------------------------------------------------------
COPY . /app/

EXPOSE 8000

CMD ["uvicorn", "run:app", "--host", "0.0.0.0", "--port", "8000"]
