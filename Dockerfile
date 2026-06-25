# syntax=docker/dockerfile:1

ARG UBUNTU_VERSION=24.04

FROM ubuntu:${UBUNTU_VERSION} AS ncnn-builder
ARG NCNN_REF=20260526
ARG NCNN_VULKAN=OFF
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    git \
    libprotobuf-dev \
    protobuf-compiler \
    && if [ "${NCNN_VULKAN}" = "ON" ] || [ "${NCNN_VULKAN}" = "1" ] || [ "${NCNN_VULKAN}" = "true" ]; then \
        apt-get install -y --no-install-recommends \
            glslang-tools \
            libvulkan-dev \
            spirv-tools; \
    fi \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
RUN git clone --depth 1 --branch "${NCNN_REF}" https://github.com/Tencent/ncnn.git \
    && if [ "${NCNN_VULKAN}" = "ON" ] || [ "${NCNN_VULKAN}" = "1" ] || [ "${NCNN_VULKAN}" = "true" ]; then \
        git -C ncnn submodule update --init --recursive; \
    fi
RUN cmake -S ncnn -B ncnn/build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/opt/ncnn \
    -DNCNN_VULKAN="${NCNN_VULKAN}" \
    -DNCNN_OPENMP=ON \
    -DNCNN_BUILD_TOOLS=OFF \
    -DNCNN_BUILD_EXAMPLES=OFF \
    -DNCNN_BUILD_BENCHMARK=OFF \
    -DNCNN_BUILD_TESTS=OFF \
    && cmake --build ncnn/build -j2 \
    && cmake --install ncnn/build

FROM ubuntu:${UBUNTU_VERSION} AS web-builder
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/veilsight
COPY web/package*.json ./web/
RUN npm --prefix web ci
COPY web ./web
RUN npm --prefix web run build

FROM ubuntu:${UBUNTU_VERSION} AS veilsight-dev-eval
ARG NCNN_VULKAN=OFF
ENV DEBIAN_FRONTEND=noninteractive
ENV CMAKE_PREFIX_PATH=/opt/ncnn
ENV PYTHONUNBUFFERED=1
ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    libgstreamer1.0-0 \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-0 \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-0 \
    libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-libav \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-tools \
    libgrpc++-dev \
    libopencv-dev \
    libpcre2-dev \
    libprotobuf-dev \
    libsqlite3-dev \
    libyaml-cpp-dev \
    nodejs \
    npm \
    pkg-config \
    protobuf-compiler \
    protobuf-compiler-grpc \
    python3 \
    python3-dev \
    python3-pip \
    python-is-python3 \
    sqlite3 \
    && if [ "${NCNN_VULKAN}" = "ON" ] || [ "${NCNN_VULKAN}" = "1" ] || [ "${NCNN_VULKAN}" = "true" ]; then \
        apt-get install -y --no-install-recommends \
            libvulkan1 \
            mesa-vulkan-drivers \
            vulkan-tools; \
    fi \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ncnn-builder /opt/ncnn /opt/ncnn

WORKDIR /opt/veilsight
COPY controller/requirements.txt ./controller/requirements.txt

RUN python3 -m pip install --no-cache-dir --root-user-action=ignore \
    -r controller/requirements.txt \
    pytest \
    numpy \
    scipy \
    scikit-image \
    opencv-python-headless \
    pycocotools \
    tabulate \
    tqdm \
    pillow \
    matplotlib

COPY . .
COPY --from=web-builder /opt/veilsight/web/dist ./web/dist
COPY --from=web-builder /opt/veilsight/web/node_modules ./web/node_modules

RUN bash scripts/generate_proto_python.sh \
    && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DVEILSIGHT_BUILD_TESTS=ON \
    && cmake --build build -j2 \
    && mkdir -p /opt/veilsight/assets /opt/veilsight/results /opt/veilsight/data /tmp

RUN chmod +x /opt/veilsight/docker/entrypoint.sh /opt/veilsight/docker/test.sh

EXPOSE 8000 8080
ENTRYPOINT ["/opt/veilsight/docker/entrypoint.sh"]
CMD ["serve"]
