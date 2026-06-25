# Veilsight Docker Dev/Eval Image

Build the CPU-only dev/eval image:

```bash
docker build -t veilsight:dev-eval .
```

Build an NCNN Vulkan-enabled image:

```bash
docker build \
  --build-arg NCNN_VULKAN=ON \
  -t veilsight:dev-eval-vulkan .
```

Run the local-parity stack with mutable folders mounted:

```bash
mkdir -p assets results data
docker run --rm -it \
  -p 8000:8000 \
  -p 8080:8080 \
  -v "$PWD/assets:/opt/veilsight/assets" \
  -v "$PWD/results:/opt/veilsight/results" \
  -v "$PWD/data:/opt/veilsight/data" \
  veilsight:dev-eval
```

Open the dashboard at http://localhost:8000. The runner stream server is exposed on port 8080.

CPU remains the runtime default even in a Vulkan-enabled image. To request NCNN Vulkan execution, run with `VEILSIGHT_NCNN_VULKAN=1` and pass through the host GPU/Vulkan devices supported by your Docker runtime. For example:

```bash
docker run --rm -it \
  --device /dev/dri \
  -e VEILSIGHT_NCNN_VULKAN=1 \
  -p 8000:8000 \
  -p 8080:8080 \
  -v "$PWD/assets:/opt/veilsight/assets" \
  -v "$PWD/results:/opt/veilsight/results" \
  -v "$PWD/data:/opt/veilsight/data" \
  veilsight:dev-eval-vulkan
```

Run checks inside the image:

```bash
docker run --rm -it veilsight:dev-eval docker/test.sh all
docker run --rm -it veilsight:dev-eval docker/test.sh unit
docker run --rm -it veilsight:dev-eval docker/test.sh app
docker run --rm -it veilsight:dev-eval docker/test.sh models
docker run --rm -it veilsight:dev-eval docker/test.sh eval-smoke
```

`models` compiles a small temporary C++ smoke binary against the built core library and runs real YOLOX nano and UHD no-post detections on a synthetic frame. This catches missing NCNN, missing model files, missing `YoloV5Focus`, unwritable `/tmp` UHD param patching, and bad `out0` output shape handling.

The image bakes in `models/` and `thirdparty/TrackEval`. `assets/`, `results/`, and `data/` are intentionally mount points so datasets, outputs, and SQLite DBs can change without rebuilding. If `data/mobilefacenet_gallery.sqlite3` is missing, the entrypoint creates an empty MobileFaceNet gallery schema before starting the services.

To run a real MOT20 single-sequence smoke when the dataset is mounted:

```bash
docker run --rm -it \
  -v "$PWD/assets:/opt/veilsight/assets" \
  -v "$PWD/results:/opt/veilsight/results" \
  -e VEILSIGHT_DOCKER_REAL_EVAL_SMOKE=1 \
  veilsight:dev-eval docker/test.sh eval-smoke
```
