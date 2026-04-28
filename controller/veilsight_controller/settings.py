from __future__ import annotations

import os
from pathlib import Path


class ControllerSettings:
    controller_id: str = "controller"

    def __init__(self) -> None:
        self.config_path = Path(os.getenv("VEILSIGHT_CONFIG", "configs/dual_example.yaml")).resolve()
        self.runner_grpc = os.getenv("VEILSIGHT_RUNNER_GRPC", "unix:///tmp/veilsight-runner.sock")
        self.runner_public_base_url = os.getenv("VEILSIGHT_RUNNER_PUBLIC_BASE_URL", "http://localhost:8080")
        self.web_dist = Path(os.getenv("VEILSIGHT_WEB_DIST", "web/dist")).resolve()


settings = ControllerSettings()
