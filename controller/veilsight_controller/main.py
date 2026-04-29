from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api import RunnerClientRegistry, router
from .settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    analytics = RunnerClientRegistry.analytics(settings)
    await analytics.start()
    client = RunnerClientRegistry.instance(settings)
    await client.start()
    try:
        yield
    finally:
        await client.close()
        await analytics.close()


app = FastAPI(title="Veilsight Controller", lifespan=lifespan)
app.include_router(router)


if settings.web_dist.exists():
    assets_dir = settings.web_dist / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str) -> FileResponse:
        requested = settings.web_dist / full_path
        if requested.is_file():
            return FileResponse(requested)
        return FileResponse(settings.web_dist / "index.html")
