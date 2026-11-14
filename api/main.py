import hashlib
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

LOG = logging.getLogger(__name__)


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    sepal_length: float = Field(gt=0, le=30)
    sepal_width: float = Field(gt=0, le=30)
    petal_length: float = Field(gt=0, le=30)
    petal_width: float = Field(gt=0, le=30)


def create_app(bundle_path: Path | None = None) -> FastAPI:
    bundle = bundle_path or Path(os.environ.get("MODEL_BUNDLE", "/app/model"))

    @asynccontextmanager
    async def lifespan(app):
        app.state.model, app.state.metadata = None, None
        try:
            metadata = json.loads((bundle / "metadata.json").read_text())
            digest = hashlib.sha256((bundle / "model.joblib").read_bytes()).hexdigest()
            if not metadata.get("quality_passed") or digest != metadata["model_sha256"]:
                raise ValueError("Unapproved or corrupted model bundle")
            # Only load the locally built, hash-verified bundle; never accept uploaded pickle files.
            app.state.model = joblib.load(bundle / "model.joblib")
            app.state.metadata = metadata
        except Exception as exc:
            LOG.error("Model unavailable: %s", type(exc).__name__)
        yield

    app = FastAPI(title="Iris Prediction API", version="1.0.0", lifespan=lifespan)

    @app.get("/health/live")
    def live():
        return {"status": "alive"}

    @app.get("/health/ready")
    def ready():
        if app.state.model is None or os.environ.get("FAIL_READINESS", "false").lower() == "true":
            raise HTTPException(503, "Model not ready")
        return {"status": "ready", "model_run_id": app.state.metadata["run_id"]}

    @app.get("/version")
    def version():
        ready()
        return {
            "app_version": "1.0.0",
            "model_run_id": app.state.metadata["run_id"],
            "dataset": app.state.metadata["provenance"],
        }

    @app.post("/predict")
    def predict(observation: Observation):
        ready()
        frame = pd.DataFrame([observation.model_dump()])[app.state.metadata["features"]]
        predicted = int(app.state.model.predict(frame)[0])
        return {
            "class_id": predicted,
            "class_name": app.state.metadata["classes"][predicted],
            "model_run_id": app.state.metadata["run_id"],
        }

    return app


app = create_app()
