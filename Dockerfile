# Enhanced Dockerfile for Crossing Challenge.
# Uses Ensemble intent (LightGBM + XGBoost) + XGBoost trajectory models.
# Target: <= 2 GB image size.
#
# Build:
#   docker build -t my-crossing .
# Test:
#   docker run --rm --network=none -v $(pwd)/data:/work my-crossing /work/dev.parquet /work/preds.csv

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY predict.py grade.py features.py ./
COPY model_intent_ensemble.pkl model_traj_xgb.pkl model.pkl ./

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "from predict import predict; print('ok')" || exit 1

ENTRYPOINT ["python", "grade.py"]
