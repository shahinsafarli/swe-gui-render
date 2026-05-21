# =============================================================================
# Async Research Assistant — Multi-stage Dockerfile (Render-ready)
# =============================================================================

# ---- Build stage ----
FROM python:3.13-slim AS builder
WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Runtime stage ----
FROM python:3.13-slim
LABEL org.opencontainers.image.description="AIENG Final Project — Async Research Assistant"

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN useradd --create-home --shell /bin/bash appuser
COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8501

CMD streamlit run streamlit_app.py \
    --server.port=${PORT:-8501} \
    --server.address=0.0.0.0 \
    --server.headless=true
