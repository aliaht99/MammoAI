# MammoAI — Hugging Face Space (Docker SDK)
# Hugging Face retired the native Streamlit SDK, so the app ships as a container.
FROM python:3.12-slim

# Spaces run the container as uid 1000.
RUN useradd -m -u 1000 user
WORKDIR /app

# Install dependencies first so layer caching survives app-code edits.
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user . .

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    MPLCONFIGDIR=/tmp/matplotlib \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# HF Spaces expect 7860 (declared as app_port in README.md); Render/Fly/Cloud Run
# inject their own $PORT. Shell form so the variable expands at runtime.
EXPOSE 7860

CMD streamlit run mammo_doctor.py \
      --server.port=${PORT:-7860} \
      --server.address=0.0.0.0 \
      --server.headless=true
