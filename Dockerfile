# KG-Skeptic: Neuro-Symbolic Auditor for Bio-Agents
# Multi-stage build for minimal image size

# Stage 1: Build dependencies
FROM python:3.13-slim AS builder

WORKDIR /app

# Install UV package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock ./

# Install dependencies without dev group
RUN uv sync --frozen --no-dev --no-install-project

# Stage 2: Runtime image
FROM python:3.13-slim AS runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy UV from builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY data/ ./data/

# Ensure venv is used
ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"

# Streamlit configuration
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# KG-Skeptic defaults (can be overridden)
ENV KG_SKEPTIC_USE_MONARCH_KG=true

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash skeptic
RUN chown -R skeptic:skeptic /app
USER skeptic

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Expose Streamlit port
EXPOSE 8501

# Default: run Streamlit UI
CMD ["streamlit", "run", "src/kg_skeptic/app.py"]
