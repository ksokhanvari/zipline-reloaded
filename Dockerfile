# syntax=docker/dockerfile:1.4
FROM python:3.11-slim

LABEL maintainer="zipline-reloaded"
LABEL description="Zipline-Reloaded with Sharadar bundle support"

# Set working directory
WORKDIR /app

# Install system dependencies including Cython build requirements
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    libhdf5-dev \
    libblosc-dev \
    pkg-config \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements*.txt pyproject.toml setup.py ./

# Use BuildKit cache mount for pip (faster rebuilds)
# If BuildKit is not available, falls back to regular pip cache
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel cython

# Install bcolz-zipline separately with verbose output
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install bcolz-zipline --verbose || \
    pip install --no-build-isolation bcolz-zipline

# Install nasdaq-data-link for Sharadar bundle support
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install nasdaq-data-link

# Copy the entire project including .git for version detection
COPY . .

# Set version environment variable for setuptools-scm
ENV SETUPTOOLS_SCM_PRETEND_VERSION=3.1.1

# Install zipline-reloaded in editable mode
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e . --no-build-isolation

# Create necessary directories
RUN mkdir -p /notebooks /data /root/.zipline /scripts

# Copy extension.py to register custom bundles
COPY extension.py /root/.zipline/extension.py

# Copy all Jupyter Lab settings (dark theme, notebook, terminal, codemirror)
COPY .jupyter/lab/user-settings /root/.jupyter/lab/user-settings

# Copy Jupyter Lab overrides for default settings (50000 line scrollback, dark theme)
COPY .jupyter/lab/settings /root/.jupyter/lab/settings

# Set up Jupyter and analysis tools
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install jupyter jupyterlab notebook matplotlib pyfolio-reloaded alphalens-reloaded

# Copy overrides to Jupyter Lab system settings (ensures 50000 line scrollback, dark theme)
RUN JUPYTER_DATA_DIR=$(python -c "import jupyter_core.paths; print(jupyter_core.paths.jupyter_data_dir())") && \
    mkdir -p "$JUPYTER_DATA_DIR/lab/settings" && \
    cp /root/.jupyter/lab/settings/overrides.json "$JUPYTER_DATA_DIR/lab/settings/overrides.json"

# Expose Jupyter port
EXPOSE 8888

# Set environment variables
ENV ZIPLINE_ROOT=/root/.zipline
ENV ZIPLINE_CUSTOM_DATA_DIR=/data/custom_databases
ENV PYTHONUNBUFFERED=1

# Start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
