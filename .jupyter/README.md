# Jupyter Lab Configuration

This directory contains JupyterLab configuration files that are copied into the Docker container.

## Notebook Settings

The `tracker.jupyterlab-settings` file configures notebook rendering behavior:

- **recordTiming: false** - Disables cell execution timing recording (cleaner notebooks)
- **renderCellOnScroll: true** - Renders cells lazily as you scroll (better performance for large notebooks)
- **maxOutputSize: 0** - No limit on cell output size (useful for large dataframes and plots)
- **codeCellConfig:**
  - **lineWrap: "on"** - Wrap long lines in code cells (better readability)
  - **rulers: []** - No vertical rulers in code editor
  - **cursorBlinkRate: 600** - Cursor blink rate in milliseconds

These settings are applied to all notebooks opened in the container.

## Customization

To modify these settings:
1. Edit `.jupyter/lab/user-settings/@jupyterlab/notebook-extension/tracker.jupyterlab-settings`
2. Rebuild the Docker image: `docker compose build`
3. Restart containers: `docker compose up -d`
