# JupyterLab Dark Theme Configuration

This directory contains JupyterLab user settings that enable dark theme by default.

## Settings Applied

### 1. Main Theme (`@jupyterlab/apputils-extension/themes.jupyterlab-settings`)
- **Theme**: JupyterLab Dark
- **Scrollbars**: Dark themed scrollbars enabled

### 2. Terminal Theme (`@jupyterlab/terminal-extension/plugin.jupyterlab-settings`)
- **Theme**: Dark
- **Font**: Monospace, 13px

### 3. Code Editor Theme (`@jupyterlab/codemirror-extension/commands.jupyterlab-settings`)
- **Theme**: Material Darker (syntax highlighting optimized for dark backgrounds)

## How It Works

When JupyterLab starts, it reads these configuration files from:
```
.jupyter/lab/user-settings/
```

The settings are automatically applied to all new JupyterLab sessions.

## Changing Themes

You can change the theme through:
1. **Settings menu**: Settings → Theme → JupyterLab Dark (or any other theme)
2. **Edit config files**: Modify the `.jupyterlab-settings` files in this directory

## Available Themes

JupyterLab includes these built-in themes:
- JupyterLab Light (default light theme)
- JupyterLab Dark (default dark theme)

Additional themes can be installed as JupyterLab extensions.

## Persistence in Docker

The `.jupyter/` directory is part of the repository, so these settings persist across Docker container restarts and are version-controlled.
