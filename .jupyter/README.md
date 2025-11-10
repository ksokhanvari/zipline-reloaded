# Jupyter Lab Configuration

This directory contains JupyterLab configuration files that are copied into the Docker container.

## Notebook Settings

The `tracker.jupyterlab-settings` file configures notebook behavior and appearance.

### Kernel & Initialization
- **enableKernelInitNotification: false** - No notifications when kernel starts
- **autoStartDefaultKernel: false** - Manual kernel start (don't auto-start)
- **kernelShutdown: false** - Kernel stays running when closing notebook
- **accessKernelHistory: false** - Don't access kernel command history
- **defaultCell: "code"** - New cells default to code type

### Code Cell Configuration
- **lineNumbers: false** - No line numbers in code cells
- **lineWrap: false** - No automatic line wrapping (horizontal scroll for long lines)

### Markdown & Raw Cells
- **markdownCellConfig.lineNumbers: false** - No line numbers in markdown cells
- **markdownCellConfig.matchBrackets: false** - No bracket matching in markdown
- **autoRenderMarkdownCells: false** - Manual markdown rendering (must run cell to render)
- **showEditorForReadOnlyMarkdown: true** - Show editor for read-only markdown cells
- **rawCellConfig.lineNumbers: false** - No line numbers in raw cells
- **rawCellConfig.matchBrackets: false** - No bracket matching in raw cells

### Output & Performance
- **maxNumberOutputs: 50** - Limit to 50 cell outputs per notebook
- **windowingMode: "none"** - Render all cells (no virtualization, better for smaller notebooks)
- **overscanCount: 1** - Minimal overscanning for performance
- **recordTiming: false** - Don't record cell execution timing

### UI & Navigation
- **showInputPlaceholder: true** - Show input placeholders in cells
- **scrollPastEnd: true** - Allow scrolling past the last cell
- **scrollHeadingToTop: true** - Scroll markdown headings to top of viewport
- **documentWideUndoRedo: false** - Undo/redo operates per cell, not document-wide
- **showHiddenCellsButton: true** - Show button to reveal hidden cells

### Kernel Status Display
- **kernelStatus.showOnStatusBar: false** - Don't show kernel status on status bar
- **kernelStatus.showProgress: true** - Show progress indicator during execution

### Side-by-Side Rendering
- **renderingLayout: "default"** - Use default rendering layout
- **sideBySideLeftMarginOverride: "10px"** - Left margin for side-by-side view
- **sideBySideRightMarginOverride: "10px"** - Right margin for side-by-side view
- **sideBySideOutputRatio: 1** - Equal split between input and output in side-by-side

### History & Scope
- **inputHistoryScope: "global"** - Share input history globally across notebooks

## Customization

To modify these settings:

1. Edit `.jupyter/lab/user-settings/@jupyterlab/notebook-extension/tracker.jupyterlab-settings`
2. Commit your changes to the repository
3. Rebuild the Docker image: `docker compose build`
4. Restart containers: `docker compose up -d`

The settings will be automatically applied to all notebooks opened in the container.

## Philosophy

These settings are optimized for:
- **Clean interface** - No line numbers, minimal UI clutter
- **Manual control** - Explicit kernel start, manual markdown rendering
- **Performance** - Limited outputs, efficient rendering
- **Flexibility** - Scroll past end, show hidden cells
- **Usability** - Global history, input placeholders
