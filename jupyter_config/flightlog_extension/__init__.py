"""
FlightLog JupyterLab Server Extension

Adds FlightLog menu to JupyterLab for launching log terminals.
"""

def _jupyter_server_extension_points():
    return [{"module": "flightlog_extension"}]


def _load_jupyter_server_extension(server_app):
    """Load the server extension."""
    server_app.log.info("FlightLog extension loaded")
