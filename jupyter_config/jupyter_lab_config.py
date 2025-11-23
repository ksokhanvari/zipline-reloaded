# JupyterLab Configuration
# This file configures JupyterLab settings including custom launchers

c = get_config()

# Terminal settings
c.ServerApp.terminado_settings = {
    'shell_command': ['/bin/bash']
}

# Allow embedding in iframes (if needed)
c.ServerApp.tornado_settings = {
    'headers': {
        'Content-Security-Policy': "frame-ancestors 'self'"
    }
}
