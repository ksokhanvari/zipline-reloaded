"""
FlightLog Commands for JupyterLab

This creates custom commands accessible via the Command Palette (Cmd+Shift+P).
Add to your IPython startup to auto-load.

Usage in notebook:
    # Load extension
    %load_ext flightlog_commands

    # Or run directly:
    from flightlog_commands import start_log_server, start_print_server, start_both

Then use Command Palette (Cmd+Shift+P) and search for "FlightLog".
"""

import subprocess
import os
import socket

def check_port(port):
    """Check if a port is in use."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    return result == 0

def start_log_server():
    """Start FlightLog LOG server on port 9020 in background."""
    if check_port(9020):
        print("LOG server already running on port 9020")
        return

    proc = subprocess.Popen(
        ['python', '/app/scripts/flightlog.py', '--port', '9020'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True
    )
    print(f"Started LOG server on port 9020 (PID: {proc.pid})")
    return proc

def start_print_server():
    """Start FlightLog PRINT server on port 9021 in background."""
    if check_port(9021):
        print("PRINT server already running on port 9021")
        return

    proc = subprocess.Popen(
        ['python', '/app/scripts/flightlog.py', '--port', '9021'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True
    )
    print(f"Started PRINT server on port 9021 (PID: {proc.pid})")
    return proc

def start_both():
    """Start both FlightLog servers."""
    start_log_server()
    start_print_server()

def status():
    """Check FlightLog server status."""
    print("FlightLog Server Status")
    print("=" * 40)

    if check_port(9020):
        print("LOG server (9020):   RUNNING")
    else:
        print("LOG server (9020):   NOT RUNNING")

    if check_port(9021):
        print("PRINT server (9021): RUNNING")
    else:
        print("PRINT server (9021): NOT RUNNING")

def stop_servers():
    """Stop all FlightLog servers."""
    import signal
    result = subprocess.run(
        ['pkill', '-f', 'flightlog.py'],
        capture_output=True
    )
    if result.returncode == 0:
        print("Stopped all FlightLog servers")
    else:
        print("No FlightLog servers running")

# IPython magic support
def load_ipython_extension(ipython):
    """Load as IPython extension."""
    from IPython.core.magic import register_line_magic

    @register_line_magic
    def flightlog(line):
        """FlightLog commands: log, print, both, status, stop"""
        cmd = line.strip().lower()
        if cmd == 'log':
            start_log_server()
        elif cmd == 'print':
            start_print_server()
        elif cmd == 'both':
            start_both()
        elif cmd == 'status':
            status()
        elif cmd == 'stop':
            stop_servers()
        else:
            print("Usage: %flightlog [log|print|both|status|stop]")

    print("FlightLog commands loaded. Use: %flightlog [log|print|both|status|stop]")
