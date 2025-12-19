#!/usr/bin/env python3
"""
FlightLog Launcher for JupyterLab

This script starts FlightLog servers on ports 9020 and 9021.
Run this from a JupyterLab terminal or use the launcher buttons.

Usage:
    python start_flightlog.py [log|print|both]

    log   - Start log server on port 9020
    print - Start print server on port 9021
    both  - Start both servers (default)
"""

import subprocess
import sys
import os

FLIGHTLOG_SCRIPT = '/app/scripts/flightlog.py'

def start_log_server():
    """Start FlightLog server for log messages on port 9020."""
    print("=" * 60)
    print("Starting FlightLog LOG Server on port 9020")
    print("=" * 60)
    os.execvp('python3', ['python3', FLIGHTLOG_SCRIPT, '--port', '9020'])

def start_print_server():
    """Start FlightLog server for print statements on port 9021."""
    print("=" * 60)
    print("Starting FlightLog PRINT Server on port 9021")
    print("=" * 60)
    os.execvp('python3', ['python3', FLIGHTLOG_SCRIPT, '--port', '9021'])

def start_both():
    """Start both FlightLog servers in background."""
    print("=" * 60)
    print("Starting FlightLog Servers")
    print("=" * 60)
    print()
    print("Starting LOG server on port 9020...")
    log_proc = subprocess.Popen(
        ['python3', FLIGHTLOG_SCRIPT, '--port', '9020'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    print(f"  PID: {log_proc.pid}")

    print("Starting PRINT server on port 9021...")
    print_proc = subprocess.Popen(
        ['python3', FLIGHTLOG_SCRIPT, '--port', '9021'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    print(f"  PID: {print_proc.pid}")

    print()
    print("Both servers started in background.")
    print("Use 'ps aux | grep flightlog' to check status.")
    print("Use 'kill <PID>' to stop servers.")

def main():
    if len(sys.argv) < 2:
        mode = 'both'
    else:
        mode = sys.argv[1].lower()

    if mode == 'log':
        start_log_server()
    elif mode == 'print':
        start_print_server()
    elif mode == 'both':
        start_both()
    else:
        print(__doc__)
        sys.exit(1)

if __name__ == '__main__':
    main()
