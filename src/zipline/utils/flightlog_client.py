"""
FlightLog Client - Send Zipline logs to FlightLog server.

Provides simple integration for sending logs from Zipline backtests
to a FlightLog server running in a separate terminal.

Usage:
    from zipline.utils.flightlog_client import enable_flightlog

    # Enable FlightLog at the start of your script
    enable_flightlog()

    # Now all Zipline logs will be sent to FlightLog server
    # (as well as regular console output)
"""

import logging
import logging.handlers
import socket
from typing import Optional


def enable_flightlog(
    host: str = 'flightlog',
    port: int = 9020,
    level: int = logging.INFO,
) -> bool:
    """
    Enable FlightLog for current session.

    Adds a SocketHandler to the root logger that sends all log records
    to a FlightLog server running on the specified host/port.

    Parameters
    ----------
    host : str, optional
        FlightLog server hostname or IP.
        Default is 'flightlog' (Docker service name).
        Use 'localhost' if running FlightLog in same container.
    port : int, optional
        FlightLog server port. Default is 9020.
    level : int, optional
        Minimum log level to send. Default is logging.INFO.

    Returns
    -------
    bool
        True if successfully connected, False otherwise.

    Examples
    --------
    >>> from zipline.utils.flightlog_client import enable_flightlog
    >>> enable_flightlog()  # Use Docker service name
    True

    >>> enable_flightlog(host='localhost')  # Same container
    True

    >>> enable_flightlog(host='localhost', level=logging.DEBUG)  # All logs
    True
    """
    try:
        # Test if server is reachable
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()

        if result != 0:
            logging.warning(
                f"FlightLog server not reachable at {host}:{port} - "
                "logs will only appear locally"
            )
            return False

        # Create socket handler
        socket_handler = logging.handlers.SocketHandler(host, port)
        socket_handler.setLevel(level)

        # Add to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(socket_handler)

        # Make sure root logger level allows these messages
        if root_logger.level > level:
            root_logger.setLevel(level)

        # Log success
        logging.info(f"FlightLog enabled - logging to {host}:{port}")

        return True

    except Exception as e:
        logging.warning(f"Could not enable FlightLog: {e}")
        return False


def disable_flightlog():
    """
    Disable FlightLog for current session.

    Removes all SocketHandler instances from the root logger.
    """
    root_logger = logging.getLogger()
    socket_handlers = [
        h for h in root_logger.handlers
        if isinstance(h, logging.handlers.SocketHandler)
    ]

    for handler in socket_handlers:
        root_logger.removeHandler(handler)
        handler.close()

    if socket_handlers:
        logging.info("FlightLog disabled")


def log_to_flightlog(message: str, level: str = 'INFO'):
    """
    Send a custom log message to FlightLog.

    This is a convenience function for sending custom messages from
    your algorithm. Messages are sent via the 'algorithm' logger.

    Parameters
    ----------
    message : str
        The message to log
    level : str, optional
        Log level: 'DEBUG', 'INFO', 'WARNING', 'ERROR', or 'CRITICAL'.
        Default is 'INFO'.

    Examples
    --------
    >>> from zipline.utils.flightlog_client import log_to_flightlog
    >>>
    >>> def initialize(context):
    ...     log_to_flightlog("Strategy initialized", level='INFO')
    ...
    >>> def handle_data(context, data):
    ...     if some_condition:
    ...         log_to_flightlog("Alert: High volatility!", level='WARNING')
    """
    # Use algorithm logger (same as Zipline uses internally)
    logger = logging.getLogger('algorithm')

    # Convert level string to logging constant
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'WARN': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL,
    }

    log_level = level_map.get(level.upper(), logging.INFO)

    # Log the message
    logger.log(log_level, message)


def get_flightlog_status() -> dict:
    """
    Get status of FlightLog connection.

    Returns
    -------
    dict
        Status dictionary with keys:
        - 'connected': bool, whether FlightLog is enabled
        - 'handlers': int, number of socket handlers attached
        - 'host': str, FlightLog host (if connected)
        - 'port': int, FlightLog port (if connected)

    Examples
    --------
    >>> from zipline.utils.flightlog_client import get_flightlog_status
    >>> status = get_flightlog_status()
    >>> print(f"Connected: {status['connected']}")
    """
    root_logger = logging.getLogger()
    socket_handlers = [
        h for h in root_logger.handlers
        if isinstance(h, logging.handlers.SocketHandler)
    ]

    status = {
        'connected': len(socket_handlers) > 0,
        'handlers': len(socket_handlers),
    }

    if socket_handlers:
        # Get info from first handler
        handler = socket_handlers[0]
        status['host'] = handler.host
        status['port'] = handler.port

    return status
