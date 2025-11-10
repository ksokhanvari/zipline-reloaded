"""
FlightLog Client - Send Zipline logs to FlightLog server.

Provides simple integration for sending logs from Zipline backtests
to a FlightLog server running in a separate terminal.

Single Channel Usage:
    from zipline.utils.flightlog_client import enable_flightlog

    # Enable FlightLog at the start of your script
    enable_flightlog()

    # Now all Zipline logs will be sent to FlightLog server
    # (as well as regular console output)

Multi-Channel Usage:
    from zipline.utils.flightlog_client import enable_multi_channel_flightlog

    # Start multiple FlightLog servers in different terminals:
    # Terminal 1: python scripts/flightlog.py --port 9020 --channel algorithm
    # Terminal 2: python scripts/flightlog.py --port 9021 --channel progress
    # Terminal 3: python scripts/flightlog.py --port 9022 --channel errors --level ERROR

    # Then in your backtest, route logs to different terminals:
    enable_multi_channel_flightlog({
        'algorithm': {'port': 9020, 'logger': 'algorithm'},
        'progress': {'port': 9021, 'logger': 'zipline.progress'},
        'errors': {'port': 9022, 'level': logging.ERROR},
    })

Preset Configurations:
    from zipline.utils.flightlog_client import FLIGHTLOG_PRESETS

    # Use preset for common scenarios
    enable_multi_channel_flightlog(FLIGHTLOG_PRESETS['split'])
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

        # Add to root logger
        root_logger = logging.getLogger()

        # Check if we already have a SocketHandler for this host:port
        existing_handlers = [
            h for h in root_logger.handlers
            if isinstance(h, logging.handlers.SocketHandler) and
               h.host == host and h.port == port
        ]

        if existing_handlers:
            # Already have a handler for this host:port, don't add another
            logging.debug(f"FlightLog handler already exists for {host}:{port}")
            return True

        # Create and add socket handler
        socket_handler = logging.handlers.SocketHandler(host, port)
        socket_handler.setLevel(level)
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


def enable_multi_channel_flightlog(
    channels: dict,
    host: str = 'localhost',
) -> dict:
    """
    Enable multiple FlightLog channels at once.

    This allows you to route different types of logs to different terminal windows.
    For example, send algorithm logs to one window and progress logs to another.

    Parameters
    ----------
    channels : dict
        Dictionary mapping channel purposes to port numbers or configurations.
        Can be:
        - Simple format: {'algorithm': 9020, 'progress': 9021}
        - Detailed format: {'algorithm': {'port': 9020, 'level': logging.INFO, 'logger': 'algorithm'}}
    host : str, optional
        FlightLog server hostname or IP. Default is 'localhost'.

    Returns
    -------
    dict
        Dictionary mapping channel names to connection status (True/False).

    Examples
    --------
    >>> from zipline.utils.flightlog_client import enable_multi_channel_flightlog
    >>>
    >>> # Simple: Route logs to different ports
    >>> enable_multi_channel_flightlog({
    ...     'all': 9020,      # All logs on port 9020
    ...     'errors': 9021,   # Error logs on port 9021
    ... })
    {'all': True, 'errors': True}

    >>> # Detailed: Specify logger filters
    >>> enable_multi_channel_flightlog({
    ...     'algorithm': {'port': 9020, 'logger': 'algorithm'},
    ...     'progress': {'port': 9021, 'logger': 'zipline.progress'},
    ... })
    {'algorithm': True, 'progress': True}

    Usage with multiple terminals:
    --------------------------------
    Terminal 1: python scripts/flightlog.py --port 9020 --channel algorithm
    Terminal 2: python scripts/flightlog.py --port 9021 --channel progress

    Then in your backtest:
        enable_multi_channel_flightlog({
            'algorithm': 9020,
            'progress': 9021,
        })
    """
    results = {}

    for channel_name, config in channels.items():
        # Handle simple format (just port number)
        if isinstance(config, int):
            port = config
            level = logging.INFO
            logger_filter = None
        # Handle detailed format
        elif isinstance(config, dict):
            port = config.get('port')
            level = config.get('level', logging.INFO)
            logger_filter = config.get('logger')
        else:
            logging.warning(f"Invalid configuration for channel '{channel_name}': {config}")
            results[channel_name] = False
            continue

        # Enable the channel
        success = _enable_channel(host, port, level, logger_filter, channel_name)
        results[channel_name] = success

    return results


def _enable_channel(
    host: str,
    port: int,
    level: int,
    logger_filter: Optional[str],
    channel_name: str,
) -> bool:
    """
    Internal function to enable a single channel with optional logger filtering.
    """
    try:
        # Test if server is reachable
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()

        if result != 0:
            logging.warning(
                f"FlightLog channel '{channel_name}' not reachable at {host}:{port}"
            )
            return False

        # If logger_filter is specified, add handler to specific logger
        # Otherwise, add to root logger (broadcasts to all)
        if logger_filter:
            target_logger = logging.getLogger(logger_filter)
        else:
            target_logger = logging.getLogger()

        # Check for existing handlers
        existing_handlers = [
            h for h in target_logger.handlers
            if isinstance(h, logging.handlers.SocketHandler) and
               h.host == host and h.port == port
        ]

        if existing_handlers:
            logging.debug(f"FlightLog channel '{channel_name}' already exists for {host}:{port}")
            return True

        # Create and add socket handler
        socket_handler = logging.handlers.SocketHandler(host, port)
        socket_handler.setLevel(level)
        target_logger.addHandler(socket_handler)

        # Make sure logger level allows these messages
        if target_logger.level > level:
            target_logger.setLevel(level)

        # Log success
        logging.info(f"FlightLog channel '{channel_name}' enabled - logging to {host}:{port}")

        return True

    except Exception as e:
        logging.warning(f"Could not enable FlightLog channel '{channel_name}': {e}")
        return False


def get_flightlog_status() -> dict:
    """
    Get status of FlightLog connection.

    Returns
    -------
    dict
        Status dictionary with keys:
        - 'connected': bool, whether FlightLog is enabled
        - 'handlers': int, number of socket handlers attached
        - 'channels': list, information about each connected channel

    Examples
    --------
    >>> from zipline.utils.flightlog_client import get_flightlog_status
    >>> status = get_flightlog_status()
    >>> print(f"Connected: {status['connected']}")
    >>> for channel in status['channels']:
    ...     print(f"  {channel['host']}:{channel['port']} at level {channel['level']}")
    """
    root_logger = logging.getLogger()

    # Get all socket handlers from all loggers
    all_socket_handlers = []

    # Check root logger
    for handler in root_logger.handlers:
        if isinstance(handler, logging.handlers.SocketHandler):
            all_socket_handlers.append({
                'host': handler.host,
                'port': handler.port,
                'level': logging.getLevelName(handler.level),
                'logger': 'root',
            })

    # Check other active loggers
    for logger_name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers:
            if isinstance(handler, logging.handlers.SocketHandler):
                all_socket_handlers.append({
                    'host': handler.host,
                    'port': handler.port,
                    'level': logging.getLevelName(handler.level),
                    'logger': logger_name,
                })

    status = {
        'connected': len(all_socket_handlers) > 0,
        'handlers': len(all_socket_handlers),
        'channels': all_socket_handlers,
    }

    return status


# Preset channel configurations for common scenarios
FLIGHTLOG_PRESETS = {
    # Split algorithm logs and progress logs into separate windows
    'split': {
        'algorithm': {'port': 9020, 'logger': 'algorithm'},
        'progress': {'port': 9021, 'logger': 'zipline.progress'},
    },

    # Three-way split: algorithm, progress, and everything else
    'three_way': {
        'algorithm': {'port': 9020, 'logger': 'algorithm'},
        'progress': {'port': 9021, 'logger': 'zipline.progress'},
        'system': {'port': 9022, 'level': logging.DEBUG},
    },

    # Separate errors from normal logs
    'errors_separate': {
        'main': {'port': 9020, 'level': logging.INFO},
        'errors': {'port': 9021, 'level': logging.ERROR},
    },

    # Debug everything in different windows by subsystem
    'debug': {
        'algorithm': {'port': 9020, 'logger': 'algorithm', 'level': logging.DEBUG},
        'data': {'port': 9021, 'logger': 'zipline.data', 'level': logging.DEBUG},
        'finance': {'port': 9022, 'logger': 'zipline.finance', 'level': logging.DEBUG},
        'other': {'port': 9023, 'level': logging.DEBUG},
    },
}
