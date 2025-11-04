#!/usr/bin/env python
"""
FlightLog - Real-time log viewer for Zipline backtests.

A standalone log server that displays backtest logs in a separate terminal,
similar to QuantRocket's flightlog.

Usage:
    # Start the log server
    python scripts/flightlog.py

    # Or with Docker
    docker compose run --rm flightlog

    # Custom options
    python scripts/flightlog.py --port 9021 --file logs/backtest.log
"""

import argparse
import logging
import logging.handlers
import pickle
import socketserver
import struct
import sys
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Formatter that adds color codes for different log levels."""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'

    def __init__(self, use_colors=True):
        super().__init__(
            fmt='%(asctime)s %(levelname)-8s %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.use_colors = use_colors

    def format(self, record):
        if self.use_colors:
            levelname = record.levelname
            if levelname in self.COLORS:
                # Color the level name
                record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"

                # Make algorithm logs bold
                if record.name == 'algorithm':
                    record.msg = f"{self.BOLD}{record.msg}{self.RESET}"

        return super().format(record)


class ProgressFilter(logging.Filter):
    """Filter to exclude progress logs."""

    def filter(self, record):
        # Exclude zipline.progress logs
        return record.name != 'zipline.progress'


class LogRecordStreamHandler(socketserver.StreamRequestHandler):
    """Handler for streaming log records."""

    def handle(self):
        """
        Handle multiple log records from a single connection.
        Each record is pickled and prefixed with a 4-byte length.
        """
        while True:
            try:
                # Read the 4-byte length prefix
                chunk = self.connection.recv(4)
                if len(chunk) < 4:
                    break

                # Unpack the length
                slen = struct.unpack('>L', chunk)[0]

                # Read the log record data
                chunk = self.connection.recv(slen)
                while len(chunk) < slen:
                    chunk += self.connection.recv(slen - len(chunk))

                # Unpickle and log the record
                record = logging.makeLogRecord(pickle.loads(chunk))
                self.server.logger.handle(record)

            except Exception as e:
                # Connection closed or error
                break


class LogRecordSocketReceiver(socketserver.ThreadingTCPServer):
    """
    Simple TCP socket-based logging receiver.
    """

    allow_reuse_address = True

    def __init__(self, host='0.0.0.0', port=9020, handler=LogRecordStreamHandler):
        socketserver.ThreadingTCPServer.__init__(self, (host, port), handler)
        self.timeout = 1
        self.logger = logging.getLogger()


def main():
    """Main entry point for FlightLog server."""
    parser = argparse.ArgumentParser(
        description='FlightLog - Real-time log viewer for Zipline backtests'
    )
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Host to bind to (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=9020,
        help='Port to listen on (default: 9020)'
    )
    parser.add_argument(
        '--level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Minimum log level to display (default: INFO)'
    )
    parser.add_argument(
        '--file',
        help='Save logs to file (optional)'
    )
    parser.add_argument(
        '--no-color',
        action='store_true',
        help='Disable color output'
    )
    parser.add_argument(
        '--filter-progress',
        action='store_true',
        help='Filter out progress logs (only show algorithm logs)'
    )

    args = parser.parse_args()

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, args.level))

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter(use_colors=not args.no_color))

    # Add progress filter if requested
    if args.filter_progress:
        console_handler.addFilter(ProgressFilter())

    root_logger.addHandler(console_handler)

    # Optional file handler
    if args.file:
        file_handler = logging.FileHandler(args.file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)-8s %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        root_logger.addHandler(file_handler)

    # Print banner
    print("=" * 70)
    print("FlightLog Server - Real-time Zipline Backtest Logging")
    print("=" * 70)
    print(f"Listening on: {args.host}:{args.port}")
    print(f"Log level: {args.level}")
    if args.file:
        print(f"Saving to: {args.file}")
    if args.filter_progress:
        print("Filter: Progress logs hidden (--filter-progress)")
    print()
    print("Waiting for backtest connections...")
    print("Press Ctrl+C to stop")
    print("=" * 70)
    print()

    # Start server
    server = LogRecordSocketReceiver(args.host, args.port)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n")
        print("=" * 70)
        print("FlightLog Server Stopped")
        print("=" * 70)
        server.shutdown()
        sys.exit(0)


if __name__ == '__main__':
    main()
