"""
FlightLog IPython Magic Commands

Load this in a notebook with:
    %load_ext flightlog_magic

Then use:
    %flightlog_log    - Opens terminal with LOG server (port 9020)
    %flightlog_print  - Opens terminal with PRINT server (port 9021)
    %flightlog_both   - Opens both terminals
    %flightlog_status - Check if servers are running
"""

from IPython.core.magic import Magics, magics_class, line_magic
from IPython.display import display, HTML, Javascript
import subprocess
import socket

@magics_class
class FlightLogMagics(Magics):

    @line_magic
    def flightlog_log(self, line):
        """Open a terminal running FlightLog LOG server on port 9020."""
        js_code = """
        (function() {
            const app = window.jupyterapp || window.lab;
            if (app && app.commands) {
                app.commands.execute('terminal:create-new').then(terminal => {
                    setTimeout(() => {
                        terminal.session.send({
                            type: 'stdin',
                            content: ['python /app/scripts/flightlog.py --port 9020\\n']
                        });
                    }, 500);
                });
            } else {
                alert('Please open a terminal manually and run: python /app/scripts/flightlog.py --port 9020');
            }
        })();
        """
        display(Javascript(js_code))
        print("Opening FlightLog LOG terminal on port 9020...")

    @line_magic
    def flightlog_print(self, line):
        """Open a terminal running FlightLog PRINT server on port 9021."""
        js_code = """
        (function() {
            const app = window.jupyterapp || window.lab;
            if (app && app.commands) {
                app.commands.execute('terminal:create-new').then(terminal => {
                    setTimeout(() => {
                        terminal.session.send({
                            type: 'stdin',
                            content: ['python /app/scripts/flightlog.py --port 9021\\n']
                        });
                    }, 500);
                });
            } else {
                alert('Please open a terminal manually and run: python /app/scripts/flightlog.py --port 9021');
            }
        })();
        """
        display(Javascript(js_code))
        print("Opening FlightLog PRINT terminal on port 9021...")

    @line_magic
    def flightlog_both(self, line):
        """Open both FlightLog terminals."""
        self.flightlog_log(line)
        self.flightlog_print(line)

    @line_magic
    def flightlog_status(self, line):
        """Check if FlightLog servers are running."""
        print("FlightLog Server Status")
        print("=" * 40)

        # Check port 9020
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', 9020))
            sock.close()
            if result == 0:
                print("LOG server (9020):   RUNNING")
            else:
                print("LOG server (9020):   NOT RUNNING")
        except:
            print("LOG server (9020):   NOT RUNNING")

        # Check port 9021
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', 9021))
            sock.close()
            if result == 0:
                print("PRINT server (9021): RUNNING")
            else:
                print("PRINT server (9021): NOT RUNNING")
        except:
            print("PRINT server (9021): NOT RUNNING")


def load_ipython_extension(ipython):
    """Load the extension in IPython."""
    ipython.register_magics(FlightLogMagics)


def unload_ipython_extension(ipython):
    """Unload the extension."""
    pass
