import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';

import { IMainMenu } from '@jupyterlab/mainmenu';
import { Menu } from '@lumino/widgets';

/**
 * FlightLog Menu Extension
 */
const plugin: JupyterFrontEndPlugin<void> = {
  id: 'flightlog-menu:plugin',
  autoStart: true,
  requires: [IMainMenu],
  activate: (app: JupyterFrontEnd, mainMenu: IMainMenu) => {
    console.log('FlightLog menu extension activated');

    const { commands } = app;

    // Command to start LOG server (port 9020)
    const commandLogServer = 'flightlog:start-log';
    commands.addCommand(commandLogServer, {
      label: 'Start LOG Server (9020)',
      caption: 'Open terminal with FlightLog LOG server on port 9020',
      execute: () => {
        commands.execute('terminal:create-new').then((terminal: any) => {
          if (terminal && terminal.session) {
            setTimeout(() => {
              terminal.session.send({
                type: 'stdin',
                content: ['python /app/scripts/flightlog.py --port 9020\n']
              });
            }, 500);
          }
        });
      }
    });

    // Command to start PRINT server (port 9021)
    const commandPrintServer = 'flightlog:start-print';
    commands.addCommand(commandPrintServer, {
      label: 'Start PRINT Server (9021)',
      caption: 'Open terminal with FlightLog PRINT server on port 9021',
      execute: () => {
        commands.execute('terminal:create-new').then((terminal: any) => {
          if (terminal && terminal.session) {
            setTimeout(() => {
              terminal.session.send({
                type: 'stdin',
                content: ['python /app/scripts/flightlog.py --port 9021\n']
              });
            }, 500);
          }
        });
      }
    });

    // Command to start both servers
    const commandBothServers = 'flightlog:start-both';
    commands.addCommand(commandBothServers, {
      label: 'Start Both Servers',
      caption: 'Open terminals for both LOG and PRINT servers',
      execute: async () => {
        await commands.execute(commandLogServer);
        await commands.execute(commandPrintServer);
      }
    });

    // Create the FlightLog menu
    const flightlogMenu = new Menu({ commands });
    flightlogMenu.title.label = 'FlightLog';

    // Add commands to menu
    flightlogMenu.addItem({ command: commandLogServer });
    flightlogMenu.addItem({ command: commandPrintServer });
    flightlogMenu.addItem({ type: 'separator' });
    flightlogMenu.addItem({ command: commandBothServers });

    // Add menu to main menu bar
    mainMenu.addMenu(flightlogMenu, { rank: 100 });
  }
};

export default plugin;
