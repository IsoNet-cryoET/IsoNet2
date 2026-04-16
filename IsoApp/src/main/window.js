import { BrowserWindow, shell } from 'electron'
import { fileURLToPath } from 'url'
import { CHANNELS } from './constants.js'
// IMPORT THE ICON HERE
// The '?asset' suffix tells Vite to handle the path resolution automatically
import icon from '../../resources/icon.png?asset'

export function createMainWindow() {
    const preloadPath = fileURLToPath(new URL('../preload/preload.mjs', import.meta.url))

    const win = new BrowserWindow({
        width: 1500,
        height: 900,
        show: false,
        autoHideMenuBar: true,
        // USE THE IMPORTED VARIABLE HERE
        icon: icon,
        title: 'IsoNet2',
        titleBarStyle: 'default',
        titleBarOverlay: {
            color: '#eaf1fb',
            symbolColor: '#74b1be',
            height: 38
        },
        webPreferences: {
            preload: preloadPath,
            sandbox: false,
            devTools: true,
            contextIsolation: true
        }
    })

    win.on('ready-to-show', () => win.show())
    win.on('close', (event) => {
        if (win.__forceClose) return
        event.preventDefault()
        // double check closing activity
        win.webContents.send(CHANNELS.APP_CLOSE_REQ)
    })
    win.webContents.setWindowOpenHandler((details) => {
        shell.openExternal(details.url)
        return { action: 'deny' }
    })

    return win
}
