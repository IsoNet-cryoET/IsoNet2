import { app } from 'electron'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import { createStore } from './store.js'
import { registerRpc } from './rpc.js'
import { createMainWindow } from './window.js'
import counter from './handlers/counter.js'
import environment from './handlers/environment.js'
import files from './handlers/files.js'
import jobs from './handlers/jobs.js'
import other from './handlers/other.js'

let mainWindow = null
const store = createStore(process.cwd())

app.whenReady().then(async () => {
    electronApp.setAppUserModelId('com.electron')
    app.on('browser-window-created', (_, win) => optimizer.watchWindowShortcuts(win))

    mainWindow = createMainWindow()

    if (is.dev && process.env.ELECTRON_RENDERER_URL) {
        mainWindow.loadURL(process.env.ELECTRON_RENDERER_URL)
    } else {
        const { join } = await import('path')
        mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
    }

    const ctx = { store, getMainWindow: () => mainWindow }
    const handlers = {
        ...counter(ctx),
        ...environment(ctx),
        ...files(ctx),
        ...jobs(ctx),
        ...other(ctx),
    }
    registerRpc(handlers)
})

app.on('activate', () => {
    if (mainWindow) return
    mainWindow = createMainWindow()
})

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') app.quit()
})
