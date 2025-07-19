import { app, shell, BrowserWindow, ipcMain, dialog } from 'electron'
import { join } from 'path'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import { handleProcess } from './process_handler'
const { spawn } = require('child_process')
const path = require('path')
const fs = require('fs')

function createWindow() {
    const mainWindow = new BrowserWindow({
        width: 1500,
        height: 900,
        show: false,
        autoHideMenuBar: true,
        frame: false, // Ensures no native frame, enabling the custom title bar
        titleBarStyle: 'hidden', // Required for custom overlays
        titleBarOverlay: {
            color: '#eaf1fb', // Title bar background color
            symbolColor: '#74b1be', // Minimize, Maximize, Close button colors
            height: 38 // Height of the title bar
        },
        webPreferences: {
            preload: join(__dirname, '../preload/preload.js'),
            sandbox: false,
            devTools: true
        }
    })

    mainWindow.on('ready-to-show', () => {
        mainWindow.show()
    })

    mainWindow.webContents.setWindowOpenHandler((details) => {
        shell.openExternal(details.url)
        return { action: 'deny' }
    })

    if (is.dev && process.env['ELECTRON_RENDERER_URL']) {
        mainWindow.loadURL(process.env['ELECTRON_RENDERER_URL'])
    } else {
        mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
    }
    // mainWindow.webContents.openDevTools()
}

ipcMain.handle('select-file', async (_, property) => {
    const result = await dialog.showOpenDialog({
        properties: [property]
    })
    if (!result.canceled) {
        return result.filePaths[0]
    }
    return null
})

ipcMain.handle('read-file', async (_, filePath) => {
    try {
        const data = await fs.promises.readFile(filePath, 'utf-8')
        return data
    } catch (error) {
        console.error('Failed to read file:', error)
        return null
    }
})

ipcMain.handle('get-image-data', async (event, relativePath) => {
    const fullPath = path.resolve(process.cwd(), relativePath)
    try {
        const data = fs.readFileSync(fullPath)
        return {
            success: true,
            content: `data:image/png;base64,${data.toString('base64')}`
        }
    } catch (err) {
        return { success: false, error: err.message }
    }
})

ipcMain.on('update_star', (event, data) => {
    const filePath = '.to_star.json'
    let updateStarProcess = null // To hold the reference of the running Python process

    fs.writeFile(filePath, JSON.stringify(data.convertedJson, null, 2), (err) => {
        if (err) {
            console.error('Error saving file:', err)
            event.reply('save-json-response', { success: false, error: err.message })
        } else {
            console.log('File saved successfully:', filePath)
            event.reply('save-json-response', { success: true, filePath })
        }
    })

    updateStarProcess = spawn(
        'isonet.py',
        ['json2star', '--json_file', filePath, '--star_name', data.star_name],
        {
            detached: true, // Create a new process group
            stdio: ['ignore', 'pipe', 'pipe'] // Ignore stdin, keep stdout/stderr
        }
    ) // Corrected the split
    let cmd = 'prepare_star'
    // Capture and print stdout in real time
    updateStarProcess.stdout.on('data', (data) => {
        event.sender.send('python-stdout', { cmd: cmd, output: data.toString() })
    })

    updateStarProcess.stderr.on('data', (data) => {
        event.sender.send('python-stderr', { cmd: cmd, output: data.toString() })
    })

    // Handle process close event
    updateStarProcess.on('close', (code) => {
        console.log(`Python process exited with code ${code}`)
        updateStarProcess = null // Reset the reference when the process ends
    })
})

ipcMain.on('run', (event, data) => {
    handleProcess(event, data)
})

ipcMain.on('view', (event, file) => {
    let viewFile = null
    viewFile = spawn('3dmod', [file], {
        detached: true,
        stdio: ['ignore', 'pipe', 'pipe']
    })
    viewFile.stdout.on('data', (data) => {
        event.sender.send('python-stdout', { cmd: 'prepare_star', output: data.toString() })
    })

    viewFile.stderr.on('data', (data) => {
        event.sender.send('python-stderr', { cmd: 'prepare_star', output: data.toString() })
    })
})
// window adaptation
app.whenReady().then(() => {
    electronApp.setAppUserModelId('com.electron')
    app.on('browser-window-created', (_, window) => {
        optimizer.watchWindowShortcuts(window)
    })
    createWindow()
    app.on('activate', function () {
        if (BrowserWindow.getAllWindows().length === 0) createWindow()
    })
})
app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit()
    }
})
