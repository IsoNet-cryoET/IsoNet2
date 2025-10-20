import { app, shell, BrowserWindow, ipcMain, dialog } from 'electron'
import { join } from 'path'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import { handleProcess } from './process_handler'
const { spawn } = require('child_process')
const path = require('path')
const fs = require('fs')
const Store = require('electron-store').default

function createWindow() {
    const mainWindow = new BrowserWindow({
        width: 1500,
        height: 900,
        show: false,
        autoHideMenuBar: true,
        frame: true, // Ensures no native frame, enabling the custom title bar
        titleBarStyle: 'default', // Required for custom overlays
        titleBarOverlay: {
            color: '#eaf1fb', // Title bar background color
            symbolColor: '#74b1be', // Minimize, Maximize, Close button colors
            height: 38 // Height of the title bar
        },
        webPreferences: {
            preload: join(__dirname, '../preload/preload.js'),
            sandbox: false,
            devTools: true
        },
        focusable: true, // make sure it's focusable
        alwaysOnTop: false, // avoid conflicts
    })

    mainWindow.on('ready-to-show', () => {
        mainWindow.show()
    })
    mainWindow.on('close', (event) => {
        // If we already confirmed, allow close
        if (mainWindow.__forceClose) return;
    
        // Stop the default close and ask renderer
        event.preventDefault();
        mainWindow.webContents.send('app-close-request'); // -> renderer shows confirm dialog
      });
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
    ipcMain.handle('app-close-confirmed', async (_evt, ok) => {
        if (!mainWindow) return;
        if (ok) {
          // mark and close for real (avoid infinite loop)
          mainWindow.__forceClose = true;
          mainWindow.close();
        } else {
          // do nothing; user canceled
        }
      });
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
// app.disableHardwareAcceleration()
// app.commandLine.appendSwitch('disable-gpu-compositing')
app.whenReady().then(() => {
    electronApp.setAppUserModelId('com.electron')
    app.on('browser-window-created', (_, window) => {
        optimizer.watchWindowShortcuts(window)
    })

    createWindow()
    app.on('activate', function () {
        if (BrowserWindow.getAllWindows().length === 0) createWindow()
    })

    const runDir = process.cwd()
    const store = new Store({
        name: 'settings',
        fileExtension: 'json',
        cwd: runDir,
        defaults: {
          counter: 0,
          jobList: []
        },
        schema: {
          counter: { type: 'number' },
          jobList: {
            type: 'array',
            items: { type: 'object' }
          }
        }
      })
      
      /** ---------------- 1) counter ---------------- */
      const KEY_COUNT = 'counter'
      
      ipcMain.handle('count:nextId', (_event) => {
        const next = Number(store.get(KEY_COUNT, 0)) + 1
        store.set(KEY_COUNT, next)
        return next
      })
      
      ipcMain.handle('count:getCurrentId', (_event) => {
        return Number(store.get(KEY_COUNT, 0))
      })
      
      // ipcMain.handle('count:reset', (_event, value = 0) => {
      //   const v = Number(value) || 0
      //   store.set(KEY_COUNT, v)
      //   return v
      // })
      
      /** ---------------- 2) jobList ---------------- */
      const KEY_JOB = 'jobList'
      
      ipcMain.handle('jobList:get', (_event) => {
        return store.get(KEY_JOB, [])
      })
      
      ipcMain.handle('jobList:add', (_event, data) => {
        if (!data || typeof data !== 'object') {
          return store.get(KEY_JOB, [])
        }
        const list = store.get(KEY_JOB, [])
        list.push(data)
        store.set(KEY_JOB, list)
        return list
      })
      
      ipcMain.handle('jobList:update', (_event, data) => {
        if (!data || (data.id == null)) {
          return store.get(KEY_JOB, [])
        }
        const list = store.get(KEY_JOB, [])
        const idx = list.findIndex(j => j && j.id === data.id)
        if (idx >= 0) {
          list[idx] = { ...list[idx], ...data }
          store.set(KEY_JOB, list)
        }
        return list
      })

      ipcMain.handle('jobList:updateStatus', (_event, {id, status}) => {
        console.log("status")
        console.log(status)
        const list = store.get(KEY_JOB, [])
        const idx = list.findIndex(j => j && j.id === id)
        list[idx].status = status
        store.set(KEY_JOB, list)
        return list
      })  

      ipcMain.handle('jobList:updatePID', (_event, {id, pid}) => {
        console.log("pid")
        console.log(pid)
        const list = store.get(KEY_JOB, [])
        const idx = list.findIndex(j => j && j.id === id)
        list[idx].pid = pid
        store.set(KEY_JOB, list)
        return list
      })  

      ipcMain.handle('jobList:remove', async (_event, id) => {
        const list = store.get(KEY_JOB, []);
        const nid = String(id); 
      
        const job = list.find(j => j && String(j.id) === nid);
        if (!job) return false;
      
        try {
          const rel = job.output_dir
          if (rel) {
            const base = app.isPackaged ? app.getPath('userData') : process.cwd();
            const abs = path.isAbsolute(rel) ? rel : path.join(base, rel);
      
            if (abs.length > base.length && abs.startsWith(path.dirname(base))) {
              await fs.promises.rm(abs, { recursive: true, force: true });
            }
          }
        } catch (e) {
          console.error('Failed to remove job output dir:', e);
        }
      
        const newList = list.filter(j => j && String(j.id) !== nid);
        const changed = newList.length !== list.length;
        if (changed) {
          store.set(KEY_JOB, newList);
        }
        return changed;
      });
})
app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit()
    }
})
