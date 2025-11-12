import { app, shell, BrowserWindow, ipcMain, dialog } from 'electron'
import path, { join } from 'path'
import fs from 'fs'
import Store from 'electron-store'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import { spawn, exec } from 'child_process'
import util from 'util'
import { handleProcess, spawnWithRuntime, inQueueList, notInQueueList } from './process.js'
import { fileURLToPath } from 'url'

let mainWindow = null

// #region settings.json
const runDir = process.cwd()
const store = new Store({
    name: 'settings',
    fileExtension: 'json',
    cwd: runDir,
    defaults: {
        counter: 0,
        condaEnv: '',
        IsoNetPath: '',
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
// #endregion

// #region const
const KEY_COUNT = 'counter'
const KEY_JOB = 'jobList'

// #endregion

async function getCurrentId() {
    return Number(store.get(KEY_COUNT, 0))
}
async function nextId() {
    const next = Number(store.get(KEY_COUNT, 0)) + 1
    store.set(KEY_COUNT, next)
    return next
}
async function getJobList() {
    return store.get(KEY_JOB, [])
}
async function addJob(_, data) {
    if (!data || typeof data !== 'object') {
        return store.get(KEY_JOB, [])
    }
    const list = store.get(KEY_JOB, [])
    list.push(data)
    store.set(KEY_JOB, list)
    return list
}
async function updateJob(_, data) {
    if (!data || data.id == null) {
        return store.get(KEY_JOB, [])
    }
    const list = store.get(KEY_JOB, [])
    const idx = list.findIndex((j) => j && j.id === data.id)
    if (idx >= 0) {
        list[idx] = { ...list[idx], ...data }
        store.set(KEY_JOB, list)
    }
    return list
}
async function updateJobStatus(_, id, status) {
    const list = store.get(KEY_JOB, [])
    const idx = list.findIndex((j) => j && j.id === id)
    list[idx].status = status
    store.set(KEY_JOB, list)
    return list
}
async function updateJobPID(_, id, pid) {
    const list = store.get(KEY_JOB, [])
    const idx = list.findIndex((j) => j && j.id === id)
    list[idx].pid = pid
    store.set(KEY_JOB, list)
    return list
}
async function updateJobName(_, id, name) {
    const list = store.get(KEY_JOB, [])
    const idx = list.findIndex((j) => j && j.id === id)
    list[idx].name = name
    store.set(KEY_JOB, list)
    return list
}
async function removeJob(_, id) {
    const list = store.get(KEY_JOB, [])
    const nid = String(id)

    const job = list.find((j) => j && String(j.id) === nid)
    if (!job) return false

    try {
        const rel = job.output_dir
        if (rel) {
            const base = app.isPackaged ? app.getPath('userData') : process.cwd()
            const abs = path.isAbsolute(rel) ? rel : path.join(base, rel)

            if (abs.length > base.length && abs.startsWith(path.dirname(base))) {
                await fs.promises.rm(abs, { recursive: true, force: true })
            }
        }
    } catch (e) {
        console.error('Failed to remove job output dir:', e)
    }

    const newList = list.filter((j) => j && String(j.id) !== nid)
    const changed = newList.length !== list.length
    if (changed) {
        store.set(KEY_JOB, newList)
    }
    return changed
}
async function setCondaEnv(_, name) {
    store.set('condaEnv', String(name || ''));
    return { success: true };
}
async function getCondaEnv() {
    return String(store.get('condaEnv', ''));
}
async function setIsoNetPath(_, absPath) {
    store.set('IsoNetPath', String(absPath || ''));
    return { success: true };
}
async function getIsoNetPath() {
    return String(store.get('IsoNetPath', ''));
}
async function getAvailableCondaEnv() {
    const execPromise = util.promisify(exec);
    try {
        const { stdout, stderr } = await execPromise('conda info --envs');
        if (stderr) console.warn('conda info --envs stderr:', stderr);

        const lines = stdout.split('\n').filter((l) => l.trim() && !l.startsWith('#'));
        const envs = lines.map((line) => {
            const parts = line.trim().split(/\s+/);
            const hasStar = parts.includes('*');
            const name = parts[0];
            const envPath = parts[parts.length - 1];
            return { name, path: envPath, active: hasStar };
        });

        return { success: true, envs };
    } catch (err) {
        console.error('Failed to get conda envs:', err);
        return { success: false, error: String(err?.message || err) };
    }
}
async function getImageData(_, relativePath) {
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
}
async function selectFile(_, property) {
    const result = await dialog.showOpenDialog({
        properties: [property]
    })
    if (!result.canceled) {
        return result.filePaths[0]
    }
    return null
}
async function readFile(_, filePath) {
    try {
        const data = await fs.promises.readFile(filePath, 'utf-8')
        return data
    } catch (error) {
        console.error('Failed to read file:', error)
        return null
    }
}
async function isFileExist(_, filePath) {
    return fs.existsSync(filePath)
}
async function appClose(_, ok) {
    if (!mainWindow) return
    if (ok) {
        // mark and close for real (avoid infinite loop)
        mainWindow.__forceClose = true
        mainWindow.close()
    }
}
async function jobsQueue() {
    const sanitizedInQueueList = inQueueList.map(({ event, ...rest }) => rest)
    const sanitizedNotInQueueList = notInQueueList.map(({ event, ...rest }) => rest)
    return {
        inQueueList: sanitizedInQueueList,
        notInQueueList: sanitizedNotInQueueList
    }
}
async function killJob(_, pid) {
    process.kill(-pid, 'SIGINT')
}
async function removeJobFromQueue(_, id) {
    const jobIndex = inQueueList.findIndex((item) => item.id === id && item.status === 'inqueue')
    inQueueList.splice(jobIndex, 1)
}
async function view(_, file) {
    let viewFile = null
    viewFile = spawn('3dmod', [file], {
        detached: true,
        stdio: ['ignore', 'pipe', 'pipe']
    })
}
async function run(event, data) {
    handleProcess(event, data)
}
async function updateStar(_, data) {
    const filePath = '.to_star.json'
    let updateStarProcess = null // To hold the reference of the running Python process

    fs.writeFile(filePath, JSON.stringify(data.convertedJson, null, 2), (err) => {
        if (err) {
            console.error('Error saving file:', err)
        } else {
            console.log('File saved successfully:', filePath)
        }
    })

    updateStarProcess = spawnWithRuntime(
        `isonet.py json2star --json_file "${filePath}" --star_name "${data.star_name}"`
    );

    let cmd = 'prepare_star'

    // Handle process close event
    updateStarProcess.on('close', (code) => {
        console.log(`Python process exited with code ${code}`)
        updateStarProcess = null // Reset the reference when the process ends
    })
}
const handlers = {
    getCurrentId,
    nextId,
    getJobList,
    addJob,
    updateJob,
    updateJobStatus,
    updateJobPID,
    updateJobName,
    removeJob,
    setCondaEnv,
    getCondaEnv,
    setIsoNetPath,
    getIsoNetPath,
    getAvailableCondaEnv,
    getImageData,
    selectFile,
    readFile,
    isFileExist,
    appClose,
    jobsQueue,
    killJob,
    removeJobFromQueue,
    view,
    run,
    updateStar,
}

ipcMain.handle('rpc', async (event, { method, params }) => {
    if (!Object.prototype.hasOwnProperty.call(handlers, method)) {
        throw new Error(`Unknown method: ${method}`)
    }
    const fn = handlers[method]
    return await fn(event, ...(params || []))
})

// #region window
function createWindow() {
    const preloadPath = fileURLToPath(new URL('../preload/preload.mjs', import.meta.url))
    mainWindow = new BrowserWindow({
        width: 1500,
        height: 900,
        show: false,
        autoHideMenuBar: true,
        frame: true, // Ensures no native frame, enabling the custom title bar
        // backgroundColor: '#ff0000',  // title bar + background color
        // titleBarStyle: 'default',
        title: 'IsoNet2',
        // titleBarStyle: 'hidden',    // hides it but keeps window controls accessible (mac)
        titleBarStyle: 'default', // Required for custom overlays
        titleBarOverlay: {
            color: '#eaf1fb', // Title bar background color
            symbolColor: '#74b1be', // Minimize, Maximize, Close button colors
            height: 38 // Height of the title bar
        },
        webPreferences: {
            preload: preloadPath,
            sandbox: false,
            devTools: true,
            contextIsolation: true
        },
        focusable: true, // make sure it's focusable
        alwaysOnTop: false // avoid conflicts
    })

    mainWindow.on('ready-to-show', () => mainWindow.show())
    mainWindow.on('close', (event) => {
        // If we already confirmed, allow close
        if (mainWindow.__forceClose) return
        // Stop the default close and ask renderer
        event.preventDefault()
        mainWindow.webContents.send('app-close-request') // -> renderer shows confirm dialog
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
}

app.whenReady().then(() => {
    electronApp.setAppUserModelId('com.electron')
    app.on('browser-window-created', (_, window) => { optimizer.watchWindowShortcuts(window) })
    createWindow()
    app.on('activate', function () {
        if (BrowserWindow.getAllWindows().length === 0) createWindow()
    })
})
app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') app.quit()
})
// #endregion