import { spawn } from 'child_process';
import { ipcMain } from 'electron';
import toCommand from '../renderer/src/utils/handle_json'
import fs from 'fs';
import path from 'path';
import ElectronStore from 'electron-store';
const Store = (ElectronStore && ElectronStore.default) ? ElectronStore.default : ElectronStore;


// same store file as main.js
const store = new Store({ name: 'settings', cwd: process.cwd() });


/** read + validate settings */
function readRuntimeSettings() {
  const condaEnv = String(store.get('condaEnv', '') || '');
  const isoNetPath = String(store.get('IsoNetPath', '') || '');

  const isoBin = isoNetPath ? path.join(isoNetPath, 'IsoNet', 'bin') : '';
  const isoOk = isoNetPath && fs.existsSync(isoNetPath);
  const isoBinOk = isoBin && fs.existsSync(isoBin);

  return { condaEnv, isoNetPath, isoBin, isoOk, isoBinOk };
}

/** build merged env (PATH / PYTHONPATH) cross-platform */
function buildEnvOverlay(isoNetPath, isoBin) {
  const sep = process.platform === 'win32' ? ';' : ':';
  const env = { ...process.env };

  if (isoBin && fs.existsSync(isoBin)) {
    env.PATH = env.PATH ? `${isoBin}${sep}${env.PATH}` : isoBin;
  }
  if (isoNetPath && fs.existsSync(isoNetPath)) {
    env.PYTHONPATH = env.PYTHONPATH
      ? `${isoNetPath}${sep}${env.PYTHONPATH}`
      : isoNetPath;
  }
  return env;
}

/**
 * spawn wrapper:
 * - augments env with IsoNet PATH/PYTHONPATH
 * - optionally routes through `conda run -n <env>`
 * - keeps your current { shell:true, detached:true, stdio:['ignore','pipe','pipe'] }
 */
function spawnWithRuntime(commandLine) {
  const { condaEnv, isoNetPath, isoBin } = readRuntimeSettings();
  const env = buildEnvOverlay(isoNetPath, isoBin);

  // Prefer conda run so we don't rely on activation scripts
  // If no condaEnv saved, run raw command
  const cmd = condaEnv
    ? `conda run -n "${condaEnv}" --no-capture-output ${commandLine}`
    : commandLine;

  return spawn(cmd, {
    shell: true,
    detached: true,
    stdio: ['ignore', 'pipe', 'pipe'],
    env, // ← inject our PATH/PYTHONPATH overlay
  });
}


let inQueueList = [] // List for queued processes
let notInQueueList = [] // List for non-queued processes
let currentInQueueProcess = null // Track the current inQueue process being executed
ipcMain.handle('get-jobs-list', () => {
    const sanitizedInQueueList = inQueueList.map(({ event, ...rest }) => rest)
    const sanitizedNotInQueueList = notInQueueList.map(({ event, ...rest }) => rest)

    return {
        inQueueList: sanitizedInQueueList,
        notInQueueList: sanitizedNotInQueueList
    }
})
// Function to process the inQueue list
function handleProcess(event, data) {
    const cmd = toCommand(data, data.id)
    if (data.type !== 'prepare_star' && data.type !== 'star2json') {
        data.output_dir = data.type + '/job' + data.id + '_' + data.name
    }
    if (data.status == 'inqueue') {
        inQueueList.push({
            id: data.id,
            type: data.type,
            command_line: cmd,
            output_dir: data.output_dir,
            event,
            status: 'inqueue'
        })
        processInQueue()
    } else {
        notInQueueList.push({
            id: data.id,
            type: data.type,
            command_line: cmd,
            output_dir: data.output_dir,
            event,
            status: 'running'
        })
        processNotInQueue()
    }
}

function processInQueue() {
    if (currentInQueueProcess) {
        let l = inQueueList.length

        inQueueList[l - 1].event.sender.send('python-stdout', {
            cmd: inQueueList[l - 1].cmd,
            output: 'Queued: ' + inQueueList[l - 1].result
        })
        return
    }

    const nextProcess = inQueueList[0] //.shift()
    if (nextProcess) {
        currentInQueueProcess = nextProcess
        currentInQueueProcess.status = 'running'
        // currentInQueueProcess.event.sender.send('jobList:updateStatus', {
        //     id: currentInQueueProcess.id,
        //     status: 'running'
        // })
        runProcess(nextProcess, () => {
            currentInQueueProcess = null // Clear when finished
            inQueueList.shift()
            processInQueue() // Process the next inQueue job
        })
    }
}

// Function to process the notInQueue list
function processNotInQueue() {
    let l = notInQueueList.length
    if (l > 0) {
        const nextProcess = notInQueueList[notInQueueList.length - 1] //.shift()
        runProcess(nextProcess, (id) => {
            nextProcess == null
            console.log('run process finished id:', id)
            notInQueueList = notInQueueList.filter((item) => item.id !== id)
        })
    }
}

// Function to spawn and handle a Python process
function runProcess(processItem, callback) {
    console.log(`Running command: ${processItem.command_line}`)

    const pythonProcess = spawnWithRuntime(processItem.command_line); // 👈
  
    // // Spawn the Python process
    // const pythonProcess = spawn(processItem.command_line, {
    //     shell: true, // <<< let the shell parse the whole string
    //     detached: true,
    //     stdio: ['ignore', 'pipe', 'pipe']
    // })

    processItem.event.sender.send('python-status-change', {
        id: processItem.id,
        status: 'running',
        pid: pythonProcess.pid
    })
    let logFileName
    let logStream
    if (processItem.type === 'prepare_star') {
        logFileName = 'prepare_log.txt'
        logStream = fs.createWriteStream(logFileName, { flags: 'w' })
    } else if (processItem.type === 'star2json') {
        logFileName = '.star2json_log.txt'
        logStream = fs.createWriteStream(logFileName, { flags: 'w' })
    } else {
        fs.mkdirSync(processItem.output_dir, { recursive: true })
        logFileName = processItem.output_dir + '/log.txt'
        logStream = fs.createWriteStream(logFileName, { flags: 'a' })
    }

    
    logStream.write(`Command: ${processItem.command_line}\n`)
    pythonProcess.stdout.on('data', (data) => {
        const output = data.toString()
        logStream.write(`${output}`) // Write stdout to log.txt
    })
    processItem.pid = pythonProcess.pid
    // Capture stderr
    pythonProcess.stderr.on('data', (data) => {
        const output = data.toString()
        // processItem.event.sender.send('python-stderr', {
        //     cmd: processItem.cmd,
        //     output: output
        // })
        logStream.write(`${output}`) // Write stderr to log.txt
    })

    // Handle process close
    pythonProcess.on('close', (code) => {
        console.log(`Python process for ${processItem.type} exited with code ${code}`)
        logStream.write(`Process exited with code ${code}\n`) // Log the exit code

        if (
            code === 0 &&
            (processItem.type === 'prepare_star' || processItem.type === 'star2json')
        ) {
            fs.readFile('.to_node.json', 'utf8', (err, data) => {
                if (!err) {
                    const jsonData = data.split('\n').map((line) => JSON.parse(line))
                    processItem.event.sender.send('json-star', {
                        cmd: 'prepare_star',
                        output: jsonData
                    })
                }
            })
        }
        processItem.event.sender.send('python-status-change', {
            id: processItem.id,
            status: 'completed',
            pid: processItem.pid
        })
        logStream.end()
        if (callback) callback(processItem.id) // Notify that the process has finished
    })
}

ipcMain.on('remove-job', (event, id) => {
    const jobIndex = inQueueList.findIndex((item) => item.id === id && item.status === 'inqueue')

    if (jobIndex !== -1) {
        // Job found, remove it
        inQueueList.splice(jobIndex, 1)
        event.reply('remove-job-response', true) // Send success response
    } else {
        event.reply('remove-job-response', false) // Send failure response
    }
})

ipcMain.on('kill-job', (event, pid) => {
    console.log(`Attempting to kill process group with PID: ${pid}`)
    try {
        process.kill(-pid, 'SIGINT') // Kill entire process group
        console.log('Python process group killed')
        event.reply('kill-job-response', true) // Send success response
    } catch (err) {
        console.error('Failed to kill Python process group:', err)
        event.reply('kill-job-response', false) // Send failure response
    }
})
export { handleProcess, spawnWithRuntime };