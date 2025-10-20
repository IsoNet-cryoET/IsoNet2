const { spawn } = require('child_process')
import { ipcMain, ipcRenderer } from 'electron'
import toCommand from '../renderer/src/utils/handle_json'
const fs = require('fs')

let inQueueList = [] // List for queued processes
let notInQueueList = [] // List for non-queued processes
let currentInQueueProcess = null // Track the current inQueue process being executed
ipcMain.handle('get-jobs-list', () => {
    console.log('main jobs', inQueueList)
    console.log('main jobs', notInQueueList)

    const sanitizedInQueueList = inQueueList.map(({ event, ...rest }) => rest)
    const sanitizedNotInQueueList = notInQueueList.map(({ event, ...rest }) => rest)

    return {
        inQueueList: sanitizedInQueueList,
        notInQueueList: sanitizedNotInQueueList
    }
})
// Function to process the inQueue list
export function handleProcess(event, data) {
    const cmd = toCommand(data, data.id)
    if (data.type !== 'prepare_star' && data.type !== 'star2json' ){
        data.output_dir = data.type+"/job"+data.id+"_"+data.name
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
        const nextProcess = notInQueueList[notInQueueList.length-1] //.shift()
        runProcess(nextProcess, (id) => {
            nextProcess == null
            console.log("run process finished id:", id)
            notInQueueList = notInQueueList.filter(item => item.id !== id);
        })
    }
}

// Function to spawn and handle a Python process
function runProcess(processItem, callback) {
    console.log(`Running command: ${processItem.command_line}`)

    // Spawn the Python process
    const pythonProcess = spawn(processItem.command_line, {
        shell: true,                // <<< let the shell parse the whole string
        detached: true,
        stdio: ['ignore', 'pipe', 'pipe'],
      });

    processItem.event.sender.send('python-status-change', { id: processItem.id, status: 'running', pid: pythonProcess.pid })
    console.log('current')
    let logFileName
    if (processItem.type === 'prepare_star'){
        logFileName = 'prepare_log.txt'
    }
    else if (processItem.type === 'star2json'){
        logFileName = '.star2json_log.txt'
    }
    else{
        fs.mkdirSync(processItem.output_dir, { recursive: true })
        logFileName = processItem.output_dir + '/log.txt'
    }

    const logStream = fs.createWriteStream(logFileName, { flags: 'a' })
    // console.log(processItem.output_dir + '/log.txt')
    logStream.write(`Command: ${processItem.command_line}\n`)
    // logStream.write(`[STDOUT] ${output}`) // Write stdout to log.txt
    pythonProcess.stdout.on('data', (data) => {
        const output = data.toString()
        // console.log(output)
        // processItem.event.sender.send('python-stdout', {
        //     cmd: processItem.cmd,
        //     output: output
        // })
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

        if (code === 0 && (processItem.type === 'prepare_star' || processItem.type === 'star2json')) {
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

        processItem.event.sender.send('python-status-change', { id: processItem.id, status: 'completed', pid: processItem.pid })
        logStream.end()
        if (callback) callback(processItem.id) // Notify that the process has finished
    })
}

ipcMain.on('remove-job', (event, id) => {
    const jobIndex = inQueueList.findIndex(
        (item) => item.id === id && item.status === 'inqueue'
    )

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
