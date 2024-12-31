const { spawn } = require('child_process')
import { app, shell, BrowserWindow, ipcMain, dialog } from 'electron'
const fs = require('fs')

function toCommand(data) {
    let result = ''
    let cmd = ''
    let output_dir = '.'
    for (const key in data) {
        if (data.hasOwnProperty(key)) {
            let value = data[key]
            if (value === true) {
                value = 'True'
            } else if (value === false) {
                value = 'False'
            }

            if (key === 'command') {
                // Prepend the command value
                result = `${value}${result}`
                cmd = `${value}`
            } else if (
                key === 'even_odd_input' ||
                key === 'split_top_bottom_halves' ||
                key === 'only_print' ||
                key === 'inqueue'
            ) {
                // Do nothing for 'even_odd_input'
            } else {
                // Append key-value pair in the format "--key value"
                result += ` --${key} ${value}`
                if (key === 'output_dir') {
                    output_dir = value
                }
            }
        }
    }
    return { cmd, result, output_dir }
}

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
    const { cmd, result, output_dir } = toCommand(data)
    if (data.hasOwnProperty('only_print') && data['only_print'] === true) {
        event.sender.send('python-stdout', { cmd: cmd, output: result })
    } else {
        if (data.inqueue) {
            inQueueList.push({ cmd, result, output_dir, event, status: 'queued' })
            processInQueue()
        } else {
            notInQueueList.push({ cmd, result, output_dir, event, status: 'running' })
            processNotInQueue()
        }
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
        const nextProcess = notInQueueList[0] //.shift()
        runProcess(nextProcess, () => {
            nextProcess == null
            notInQueueList.shift()
        })
    }
}

// Function to spawn and handle a Python process
function runProcess(processItem, callback) {
    console.log(`Running command: isonet.py ${processItem.result}`)

    // Spawn the Python process
    const pythonProcess = spawn('isonet.py', [...processItem.result.split(' ')], {
        detached: true,
        stdio: ['ignore', 'pipe', 'pipe']
    })

    processItem.event.sender.send('python-running', { cmd: processItem.cmd, output: 'running' })
    const logStream = fs.createWriteStream(processItem.output_dir + '/log.txt', { flags: 'a' })
    logStream.write(`Command: isonet.py ${processItem.result}\n`)
    // logStream.write(`[STDOUT] ${output}`) // Write stdout to log.txt
    pythonProcess.stdout.on('data', (data) => {
        const output = data.toString()
        processItem.event.sender.send('python-stdout', {
            cmd: processItem.cmd,
            output: output
        })
        logStream.write(`${output}`) // Write stdout to log.txt
    })
    processItem.pid = pythonProcess.pid
    // Capture stderr
    pythonProcess.stderr.on('data', (data) => {
        const output = data.toString()
        processItem.event.sender.send('python-stderr', {
            cmd: processItem.cmd,
            output: output
        })
        logStream.write(`${output}`) // Write stderr to log.txt
    })

    // Handle process close
    pythonProcess.on('close', (code) => {
        console.log(`Python process for ${processItem.cmd} exited with code ${code}`)
        logStream.write(`Process exited with code ${code}\n`) // Log the exit code

        if (code === 0 && (processItem.cmd === 'prepare_star' || processItem.cmd === 'star2json')) {
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

        processItem.event.sender.send('python-closed', { cmd: processItem.cmd, output: 'closed' })
        logStream.end()
        if (callback) callback() // Notify that the process has finished
    })
}

ipcMain.on('remove-job', (event, result) => {
    const jobIndex = inQueueList.findIndex(
        (item) => item.result === result && item.status === 'queued'
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
// ipcMain.on('run', (event, data) => {
//     const { cmd, result } = toCommand(data)
//     if (data.hasOwnProperty('only_print') && data['only_print'] === true) {
//         event.sender.send('python-stdout', { cmd: cmd, output: result })
//     } else {
//         console.log(`isonet.py ${result}`)

//         // Spawn the Python process
//         pythonProcess = spawn('isonet.py', [...result.split(' ')], {
//             detached: true, // Create a new process group
//             stdio: ['ignore', 'pipe', 'pipe'] // Ignore stdin, keep stdout/stderr
//         }) // Corrected the split

//         event.sender.send('python-running', { cmd: cmd, output: 'running' })

//         // Capture and print stdout in real time
//         pythonProcess.stdout.on('data', (data) => {
//             event.sender.send('python-stdout', { cmd: cmd, output: data.toString() })
//         })

//         pythonProcess.stderr.on('data', (data) => {
//             event.sender.send('python-stderr', { cmd: cmd, output: data.toString() })
//         })

//         // Handle process close event
//         pythonProcess.on('close', (code) => {
//             console.log(`Python process exited with code ${code}`)
//             if (code === 0 && (cmd === 'prepare_star' || cmd === 'star2json')) {
//                 fs.readFile('.to_node.json', 'utf8', (err, data) => {
//                     if (err) {
//                         console.error('Error reading JSON:', err)
//                         return
//                     }
//                     const jsonData = data.split('\n').map((line) => JSON.parse(line)) // Parsing each line as an individual object
//                     event.sender.send('json-star', { cmd: 'prepare_star', output: jsonData })
//                     //console.log(jsonData) // Logs an array of objects
//                 })
//             }
//             pythonProcess = null // Reset the reference when the process ends
//             event.sender.send('python-closed', { cmd: cmd, output: 'closed' })
//         })
//     }
// })
