import { contextBridge, ipcRenderer } from 'electron'
import { electronAPI } from '@electron-toolkit/preload'
ipcRenderer.setMaxListeners(100)
// Custom APIs for renderer
const api = {
    selectFile(property) {
        const folderPath = ipcRenderer.invoke('select-file', property)
        return folderPath
    },
    // loadStar(star_name) {
    //     ipcRenderer.send('load_star', star_name)
    // },
    updateStar: (data_json) => {
        ipcRenderer.send('update_star', data_json)
    },
    run: (data) => {
        ipcRenderer.send('run', data)
    },
    view: (file) => {
        ipcRenderer.send('view', file)
    },
    // Listen for Python stderr messages
    onPythonRunning: (callback) => {
        ipcRenderer.on('python-running', (event, data) => {
            callback(data)
        })
    },
    onPythonClosed: (callback) => {
        ipcRenderer.on('python-closed', (event, data) => {
            callback(data)
        })
    },
    // Listen for Python stderr messages
    onPythonStderr: (callback) => {
        ipcRenderer.on('python-stderr', (event, data) => {
            callback(data)
        })
    },
    onPythonStdout: (callback) => {
        ipcRenderer.on('python-stdout', (event, data) => {
            callback(data)
        })
    },
    killJob: (pid) => {
        return new Promise((resolve, reject) => {
            ipcRenderer.once('kill-job-response', (event, success) => {
                if (success) {
                    resolve() // Resolve the promise if the process was killed
                } else {
                    reject(new Error('Failed to kill job')) // Reject if there was an error
                }
            })

            ipcRenderer.send('kill-job', pid) // Send the kill request
        })
    },
    removeJobFromQueue: (result) => {
        return new Promise((resolve, reject) => {
            ipcRenderer.once('remove-job-response', (event, success) => {
                if (success) {
                    resolve() // Resolve the promise when the job is removed
                } else {
                    reject(new Error('Failed to remove job from queue')) // Reject the promise if there's an error
                }
            })

            ipcRenderer.send('remove-job', result) // Send the job removal request
        })
    },

    onJson: (callback) => {
        ipcRenderer.on('json-star', (event, data) => {
            console.log(data)
            callback(data)
        })
    },
    onFetchJobs: async () => {
        try {
            const jobs = await ipcRenderer.invoke('get-jobs-list')
            console.log('preload jobs', jobs)

            return jobs
        } catch (error) {
            console.error('Error fetching jobs:', error)
            return null // Handle error gracefully
        }
    }

    // offJson: (callback) => {
    //     ipcRenderer.removeListener('json-star', callback)
    // }
}

if (process.contextIsolated) {
    try {
        contextBridge.exposeInMainWorld('electron', electronAPI)
        contextBridge.exposeInMainWorld('api', api)
    } catch (error) {
        console.error(error)
    }
} else {
    // window.electron = electronAPI
    window.api = api
}
