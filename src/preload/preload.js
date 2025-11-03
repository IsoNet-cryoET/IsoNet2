import { contextBridge, ipcRenderer } from 'electron'
import { electronAPI } from '@electron-toolkit/preload'
ipcRenderer.setMaxListeners(100)

contextBridge.exposeInMainWorld('count', {
    next: () => ipcRenderer.invoke('count:nextId'),
    current: () => ipcRenderer.invoke('count:getCurrentId')
    // reset: (value) => ipcRenderer.invoke('count:resetCounter', value),
})
contextBridge.exposeInMainWorld('environment', {
  setCondaEnv: (name) =>
    ipcRenderer.invoke('environment:setCondaEnv', name),
  getAvailableCondaEnv: () =>
    ipcRenderer.invoke('environment:getAvailableCondaEnv'),
  getCondaEnv: () =>
    ipcRenderer.invoke('environment:getCondaEnv'),
  setIsoNetPath: (absPath) =>
    ipcRenderer.invoke('environment:setIsoNetPath', absPath),
  getIsoNetPath: () =>
    ipcRenderer.invoke('environment:getIsoNetPath'),
});
contextBridge.exposeInMainWorld('jobList', {
    get: () => ipcRenderer.invoke('jobList:get'),
    add: (data) => ipcRenderer.invoke('jobList:add', data),
    update: (data) => ipcRenderer.invoke('jobList:update', data),
    updateStatus: ({ id, status }) => ipcRenderer.invoke('jobList:updateStatus', { id, status }),
    updatePID: ({ id, pid }) => ipcRenderer.invoke('jobList:updatePID', { id, pid }),
    updateName: ({ id, name }) => ipcRenderer.invoke('jobList:updateName', { id, name }),
    remove: (id) => ipcRenderer.invoke('jobList:remove', id)
})

contextBridge.exposeInMainWorld('appClose', {
    onRequest: (cb) => {
        const listener = () => cb?.()
        ipcRenderer.on('app-close-request', listener)
        return () => ipcRenderer.removeListener('app-close-request', listener)
    },
    reply: (ok) => ipcRenderer.invoke('app-close-confirmed', ok)
})
// Custom APIs for renderer
const api = {
    getImageData: (relativePath) => ipcRenderer.invoke('get-image-data', relativePath),
    selectFile: (property) => ipcRenderer.invoke('select-file', property),
    readFile: (filePath) => ipcRenderer.invoke('read-file', filePath),
    exists: (filePath) => ipcRenderer.invoke('check-exists', filePath),

    updateStar: (data_json) => ipcRenderer.send('update_star', data_json),
    run: (data) => ipcRenderer.send('run', data),
    view: (file) => ipcRenderer.send('view', file),
    // onPythonRunning: (cb) => { ipcRenderer.on('python-running', (_e, d) => cb(d)) },

    onPythonUpdateStatus: (cb) => {
        const channel = 'python-status-change'
        const handler = (_e, payload) => cb(payload)
        ipcRenderer.on(channel, handler)
        return () => ipcRenderer.removeListener(channel, handler)
    },

    // onPythonClosed: (cb) => {
    //     const channel = 'python-closed';
    //     const handler = (_e, payload) => cb(payload)
    //     ipcRenderer.on(channel, handler)
    //     return () => ipcRenderer.removeListener(channel, handler)
    //   },

    onPythonStderr: (cb) => {
        ipcRenderer.on('python-stderr', (_e, d) => cb(d))
    },
    onPythonStdout: (cb) => {
        ipcRenderer.on('python-stdout', (_e, d) => cb(d))
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
    removeJobFromQueue: (id) => {
        return new Promise((resolve, reject) => {
            ipcRenderer.once('remove-job-response', (event, success) => {
                if (success) {
                    resolve() // Resolve the promise when the job is removed
                } else {
                    reject(new Error('Failed to remove job from queue')) // Reject the promise if there's an error
                }
            })

            ipcRenderer.send('remove-job', id) // Send the job removal request
        })
    },

    onJson: (callback) => {
        ipcRenderer.on('json-star', (event, data) => {
            callback(data)
        })
    },
    onFetchJobs: async () => {
        try {
            const jobs = await ipcRenderer.invoke('get-jobs-list')

            return jobs
        } catch (error) {
            console.error('Error fetching jobs:', error)
            return null // Handle error gracefully
        }
    }
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
