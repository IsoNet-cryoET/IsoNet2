import { contextBridge, ipcRenderer } from 'electron'

// Expose a unified API to the renderer process
contextBridge.exposeInMainWorld('api', {
    /**
     * Call a handler in the main process (two-way request/response)
     * Corresponds to: ipcMain.handle(channel, handler)
     * Returns a Promise that resolves with the result
     */
    call: (method, ...params) => ipcRenderer.invoke('rpc', { method, params }),

    /**
     * Listen for events sent from the main process (can trigger multiple times)
     * Corresponds to: webContents.send(channel, data)
     * Returns an unsubscribe function to remove the listener
     */
    on: (channel, listener) => {
        const wrapped = (_event, ...args) => listener(...args)
        ipcRenderer.on(channel, wrapped)
        return () => ipcRenderer.removeListener(channel, wrapped)
    },

    /**
     * Listen for a one-time event from the main process
     * The listener will be automatically removed after the first trigger
     * Corresponds to: webContents.send(channel, data)
     */
    once: (channel, listener) => {
        const wrapped = (_event, ...args) => listener(...args)
        ipcRenderer.once(channel, wrapped)
    },

    /**
     * Send a one-way message to the main process (fire and forget)
     * Corresponds to: ipcMain.on(channel, handler)
     * Does not return a Promise or expect a response
     */
    send: (channel, ...args) => {
        ipcRenderer.send(channel, ...args)
    }
})