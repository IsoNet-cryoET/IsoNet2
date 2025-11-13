import { ipcMain } from 'electron'

export function registerRpc(handlers) {
    ipcMain.handle('rpc', async (event, { method, params }) => {
        try {
            if (!Object.prototype.hasOwnProperty.call(handlers, method)) {
                throw new Error(`Unknown method: ${method}`)
            }
            const fn = handlers[method]
            return await fn(event, ...(params || []))
        } catch (err) {
            console.error(`RPC error [${method}]:`, err)
            return { ok: false, error: String(err?.message || err) }
        }
    })
}
