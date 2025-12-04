import util from 'util'
import { exec } from 'child_process'
import { STORE_KEYS } from '../constants.js'

const execPromise = util.promisify(exec)

export default function environment({ environmentStore }) {
    return {
        async setCondaEnv(_, name) {
            environmentStore.set(STORE_KEYS.CONDA_ENV, String(name || ''))
            return { success: true }
        },
        async getCondaEnv() {
            return String(environmentStore.get(STORE_KEYS.CONDA_ENV, ''))
        },
        async setIsoNetPath(_, absPath) {
            environmentStore.set(STORE_KEYS.ISONET_PATH, String(absPath || ''))
            return { success: true }
        },
        async getIsoNetPath() {
            return String(environmentStore.get(STORE_KEYS.ISONET_PATH, ''))
        },
        async getAvailableCondaEnv() {
            try {
                const { stdout, stderr } = await execPromise('conda info --envs')
                if (stderr) console.warn('conda info --envs stderr:', stderr)
                const lines = stdout.split('\n').filter(l => l.trim() && !l.startsWith('#'))
                const envs = lines.map(line => {
                    const parts = line.trim().split(/\s+/)
                    const hasStar = parts.includes('*')
                    const name = parts[0]
                    const envPath = parts.at(-1)
                    return { name, path: envPath, active: hasStar }
                })
                return { success: true, envs }
            } catch (err) {
                console.error('Failed to get conda envs:', err)
                return { success: false, error: String(err?.message || err) }
            }
        },
    }
}
