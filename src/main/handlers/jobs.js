import fs from 'fs'
import path from 'path'
import { app } from 'electron'
import { STORE_KEYS } from '../constants.js'
import { inQueueList, notInQueueList } from '../process.js'

export default function jobs({ store }) {
    const KEY_JOB = STORE_KEYS.JOBS

    return {
        async getJobList() {
            return store.get(KEY_JOB, [])
        },
        async addJob(_, data) {
            if (!data || typeof data !== 'object') {
                return store.get(KEY_JOB, [])
            }
            const list = store.get(KEY_JOB, [])
            list.push(data)
            store.set(KEY_JOB, list)
            return list
        },
        async updateJob(_, data) {
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
        },
        async updateJobStatus(_, id, status) {
            const list = store.get(KEY_JOB, [])
            const idx = list.findIndex(j => j && j.id === id)
            if (idx >= 0) list[idx].status = status
            store.set(KEY_JOB, list)
            return list
        },
        async updateJobPID(_, id, pid) {
            const list = store.get(KEY_JOB, [])
            const idx = list.findIndex(j => j && j.id === id)
            if (idx >= 0) list[idx].pid = pid
            store.set(KEY_JOB, list)
            return list
        },
        async updateJobName(_, id, name) {
            const list = store.get(KEY_JOB, [])
            const idx = list.findIndex(j => j && j.id === id)
            if (idx >= 0) list[idx].name = name
            store.set(KEY_JOB, list)
            return list
        },
        async removeJob(_, id) {
            const list = store.get(KEY_JOB, [])
            const nid = String(id)
            const job = list.find(j => j && String(j.id) === nid)
            if (!job) return false

            try {
                const rel = job.output_dir
                if (rel) {
                    const base = app.isPackaged ? app.getPath('userData') : process.cwd()
                    const abs = path.isAbsolute(rel) ? rel : path.join(base, rel)
                    if (abs.startsWith(base)) await fs.promises.rm(abs, { recursive: true, force: true })
                }
            } catch (e) {
                console.error('Failed to remove job output dir:', e)
            }

            const newList = list.filter(j => j && String(j.id) !== nid)
            store.set(KEY_JOB, newList)
            return true
        },
        async jobsQueue() {
            const sanitizedIn = inQueueList.map(({ event, ...rest }) => rest)
            const sanitizedNot = notInQueueList.map(({ event, ...rest }) => rest)
            return {
                inQueueList: sanitizedIn,
                notInQueueList: sanitizedNot
            }
        },
        async killJob(_, pid) {
            process.kill(-pid, 'SIGINT')
        },
        async removeJobFromQueue(_, id) {
            const idx = inQueueList.findIndex(i => i.id === id && i.status === 'inqueue')
            if (idx >= 0) inQueueList.splice(idx, 1)
        },
    }
}
