import fs from 'fs'
import path from 'path'
import { app } from 'electron'
import { STORE_KEYS } from '../constants.js'
import { inQueueList, notInQueueList } from '../process.js'

export default function jobs({ store }) {
    const KEY_JOB = STORE_KEYS.JOBS
    function getList() {
        const list = store.get(KEY_JOB, [])
        return Array.isArray(list) ? list : []
    }
    function setList(list) {
        store.set(KEY_JOB, list)
        return list
    }

    return {
        async getJobList() {
            return getList()
        },

        async addJob(_, data) {
            if (!data || typeof data !== 'object') {
                return getList()
            }
            const list = getList()
            const newList = [...list, data]
            return setList(newList)
        },

        async updateJob(_, data) {
            if (!data || data.id == null) {
                return getList()
            }
            const list = getList()
            const idx = list.findIndex((j) => j && j.id === data.id)
            if (idx >= 0) {
                const newList = [...list]
                newList[idx] = { ...newList[idx], ...data }
                return setList(newList)
            }
            return list
        },

        async updateJobStatus(_, id, status) {
            const list = getList()
            const idx = list.findIndex((j) => j && j.id === id)
            if (idx >= 0) {
                const newList = [...list]
                newList[idx] = { ...newList[idx], status }
                return setList(newList)
            }
            return list
        },

        async updateJobPID(_, id, pid) {
            const list = getList()
            const idx = list.findIndex((j) => j && j.id === id)
            if (idx >= 0) {
                const newList = [...list]
                newList[idx] = { ...newList[idx], pid }
                return setList(newList)
            }
            return list
        },

        async updateJobName(_, id, name) {
            const list = getList()
            const idx = list.findIndex((j) => j && j.id === id)
            if (idx >= 0) {
                const newList = [...list]
                newList[idx] = { ...newList[idx], name }
                return setList(newList)
            }
            return list
        },

        async removeJob(_, id) {
            const list = getList()
            const nid = String(id)
            const job = list.find((j) => j && String(j.id) === nid)

            if (job) {
                try {
                    const rel = job.output_dir
                    if (rel) {
                        const base = app.isPackaged ? app.getPath('userData') : process.cwd()
                        const abs = path.isAbsolute(rel) ? rel : path.join(base, rel)
                        if (abs.startsWith(base)) {
                            await fs.promises.rm(abs, { recursive: true, force: true })
                        }
                    }
                } catch (e) {
                    console.error('Failed to remove job output dir:', e)
                }
            }

            const newList = list.filter((j) => j && String(j.id) !== nid)
            return setList(newList)
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
