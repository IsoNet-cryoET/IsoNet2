import Store from 'electron-store'
import { app } from 'electron'

export function createStore(cwd) {
    return new Store({
        name: 'settings',
        fileExtension: 'json',
        cwd,
        defaults: { counter: 0, jobList: [] },
        schema: {
            counter: { type: 'number' },
            jobList: { type: 'array', items: { type: 'object' } },
        },
    })
}

export function createEnvironmentStore() {
    return new Store({
        name: 'environment',
        fileExtension: 'json',
        cwd: app.getPath('userData'),
        defaults: { condaEnv: '', IsoNetPath: '' },
        schema: {
            condaEnv: { type: 'string' },
            IsoNetPath: { type: 'string' },
        },
    })
}