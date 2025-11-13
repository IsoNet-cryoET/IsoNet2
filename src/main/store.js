import Store from 'electron-store'

export function createStore(cwd) {
    return new Store({
        name: 'settings',
        fileExtension: 'json',
        cwd,
        defaults: { counter: 0, condaEnv: '', IsoNetPath: '', jobList: [] },
        schema: {
            counter: { type: 'number' },
            jobList: { type: 'array', items: { type: 'object' } },
        },
    })
}
