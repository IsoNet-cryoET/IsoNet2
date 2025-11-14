import { handleProcess, spawnWithRuntime } from '../process.js'

export default function other({ getMainWindow }) {
    return {
        async run(event, data) {
            handleProcess(event, data)
        },
        async updateStar(_, data) {
            const filePath = '.to_star.json'
            let updateStarProcess = null
            const fs = await import('fs')

            fs.writeFileSync(filePath, JSON.stringify(data.convertedJson, null, 2))
            updateStarProcess = spawnWithRuntime(
                `isonet.py json2star --json_file "${filePath}" --star_name "${data.star_name}"`
            )
            updateStarProcess.on('close', (code) => {
                console.log(`Python process exited with code ${code}`)
                updateStarProcess = null
            })
        },
        async appClose(_, ok) {
            const win = getMainWindow();
            if (!win) return
            if (ok) {
                win.__forceClose = true
                win.close()
            }
        }
    }
}
