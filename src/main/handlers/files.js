import fs from 'fs'
import path from 'path'
import { dialog } from 'electron'

export default function files() {
    return {
        async getImageData(_, relativePath) {
            const fullPath = path.resolve(process.cwd(), relativePath)
            try {
                const data = await fs.promises.readFile(fullPath)
                return {
                    success: true,
                    content: `data:image/png;base64,${data.toString('base64')}`
                }
            } catch (err) {
                return { success: false, error: err.message }
            }
        },
        async selectFile(_, property) {
            const result = await dialog.showOpenDialog({
                properties: [property]
            })
            return result.canceled ? null : result.filePaths[0]
        },
        async readFile(_, filePath) {
            try {
                return await fs.promises.readFile(filePath, 'utf-8')
            } catch {
                return null
            }
        },
        isFileExist(_, filePath) {
            return fs.existsSync(filePath)
        },
    }
}
