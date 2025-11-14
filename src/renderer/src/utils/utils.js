export const processMessage = (msg) => {
    const output = msg.output

    const progressPercentage = output.match(/(\d+)%/)
    if (progressPercentage) {
        return {
            type: 'bar',
            cmd: msg.cmd,
            description: output.match(/^[^:]+/)[0],
            percentage: parseInt(progressPercentage?.[1] || 0, 10),
            details: output.match(/^[^|]*\|[^|]*\|(.*)/)[1].trim()
        }
    } else if (output.includes('power spectrum')) {
        const epochMatch = output.match(/epoch\s+(\d+)/i)
        const folderMatch = output.match(/to\s+'([^']+)'/i)
        const volumeMatch = output.match(/the tomo file name is\s+(.+)$/i)
        console.log(volumeMatch[1].trim())

        return {
            type: 'power_spectrum',
            cmd: msg.cmd,
            epoch: epochMatch ? parseInt(epochMatch[1], 10) : null,
            folder: folderMatch ? folderMatch[1] : null,
            volume_file: volumeMatch ? volumeMatch[1].trim() : null,
            output: output
        }
    } else {
        return {
            type: 'text',
            cmd: msg.cmd,
            output: output
        }
    }
}

export const mergeMsg = (prevMessages, newMsg) => {
    if (!prevMessages || prevMessages.length == 0) {
        prevMessages = [newMsg]
        return prevMessages
    } else if (newMsg.type === 'bar' && prevMessages[prevMessages.length - 1]?.type === 'bar') {
        return [...prevMessages.slice(0, -1), newMsg]
    } else {
        return [...prevMessages, newMsg]
    }
}
