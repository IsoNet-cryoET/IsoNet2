import { useState } from 'react'
import { renderContent } from '../utils/log_handler'
import { IconButton, Box, Button } from '@mui/material'
import AddCardIcon from '@mui/icons-material/AddCard'
import CleaningServicesIcon from '@mui/icons-material/CleaningServices'
import { mergeMsg, processMessage } from '../utils/utils'

const PageCommon = (props) => {
    const handleFileSelect = async (field, property) => {
        const filePath = await window.api.selectFile(property)
        if (!filePath) return

        const fileContent = await window.api.readFile(filePath)
        if (!fileContent) return

        const lines = fileContent
            .split(/\r?\n|\r/g) // handles both \n and ^M
            .filter((line) => line.trim() !== '') // remove empty lines
        const newMessages = []
        for (const line of lines) {
            const msg = { cmd: 'denoise', output: line }
            const processed = processMessage(msg)
            const merged = mergeMsg(newMessages, processed)
            newMessages.length = 0
            newMessages.push(...merged)
        }

        props.setMessages(newMessages)
    }

    const handleClear = () => {
        props.setMessages([])
    }
    return (
        <div>
            <Box display="flex" alignItems="center" gap={2} marginY={2}>
                <Button
                    variant="outlined"
                    color="primary"
                    startIcon={<CleaningServicesIcon />}
                    onClick={() => handleClear()}
                    sx={{ height: '56px' }} // Ensure the button has a height
                >
                    clear screen
                </Button>

                <Button
                    variant="outlined"
                    color="primary"
                    startIcon={<AddCardIcon />}
                    onClick={() => handleFileSelect('log_file', 'openFile')}
                    sx={{ height: '56px' }} // Ensure the button has a height
                >
                    reload logs
                </Button>
            </Box>
            {renderContent(props.messages || [])}
        </div>
    )
}
export default PageCommon
