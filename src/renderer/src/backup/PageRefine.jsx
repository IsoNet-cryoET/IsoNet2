import React from 'react'
import { renderContent } from '../utils/log_handler'
import { IconButton, Box, Button } from '@mui/material'
import StopIcon from '@mui/icons-material/Stop'
import CancelIcon from '@mui/icons-material/Cancel'
import CleaningServicesIcon from '@mui/icons-material/CleaningServices'
const PageRefine = (props) => {
    const handleClear = () => {
        props.setMessages((prev) => ({ ...prev, refine: [] }))
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
            </Box>
            {renderContent(props.messages.refine)}
        </div>
    )
}
export default PageRefine
