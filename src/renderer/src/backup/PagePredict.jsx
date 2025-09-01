import React from 'react'
import { renderContent } from '../utils/log_handler'
import CleaningServicesIcon from '@mui/icons-material/CleaningServices'
import { Box, TextField, Button } from '@mui/material'

const PagePredict = (props) => {
    const handleClear = () => {
        props.setMessages((prev) => ({ ...prev, predict: [] }))
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
            {renderContent(props.messages.predict)}
        </div>
    )
}

export default PagePredict
