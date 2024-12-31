import React from 'react'
import { renderContent } from './log_handler'
import CleaningServicesIcon from '@mui/icons-material/CleaningServices'
import { Box, TextField, Button } from '@mui/material'

const PageMask = (props) => {
    const handleClear = () => {
        props.setMaskMessages([])
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
            {renderContent(props.maskMessages)}
        </div>
    )
}

export default PageMask
