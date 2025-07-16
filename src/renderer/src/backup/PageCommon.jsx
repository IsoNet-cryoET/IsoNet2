import React, { useState, useEffect, useRef } from 'react'
import { renderContent } from './log_handler'
import { Box, TextField, Button } from '@mui/material'
import FolderOpenIcon from '@mui/icons-material/FolderOpen'
import CleaningServicesIcon from '@mui/icons-material/CleaningServices'
import DataTable from './DataTable'

const PageCommon = ({ pageType, messages }) => {
    return (
        <div>
            <Box display="flex" alignItems="center" gap={2} marginY={2}>
                <Button
                    variant="outlined"
                    color="primary"
                    startIcon={<CleaningServicesIcon />}
                    onClick={handleClear}
                    sx={{ height: '56px' }}
                >
                    Clear screen
                </Button>
            </Box>
            {renderContent(messages[pageType])}
        </div>
    )
}

export default PageCommon
