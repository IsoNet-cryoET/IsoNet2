import React, { useState, useEffect } from 'react'
import { renderContent } from '../utils/log_handler'
import { Box, TextField, Button, CircularProgress } from '@mui/material'
import FolderOpenIcon from '@mui/icons-material/FolderOpen'
import DataTable from './DataTable'

const PagePrepare = (props) => {
    const [JsonData, setJsonData] = useState('')
    const [loading, setLoading] = useState(false)

    useEffect(() => {
        const handleJsonUpdate = (data) => {
            setJsonData(data.output) // Update the table data
            setLoading(false);
        }
        window.api.on('json-star', handleJsonUpdate)
    }, [])

    useEffect(() => {
        if (!props.starName) return;
        const runStar2Json = async () => {
            setLoading(true);
            await window.api.call('run', {
                id: -1,
                type: 'star2json',
                star_file: props.starName,
                json_file: '.to_node.json',
                status: 'completed',
            });
        };
        runStar2Json();
    }, [props.starName]);

    const handleFileSelect = async (property) => {
        try {
            const filePath = await window.api.call('selectFile', property)
            props.setStarName(filePath) // Update the state
        } catch (error) {
            console.error('Error selecting file:', error)
        }
    }
    return (
        <div>
            <Box display="flex" alignItems="center" gap={2} marginY={2}>
                <Button
                    variant="outlined"
                    color="primary"
                    startIcon={<FolderOpenIcon />}
                    onClick={() => handleFileSelect('openFile')}
                    sx={{
                        height: '56px',
                        trainsition: 'auto'
                    }} // Ensure the button has a height
                >
                    Load from star
                </Button>
                <TextField
                    label="current star file"
                    value={props.starName}
                    fullWidth
                    disabled
                    sx={{ height: '56px' }} // Set the TextField's height to match the button
                />
            </Box>
            <Box position="relative" minHeight={200}>
                {loading && (
                    <Box
                        sx={{
                            position: 'absolute',
                            inset: 0,
                            backgroundColor: 'rgba(255,255,255,0.6)',
                            display: 'flex',
                            flexDirection: 'column',
                            alignItems: 'center',
                            justifyContent: 'center',
                            zIndex: 10,
                        }}
                    >
                        <CircularProgress color="primary" />
                        <Box sx={{ mt: 2, fontWeight: 500, color: 'text.secondary' }}>
                            Loading data...
                        </Box>
                    </Box>
                )}
                <DataTable jsonData={JsonData} star_name={props.starName} />
            </Box>
            <div className='page-prepare-logs-container'>
                {renderContent(props.messages, props?.selectedJob?.id)}
            </div>
        </div>
    )
}


export default PagePrepare
