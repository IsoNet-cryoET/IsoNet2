import React, { useState, useEffect } from 'react'
import { renderContent } from '../utils/log_handler'
import { Box, TextField, Button } from '@mui/material'
import FolderOpenIcon from '@mui/icons-material/FolderOpen'
import DataTable from './DataTable'
import CleaningServicesIcon from '@mui/icons-material/CleaningServices'

const PagePrepare = (props) => {
    const [JsonData, setJsonData] = useState('')

    useEffect(() => {
        const handleJsonUpdate = (data) => {
            setJsonData(data.output) // Update the table data
        }
        window.api.onJson(handleJsonUpdate)
    }, [])

    useEffect(() => {
        if (!props.starName) return
        api.run({
            id: -1,
            type: 'star2json',
            star_file: props.starName,
            json_file: '.to_node.json',
            status: 'completed'
            //command_line: 'isonet.py star2json ' + props.starName + ' .to_node.json',
        })
    }, [props.starName])

    const handleFileSelect = async (property) => {
        try {
            const filePath = await api.selectFile(property)
            props.setStarName(filePath) // Update the state
            api.run({
                id: -1,
                type: 'star2json',
                star_file: props.starName,
                json_file: '.to_node.json',
                status: 'completed'
                //command_line: 'isonet.py star2json ' + props.starName + ' .to_node.json',
            })
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
                {/* <Button
                    variant="outlined"
                    color="primary"
                    startIcon={<CleaningServicesIcon />}
                    onClick={() => handleClear()}
                    sx={{ height: '56px' }} // Ensure the button has a height
                >
                    clear screen
                </Button> */}
            </Box>
            <DataTable jsonData={JsonData} star_name={props.starName} />

            <div className='page-prepare-logs-container'>
                {renderContent(props.messages, props?.selectedJob?.id)}
            </div>
        </div>
    )
}

export default PagePrepare
