import "./index.css";
import { useState, useEffect } from 'react'
import { renderContent } from '../LogHandler/log_handler'
import { Box, TextField, Button, CircularProgress } from '@mui/material'
import FolderOpenIcon from '@mui/icons-material/FolderOpen'
import DataTable from '../DataTable'

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
            <Box className="load-star-row">
                <Button
                    variant="outlined"
                    color="primary"
                    startIcon={<FolderOpenIcon />}
                    onClick={() => handleFileSelect('openFile')}
                    className="load-star-button"
                >
                    Load from star
                </Button>

                <TextField
                    label="current star file"
                    value={props.starName}
                    fullWidth
                    disabled
                    className="load-star-textfield"
                />
            </Box>

            <Box className="data-table-wrapper">
                {loading && (
                    <Box className="loading-overlay">
                        <CircularProgress color="primary" />
                        <Box className="loading-text">Loading data...</Box>
                    </Box>
                )}
                <DataTable jsonData={JsonData} star_name={props.starName} />
            </Box>

            <div className="page-prepare-logs-container">
                {renderContent(props.messages, props?.selectedJob?.id)}
            </div>
        </div>
    )
}
export default PagePrepare