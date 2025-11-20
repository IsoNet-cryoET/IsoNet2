import "./index.css";
import { useState, useEffect } from 'react'
import { renderContent } from '../LogHandler/log_handler'
import { Box, TextField, Button, CircularProgress } from '@mui/material'
import FolderOpenIcon from '@mui/icons-material/FolderOpen'
import DataTable from '../DataTable'
import { useError } from '../../context/ErrorContext';

const PagePrepare = (props) => {
    const [loading, setLoading] = useState(false)
    const { showError } = useError();

    useEffect(() => {
        const handleJsonUpdate = (data) => {
            props.setJsonData(data.output) // Update the table data
            setLoading(false);
            if (data.error) {
                showError(data.error);
            }
        }
        window.api.on('json-star', handleJsonUpdate)
    }, [])

    const handleFileSelect = async (property) => {
        let filePath
        try {
            filePath = await window.api.call('selectFile', property)
            if (!filePath) { return }
            props.setStarName(filePath) // Update the state
        } catch (error) {
            console.error('Error selecting file:', error)
            showError(`'Error selecting file:'${error}`);
        }
        setLoading(true);
        try {
            await window.api.call('run', {
                id: -1,
                type: 'star2json',
                star_file: filePath,
                json_file: '.to_node.json',
                status: 'completed',
            });
        } catch (error) {
            showError(data.error);
            setLoading(false);
        }
    }
    return (
        <div>
            <Box className="load-star-row">
                <Button
                    disableFocusRipple
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
                <DataTable jsonData={props.jsonData} star_name={props.starName} />
            </Box>

            <div className="page-prepare-logs-container">
                {renderContent(props.messages, props?.selectedJob?.id)}
            </div>
        </div>
    )
}
export default PagePrepare