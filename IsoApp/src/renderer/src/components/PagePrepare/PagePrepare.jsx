import "./index.css";
import { useEffect, useState } from "react";
import { renderContent } from "../LogHandler/log_handler";
import { Box, TextField, Button, CircularProgress } from "@mui/material";
import FolderOpenIcon from "@mui/icons-material/FolderOpen";
import DataTable from "../DataTable";
import { useError } from "../../context/ErrorContext";

const PagePrepare = (props) => {
    const [loading, setLoading] = useState(false);
    const { showError } = useError();

    useEffect(() => {
        const unsubscribe = window.api.on("json-star", (data) => {
            if (data?.output) {
                props.setJsonData(data.output);
            }
            if (data?.error) {
                showError(data.error);
            }
            setLoading(false);
        });

        return () => {
            unsubscribe();
        };
    }, [props.setJsonData, showError]);

    const handleFileSelect = async (property) => {
        let filePath;
        try {
            filePath = await window.api.call("selectFile", property);
            if (!filePath) {
                return;
            }
            props.setStarName(filePath); // Update the state
        } catch (error) {
            console.error("Error selecting file:", error);
            showError(`Error selecting file: ${error}`);
            return;
        }
        setLoading(true);
        try {
            const result = await window.api.call("parseStarFile", filePath);
            if (!result.ok) {
                throw new Error(result.error || "Failed to parse STAR file");
            }
            props.setJsonData(result.output);
        } catch (error) {
            console.error("Error parsing STAR file:", error);
            showError(`Error parsing STAR file: ${error}`);
        } finally {
            setLoading(false);
        }
    };
    return (
        <div>
            <Box className="load-star-row">
                <Button
                    disableFocusRipple
                    variant="outlined"
                    color="primary"
                    startIcon={<FolderOpenIcon />}
                    onClick={() => handleFileSelect("openFile")}
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
                <DataTable
                    jsonData={props.jsonData}
                    star_name={props.starName}
                />
            </Box>

            <div className="page-prepare-logs-container">
                {renderContent(props.messages, props?.selectedJob?.id)}
            </div>
        </div>
    );
};
export default PagePrepare;
