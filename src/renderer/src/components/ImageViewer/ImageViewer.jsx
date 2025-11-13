import "./index.css";
import { Box, Typography } from "@mui/material";

const ImageViewer = ({ imageSrc }) => {
    return (
        <>
            <Typography variant="h5" gutterBottom>
                Uploaded Image
            </Typography>
            <Box
                sx={{
                    height: 300,
                    marginBottom: 2,
                    display: "flex",
                    justifyContent: "center",
                    alignItems: "center",
                    border: "1px solid #ddd",
                }}
            >
                {imageSrc ? (
                    <img
                        src={imageSrc}
                        alt="Processed"
                        style={{ maxHeight: "100%", maxWidth: "100%" }}
                    />
                ) : (
                    <Typography>No Image to Display</Typography>
                )}
            </Box>
        </>
    );
};

export default ImageViewer;
