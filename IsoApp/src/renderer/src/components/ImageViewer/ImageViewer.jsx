import "./index.css";
import { Box, Typography } from "@mui/material";

const ImageViewer = ({ imageSrc }) => {
    return (
        <>
            <Typography variant="h5" gutterBottom>
                Uploaded Image
            </Typography>
            <Box className="image-preview-box">
                {imageSrc ? (
                    <img src={imageSrc} alt="Processed" className="image-preview-img" />
                ) : (
                    <Typography>No Image to Display</Typography>
                )}
            </Box>
        </>
    );
};

export default ImageViewer;
