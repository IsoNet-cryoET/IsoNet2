import "./progressBar.css";
import { LinearProgress, Typography, Box } from '@mui/material' // For displaying progress

const ProgressBar = ({ currentProgress }) => {
    return (
        <Box className="progressbar-row">
            <Typography variant="body2" className="progressbar-label">
                {currentProgress.description}
            </Typography>

            <Box className="progressbar-track">
                <LinearProgress
                    variant="determinate"
                    value={currentProgress.percentage}
                />
            </Box>

            <Typography variant="body2" className="progressbar-label">
                {currentProgress.details}
            </Typography>
        </Box>
    );
};

export default ProgressBar
