import { LinearProgress, Typography, Box } from '@mui/material' // For displaying progress

const ProgressBar = (props) => {
    return (
        <Box sx={{ display: 'flex', alignItems: 'center', width: '100%', margin: '10px 0' }}>
            <Typography
                variant="body2"
                sx={{ minWidth: 35, color: 'text.secondary', whiteSpace: 'nowrap', marginRight: '10px' }} // whiteSpace ensures no wrapping
            >
                {props.currentProgress.description}
            </Typography>
            <Box sx={{ flexGrow: 1, mr: 1 }}>
                <LinearProgress
                    variant="determinate"
                    value={props.currentProgress.percentage}
                // sx={{ marginBottom: 0, height: 8 }} // Removed marginBottom to align details
                />
            </Box>
            <Typography
                variant="body2"
                sx={{ minWidth: 35, color: 'text.secondary', whiteSpace: 'nowrap' }} // whiteSpace ensures no wrapping
            >
                {props.currentProgress.details}
            </Typography>
        </Box>
    )
}

export default ProgressBar
