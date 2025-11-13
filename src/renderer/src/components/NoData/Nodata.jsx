import "./index.css";
import { Box, Typography } from "@mui/material";

export default function Nodata({ message, sub }) {
  return (
    <Box className="empty-state-container">
      <img className="empty-state-image" src="figures/no_data.svg" />
      <Typography variant="h6" className="empty-state-title">
        {message}
      </Typography>
      <Typography variant="body2" className="empty-state-subtitle">
        {sub}
      </Typography>
    </Box>
  );
}
