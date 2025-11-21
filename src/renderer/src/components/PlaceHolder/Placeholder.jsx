import "./index.css";
import { Box, Typography } from "@mui/material";

export default function Placeholder({ src, title, subtitle }) {
  return (
    <Box className="empty-state-container">
      <img className="empty-state-image" src={src} />
      <Typography variant="h6" className="empty-state-title">
        {title}
      </Typography>
      <Typography variant="body2" className="empty-state-subtitle">
        {subtitle}
      </Typography>
    </Box>
  );
}
