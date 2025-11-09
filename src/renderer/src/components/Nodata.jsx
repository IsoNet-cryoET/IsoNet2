import React from "react";
import { Box, Typography } from "@mui/material";
import "../assets/nodata.css";

export default function Nodata({message, sub}) {
  return (
    <Box className="empty-state-container">
      <img className="empty-state-image" src="figures/no_data.svg"/>
      <Typography variant="h6" className="empty-state-title">
        {message}
      </Typography>
      <Typography variant="body2" className="empty-state-subtitle">
        {sub}
      </Typography>
    </Box>
  );
}
