// src/theme.js
import { createTheme } from '@mui/material/styles';

const theme = createTheme({
  components: {
    MuiButton: {
      styleOverrides: {
        outlined: {
          transition: `border-color 150ms ease, background-color 150ms ease`,
          '&:hover': {
            backgroundColor: 'rgba(0, 0, 0, 0.05)',
          },
          '&:active': {
            backgroundColor: 'rgba(0, 0, 0, 0.1)',
          },
        },
      },
    },
  },
});

export default theme;
