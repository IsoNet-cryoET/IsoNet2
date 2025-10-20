// src/theme.js
import { createTheme } from '@mui/material/styles';

const ease = 'cubic-bezier(0.4, 0, 0.2, 1)';

const theme = createTheme({
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          transition: `opacity 150ms ${ease} 0ms`,
          '&:hover': { opacity: 0.96 },
          '&:active': { opacity: 0.92 },
        },
      },
    },
  },
});

export default theme;
