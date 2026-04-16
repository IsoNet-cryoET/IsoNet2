import { createContext, useContext, useState, useCallback } from 'react';
import { Snackbar, Alert } from '@mui/material';

const ErrorContext = createContext(null); // VERY IMPORTANT: default value is null

export const useError = () => {
    const ctx = useContext(ErrorContext);
    if (!ctx) {
        throw new Error("useError must be used inside <ErrorProvider>");
    }
    return ctx;
};

export const ErrorProvider = ({ children }) => {
    const [errorMessage, setErrorMessage] = useState('');
    const [errorOpen, setErrorOpen] = useState(false);

    const showError = useCallback((msg) => {
        setErrorMessage(msg);
        setErrorOpen(true);
    }, []);

    const closeError = () => setErrorOpen(false);

    return (
        <ErrorContext.Provider value={{ showError }}>
            {children}

            <Snackbar
                open={errorOpen}
                autoHideDuration={5000}
                onClose={closeError}
                anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
            >
                <Alert
                    severity="error"
                    variant="filled"
                    onClose={closeError}
                    sx={{ width: '100%' }}
                >
                    {errorMessage}
                </Alert>
            </Snackbar>
        </ErrorContext.Provider>
    );
};
