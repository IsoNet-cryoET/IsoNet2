import "./index.css";
import { Drawer, Box } from '@mui/material'
import CommandAccordion from './CommandAccordion'

export default function DrawerBase({
    formData,
    open,
    onClose,
    children,
}) {
    return (
        <Drawer
            anchor="right"
            open={open}
            onClose={onClose}
            PaperProps={{
                sx: {
                    width: '442px',
                    overflowY: 'scroll'
                }
            }}
        >
            <Box sx={{ width: 400, padding: 2 }}>
                {children}
                <CommandAccordion formData={formData} />
            </Box>
        </Drawer >
    )
}
