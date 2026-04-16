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
                className: 'drawer-paper',   // ⬅️ 用 className 代替 sx
            }}
        >
            <Box className="drawer-content">
                {children}
                <CommandAccordion formData={formData} />
            </Box>
        </Drawer>

    )
}
