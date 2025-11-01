import React, { useState } from 'react'
import {
    Box,
    Drawer,
    Typography,
    Divider,
    TextField,
    FormControl,
    FormControlLabel,
    Switch,
    InputLabel,
    Select,
    MenuItem,
    Button,
    Accordion,
    AccordionSummary,
    AccordionDetails
} from '@mui/material'
import FolderOpenIcon from '@mui/icons-material/FolderOpen'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import CleaningServicesIcon from '@mui/icons-material/CleaningServices'
import CommandAccordion from './CommandAccordion'

const DrawerMask = ({ open, onClose, onSubmit }) => {
    const [formData, setFormData] = useState({
        type: 'make_mask',
        star_file: 'tomograms.star',
        name: 'mask',
        input_column: 'rlnTomoName',
        patch_size: 4,
        density_percentage: 50,
        std_percentage: 50,
        z_crop: 0.2,
        tomo_idx: 'all'
    })

    // 处理表单字段变化
    const handleChange = (field, value) => {
        setFormData((prev) => ({ ...prev, [field]: value }))
    }

    const handleFileSelect = async (field, property) => {
        let folderPath = await api.selectFile(property)
        setFormData((prevState) => ({
            ...prevState,
            [field]: folderPath
        }))
    }
    const handleSubmit = (status) => {
        const updatedFormData = {
            ...formData,
            status
        }
        onSubmit(updatedFormData)
        onClose()
    }

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
                <Typography variant="h6" gutterBottom>
                    Create mask
                </Typography>
                <Typography variant="body" gutterBottom>
                    does not support even odd input
                </Typography>
                <Divider sx={{ marginBottom: 2 }} />
                <Box display="flex" alignItems="center" gap={2} marginY={2}>
                    <TextField
                        label="input star file"
                        value={formData.star_file}
                        fullWidth
                        onChange={(e) => handleChange('star_file', e.target.value)}
                    />
                    <Button
                        variant="contained"
                        color="primary"
                        startIcon={<FolderOpenIcon />}
                        onClick={() => handleFileSelect('star_file', 'openFile')}
                    ></Button>
                </Box>
                <TextField
                    label="job name"
                    type="string"
                    value={formData.name}
                    onChange={(e) => handleChange('name', e.target.value)}
                    fullWidth
                    margin="normal"
                />
                <FormControl variant="standard" sx={{ m: 1, minWidth: 120 }}>
                    <InputLabel>input column</InputLabel>
                    <Select
                        // labelId="demo-simple-select-standard-label"
                        // id="demo-simple-select-standard"
                        value={formData.input_column}
                        onChange={(e) => handleChange('input_column', e.target.value)}
                        // label="Age"
                    >
                        <MenuItem value={'rlnTomoName'}>rlnTomoName</MenuItem>
                        <MenuItem value={'rlnDeconvTomoName'}>rlnDeconvTomoName</MenuItem>
                        <MenuItem value={'rlnTomoReconstructedTomogramHalf1'}>
                            rlnTomoReconstructedTomogramHalf1
                        </MenuItem>
                        <MenuItem value={'rlnTomoReconstructedTomogramHalf2'}>
                            rlnTomoReconstructedTomogramHalf2
                        </MenuItem>
                        <MenuItem value={'rlnDenoisedTomoName'}>rlnDenoisedTomoName</MenuItem>
                        <MenuItem value={'rlnCorrectedTomoName'}>rlnCorrectedTomoName</MenuItem>
                    </Select>
                </FormControl>
                <Box display="flex" alignItems="center" gap={2} marginY={2}>
                    <TextField
                        label="density_percentage"
                        type="number"
                        value={formData.density_percentage}
                        onChange={(e) => handleChange('density_percentage', e.target.value)}
                        fullWidth
                    />
                    <TextField
                        label="std_percentage"
                        type="number"
                        value={formData.std_percentage}
                        onChange={(e) => handleChange('std_percentage', e.target.value)}
                        fullWidth
                    />
                </Box>
                <Box display="flex" alignItems="center" gap={2} marginY={2}>
                    <TextField
                        label="patch_size"
                        type="number"
                        value={formData.patch_size}
                        onChange={(e) => handleChange('patch_size', e.target.value)}
                        fullWidth
                    />
                    <TextField
                        label="z_crop"
                        type="number"
                        value={formData.z_crop}
                        onChange={(e) => handleChange('z_crop', e.target.value)}
                        fullWidth
                    />
                </Box>
                <Box display="flex" alignItems="center" gap={2} marginY={2}>
                    <TextField
                        label="tomo index"
                        type="string"
                        value={formData.tomo_idx}
                        onChange={(e) => handleChange('tomo_idx', e.target.value)}
                        fullWidth
                    />
                </Box>
                <Button
                    variant="contained"
                    color="primary"
                    fullWidth
                    sx={{ marginTop: 2 }}
                    onClick={() => handleSubmit('inqueue')}
                >
                    Submit (in queue)
                </Button>
                <Button
                    variant="contained"
                    color="primary"
                    fullWidth
                    sx={{ marginTop: 2 }}
                    onClick={() => handleSubmit('running')}
                >
                    Submit (run immediately)
                </Button>
                <CommandAccordion formData={formData} />
            </Box>
        </Drawer>
    )
}

export default DrawerMask
