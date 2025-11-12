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
import CommandAccordion from './CommandAccordion'

const DrawerDeconv = ({ open, onClose, onSubmit }) => {
    const [formData, setFormData] = useState({
        type: 'deconv',
        name: 'deconv',

        star_file: 'tomograms.star',
        input_column: 'rlnTomoName',
        snrfalloff: 1,
        deconvstrength: 1,
        highpassnyquist: 0.02,
        // chunk_size: int=None,
        // overlap_rate: float= 0.25,
        ncpus: 4,
        tomo_idx: 'all'
    })

    // 处理表单字段变化
    const handleChange = (field, value) => {
        setFormData((prev) => ({ ...prev, [field]: value }))
    }

    const handleFileSelect = async (field, property) => {
        let folderPath = await window.api.call('selectFile', property)
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
                    Deconvolve
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
                <TextField
                    label="job name"
                    type="string"
                    value={formData.name}
                    onChange={(e) => handleChange('name', e.target.value)}
                    fullWidth
                    margin="normal"
                />
                <Box display="flex" alignItems="center" gap={2} marginY={2}>
                    <TextField
                        label="snrfalloff"
                        type="number"
                        value={formData.snrfalloff}
                        onChange={(e) => handleChange('snrfalloff', e.target.value)}
                        fullWidth
                    />
                    <TextField
                        label="deconvstrength"
                        type="number"
                        value={formData.deconvstrength}
                        onChange={(e) => handleChange('deconvstrength', e.target.value)}
                        fullWidth
                    />
                    <TextField
                        label="highpassnyquist"
                        type="number"
                        value={formData.highpassnyquist}
                        onChange={(e) => handleChange('highpassnyquist', e.target.value)}
                        fullWidth
                    />
                </Box>
                <Box display="flex" alignItems="center" gap={2} marginY={2}>
                    <TextField
                        label="No. of CPUs"
                        type="number"
                        value={formData.ncpus}
                        onChange={(e) => handleChange('ncpus', e.target.value)}
                        fullWidth
                    />
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

export default DrawerDeconv
