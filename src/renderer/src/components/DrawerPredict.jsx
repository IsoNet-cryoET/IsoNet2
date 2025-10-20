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
import CommandAccordion from './CommandAccordion';

const DrawerPredict = ({ open, onClose, onSubmit }) => {
    const [formData, setFormData] = useState({
        type: 'predict',
        star_file: 'tomograms.star',
        model: 'None',
        output_dir: './corrected_tomos',
        gpuID: 'None',
        input_column: 'rlnDeconvTomoName',
        correct_CTF: false,
        isCTFflipped: false,
        tomo_idx: 'all',
        even_odd_input: true,
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
                    Predict
                </Typography>
                <Divider sx={{ marginBottom: 2 }} />
                <Box display="flex" alignItems="center" gap={2} marginY={2}>
                    <FormControlLabel
                        control={
                            <Switch
                                checked={formData.even_odd_input}
                                onChange={(e) => {
                                    const isChecked = e.target.checked
                                    handleChange('even_odd_input', isChecked)
                                    handleChange('method', isChecked ? 'isonet2-n2n' : 'isonet2')
                                }}
                            />
                        }
                        label="Even/Odd Input"
                    />
                </Box>
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
                {!formData.even_odd_input && (
                    <FormControl variant="standard" sx={{ m: 1, minWidth: 120 }}>
                        <InputLabel>input column</InputLabel>
                        <Select
                            // labelId="demo-simple-select-standard-label"
                            // id="demo-simple-select-standard"
                            value={formData.input_column}
                            onChange={(e) => handleChange('input_column', e.target.value)}
                            // label="Age"
                        >
                            <MenuItem value={'rlnDeconvTomoName'}>rlnDeconvTomoName</MenuItem>
                            <MenuItem value={'rlnTomoName'}>rlnTomoName</MenuItem>
                            <MenuItem value={'rlnDenoisedTomoName'}>rlnDenoisedTomoName</MenuItem>
                            <MenuItem value={'rlnCorrectedTomoName'}>rlnCorrectedTomoName</MenuItem>
                        </Select>
                    </FormControl>
                )}
                <Box display="flex" alignItems="center" gap={2} marginY={2}>
                    <TextField label="network model" value={formData.model} fullWidth />
                    <Button
                        variant="contained"
                        color="primary"
                        startIcon={<FolderOpenIcon />}
                        onClick={() => handleFileSelect('model', 'openFile')}
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
                        <MenuItem value={'rlnDeconvTomoName'}>rlnDeconvTomoName</MenuItem>
                        <MenuItem value={'rlnTomoName'}>rlnTomoName</MenuItem>
                        <MenuItem value={'rlnDenoisedTomoName'}>rlnDenoisedTomoName</MenuItem>
                        <MenuItem value={'rlnCorrectedTomoName'}>rlnCorrectedTomoName</MenuItem>
                    </Select>
                </FormControl>
                <Box display="flex" alignItems="center" gap={2} marginY={2}>
                    <TextField
                        label="output directory"
                        type="string"
                        value={formData.output_dir}
                        onChange={(e) => handleChange('output_dir', e.target.value)}
                        fullWidth
                        margin="normal"
                    />
                </Box>

                <Box display="flex" alignItems="center" gap={2} marginY={2}>
                    <TextField
                        label="gpuID"
                        type="string"
                        value={formData.gpuID}
                        onChange={(e) => handleChange('gpuID', e.target.value)}
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
                <Box display="flex" alignItems="center" gap={2} marginY={2}>
                    <FormControlLabel
                        control={
                            <Switch
                                checked={formData.correct_CTF}
                                onChange={(e) => handleChange('correct_CTF', e.target.checked)}
                            />
                        }
                        label="correct CTF"
                    />
                    <FormControlLabel
                        control={
                            <Switch
                                checked={formData.isCTFflipped}
                                onChange={(e) => handleChange('isCTFflipped', e.target.checked)}
                            />
                        }
                        label="isCTFflipped"
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
                <CommandAccordion formData={formData}/>
            </Box>
        </Drawer>
    )
}

export default DrawerPredict
