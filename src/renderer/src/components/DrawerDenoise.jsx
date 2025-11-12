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

const DrawerDenoise = ({ open, onClose, onSubmit }) => {
    const [formData, setFormData] = useState({
        type: 'denoise',
        name: 'denoise',

        star_file: 'tomograms.star',

        gpuID: 'None',
        ncpus: 16,

        arch: 'unet-medium',
        pretrained_model: 'None',

        cube_size: 96,
        epochs: 50,

        batch_size: 'None',
        loss_func: 'L2',

        save_interval: 10,

        learning_rate: 3e-4,
        learning_rate_min: 3e-4,

        mixed_precision: true,

        CTF_mode: 'None',
        isCTFflipped: false,
        do_phaseflip_input: true,
        bfactor: 0,
        clip_first_peak_mode: 1,

        snrfalloff: 0,
        deconvstrength: 1,
        highpassnyquist: 0.02,

        with_predict: true,
        pred_tomo_idx: 1
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
                    Denoise
                </Typography>
                <Typography variant="body" gutterBottom>
                    only support even odd input
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

                {/* Numeric Input */}
                <TextField
                    label="job name"
                    type="string"
                    value={formData.name}
                    onChange={(e) => handleChange('name', e.target.value)}
                    fullWidth
                    margin="normal"
                />

                <Box display="flex" alignItems="center" gap={2} marginY={2}>
                    <FormControl variant="standard" sx={{ m: 1, minWidth: 120 }}>
                        <InputLabel>network architecture</InputLabel>
                        <Select
                            // labelId="demo-simple-select-standard-label"
                            // id="demo-simple-select-standard"
                            value={formData.arch}
                            onChange={(e) => handleChange('arch', e.target.value)}
                        // label="Age"
                        >
                            <MenuItem value={'unet-small'}>unet-small</MenuItem>
                            <MenuItem value={'unet-medium'}>unet-medium</MenuItem>
                            <MenuItem value={'unet-large'}>unet-large</MenuItem>
                            <MenuItem value={'scunet-fast'}>scunet-fast</MenuItem>
                        </Select>
                    </FormControl>
                </Box>
                <Box display="flex" alignItems="center" gap={2} marginY={2}>
                    <FormControlLabel
                        control={
                            <Switch
                                checked={formData.with_predict}
                                onChange={(e) => handleChange('with_predict', e.target.checked)}
                            />
                        }
                        label="with predict"
                    />
                    {formData.with_predict && (
                        <TextField
                            label="predict tomo index"
                            type="str"
                            value={formData.pred_tomo_idx}
                            onChange={(e) => handleChange('pred_tomo_idx', e.target.value)}
                            fullWidth
                        />
                    )}
                </Box>

                <Box display="flex" alignItems="center" gap={2} marginY={2}>
                    <TextField
                        label="subtomo size"
                        type="int"
                        value={formData.cube_size}
                        onChange={(e) => handleChange('cube_size', e.target.value)}
                        fullWidth
                    />

                    <TextField
                        label="No. epochs"
                        type="int"
                        value={formData.epochs}
                        onChange={(e) => handleChange('epochs', e.target.value)}
                        fullWidth
                    />
                    <TextField
                        label="saving interval"
                        type="int"
                        value={formData.save_interval}
                        onChange={(e) => handleChange('save_interval', e.target.value)}
                        fullWidth
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
                        label="No. of CPUs"
                        type="number"
                        value={formData.ncpus}
                        onChange={(e) => handleChange('ncpus', e.target.value)}
                        fullWidth
                    />
                </Box>
                <Box display="flex" alignItems="center" gap={2} marginY={2}>
                    <TextField
                        label="pretrained model"
                        value={formData.pretrained_model}
                        fullWidth
                    />
                    <Button
                        variant="contained"
                        color="primary"
                        startIcon={<FolderOpenIcon />}
                        onClick={() => handleFileSelect('pretrained_model', 'openFile')}
                    ></Button>
                </Box>
                <Box
                    display="flex"
                    flexDirection="column"
                    gap={2}
                    marginY={2}
                    padding={2}
                    border="1px solid #ccc"
                    borderRadius={2}
                >
                    {/* Mode selector */}
                    <FormControl variant="standard" sx={{ minWidth: 180 }}>
                        <InputLabel>CTF_mode</InputLabel>
                        <Select
                            value={formData.CTF_mode}
                            onChange={(e) => handleChange('CTF_mode', e.target.value)}
                        >
                            <MenuItem value={'None'}>None</MenuItem>
                            <MenuItem value={'network'}>network</MenuItem>
                            <MenuItem value={'phase_only'}>phase_only</MenuItem>
                            <MenuItem value={'wiener'}>wiener</MenuItem>
                        </Select>
                    </FormControl>

                    {/* Top switch */}
                    <FormControlLabel
                        control={
                            <Switch
                                checked={formData.isCTFflipped}
                                onChange={(e) => handleChange('isCTFflipped', e.target.checked)}
                            />
                        }
                        label="isCTFflipped"
                    />

                    {/* Bottom switch (only visible if CTF_mode ≠ None) */}
                    {formData.CTF_mode !== 'None' && (
                        <FormControlLabel
                            control={
                                <Switch
                                    checked={formData.do_phaseflip_input}
                                    onChange={(e) =>
                                        handleChange('do_phaseflip_input', e.target.checked)
                                    }
                                />
                            }
                            label="do_phaseflip_input"
                        />
                    )}

                    {/* Mode-specific inputs */}
                    {formData.CTF_mode === 'wiener' && (
                        <Box display="flex" alignItems="center" gap={2} marginY={1}>
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
                    )}

                    {formData.CTF_mode === 'network' && (
                        <Box display="flex" alignItems="center" gap={2} marginY={1}>
                            <TextField
                                label="bfactor"
                                type="number"
                                value={formData.bfactor}
                                onChange={(e) => handleChange('bfactor', e.target.value)}
                                fullWidth
                            />
                            <FormControl fullWidth>
                                <InputLabel>clip_first_peak_mode</InputLabel>
                                <Select
                                    value={formData.clip_first_peak_mode}
                                    onChange={(e) =>
                                        handleChange('clip_first_peak_mode', e.target.value)
                                    }
                                >
                                    <MenuItem value="0">0</MenuItem>
                                    <MenuItem value="1">1</MenuItem>
                                    <MenuItem value="2">2</MenuItem>
                                    <MenuItem value="3">3</MenuItem>
                                </Select>
                            </FormControl>
                        </Box>
                    )}
                </Box>

                <Accordion>
                    <AccordionSummary
                        expandIcon={<ExpandMoreIcon />}
                        aria-controls="advanced-settings-content"
                        id="advanced-settings-header"
                    >
                        <Typography>Advanced Settings</Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                        {/* Batch Size */}
                        <Box display="flex" alignItems="center" gap={2} marginY={2}>
                            <TextField
                                label="Batch Size"
                                type="string"
                                value={formData.batch_size}
                                onChange={(e) => handleChange('batch_size', e.target.value)}
                                fullWidth
                            />

                            <FormControl fullWidth>
                                <InputLabel>Loss Function</InputLabel>
                                <Select
                                    value={formData.loss_func}
                                    onChange={(e) => handleChange('loss_func', e.target.value)}
                                >
                                    <MenuItem value="L2">L2</MenuItem>
                                    <MenuItem value="L1">L1</MenuItem>
                                    <MenuItem value="Huber">Huber</MenuItem>
                                </Select>
                            </FormControl>
                            {/* Learning Rate */}
                        </Box>

                        <Box display="flex" alignItems="center" gap={2} marginY={2}>
                            {/* Loss Function */}

                            <TextField
                                label="Learning Rate"
                                type="number"
                                value={formData.learning_rate}
                                onChange={(e) => handleChange('learning_rate', e.target.value)}
                                fullWidth
                            />

                            {/* Minimum Learning Rate */}
                            <TextField
                                label="Min LR"
                                type="number"
                                value={formData.learning_rate_min}
                                onChange={(e) => handleChange('learning_rate_min', e.target.value)}
                                fullWidth
                            />
                        </Box>

                        {/* Mixed Precision */}
                        <FormControlLabel
                            control={
                                <Switch
                                    checked={formData.mixed_precision}
                                    onChange={(e) =>
                                        handleChange('mixed_precision', e.target.checked)
                                    }
                                />
                            }
                            label="Mixed Precision"
                        />
                    </AccordionDetails>
                </Accordion>
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

export default DrawerDenoise
