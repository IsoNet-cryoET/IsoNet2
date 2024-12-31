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

const DrawerRefine = ({ open, onClose, onSubmit }) => {
    const [formData, setFormData] = useState({
        command: 'refine',
        star_file: 'tomograms.star',
        output_dir: 'isonet_maps',

        gpuID: 'None',
        ncpus: 16,

        method: 'isonet2-n2n',
        arch: 'unet-medium',
        pretrained_model: 'None',
        pretrained_model2: 'None',

        cube_size: 96,
        epochs: 50,

        input_column: 'rlnDeconvTomoName',
        batch_size: 'None',
        acc_batches: 1,
        loss_func: 'L2',
        learning_rate: 3e-4,
        T_max: 10,
        learning_rate_min: 3e-4,
        random_rotation: true,
        mw_weight: 20,
        apply_mw_x1: true,
        compile_model: false,
        mixed_precision: true,

        correct_CTF: false,
        isCTFflipped: false,

        correct_between_tilts: false,
        start_bt_size: 128,

        noise_level: 0,
        noise_mode: 'nofilter',

        with_predict: true,
        split_halves: false,

        even_odd_input: true,
        snrfalloff: 0,
        deconvstrength: 1,
        highpassnyquist: 0.02,
        only_print: true,
        inqueue: true
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

    const handleSubmit = (signal) => {
        const updatedFormData = {
            ...formData,
            only_print: signal.onlyPrint,
            inqueue: signal.inqueue
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
                    Refine
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
                    <FormControlLabel
                        control={
                            <Switch
                                checked={formData.split_halves}
                                onChange={(e) => handleChange('split_halves', e.target.checked)}
                            />
                        }
                        label="Split Halves"
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

                {/* Numeric Input */}
                <TextField
                    label="output directory"
                    type="string"
                    value={formData.output_dir}
                    onChange={(e) => handleChange('output_dir', e.target.value)}
                    fullWidth
                    margin="normal"
                />

                <Box display="flex" alignItems="center" gap={2} marginY={2}>
                    {!formData.even_odd_input && (
                        <FormControl variant="standard" sx={{ m: 1, minWidth: 120 }}>
                            <InputLabel>algorithum</InputLabel>
                            <Select
                                // labelId="demo-simple-select-standard-label"
                                // id="demo-simple-select-standard"
                                value={formData.method}
                                onChange={(e) => handleChange('method', e.target.value)}
                                // label="Age"
                            >
                                <MenuItem value={'isonet2'}>isonet2</MenuItem>
                            </Select>
                        </FormControl>
                    )}
                    {formData.even_odd_input && (
                        <FormControl variant="standard" sx={{ m: 1, minWidth: 120 }}>
                            <InputLabel>algorithum</InputLabel>
                            <Select
                                // labelId="demo-simple-select-standard-label"
                                // id="demo-simple-select-standard"
                                value={formData.method}
                                onChange={(e) => handleChange('method', e.target.value)}
                                // label="Age"
                            >
                                <MenuItem value={'isonet2-n2n'}>isonet2-n2n</MenuItem>
                                <MenuItem value={'n2n'}>n2n</MenuItem>
                            </Select>
                        </FormControl>
                    )}

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
                    <FormControlLabel
                        control={
                            <Switch
                                checked={formData.with_predict}
                                onChange={(e) => handleChange('with_predict', e.target.checked)}
                            />
                        }
                        label="with predict"
                    />
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
                        value={formData.T_max}
                        onChange={(e) => handleChange('T_max', e.target.value)}
                        fullWidth
                    />
                </Box>

                <Box display="flex" alignItems="center" gap={2} marginY={2}>
                    <TextField
                        label="mw weight"
                        type="number"
                        value={formData.mw_weight}
                        onChange={(e) => handleChange('mw_weight', e.target.value)}
                        fullWidth
                    />
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

                {!formData.even_odd_input && (
                    <Box display="flex" alignItems="center" gap={2} marginY={2}>
                        <TextField
                            label="Noise Level"
                            type="number"
                            value={formData.noise_level}
                            onChange={(e) => handleChange('noise_level', e.target.value)}
                            fullWidth
                        />

                        <FormControl variant="standard" sx={{ m: 1, minWidth: 120 }}>
                            <InputLabel>Noise Mode</InputLabel>
                            <Select
                                value={formData.noise_mode}
                                onChange={(e) => handleChange('noise_mode', e.target.value)}
                            >
                                <MenuItem value={'nofilter'}>nofilter</MenuItem>
                                <MenuItem value={'ramp'}>ramp</MenuItem>
                                <MenuItem value={'hamming'}>hamming</MenuItem>
                            </Select>
                        </FormControl>
                    </Box>
                )}

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
                {formData.split_halves && (
                    <Box display="flex" alignItems="center" gap={2} marginY={2}>
                        <TextField
                            label="Pretrained Model 2"
                            value={formData.pretrained_model2}
                            fullWidth
                        />
                        <Button
                            variant="contained"
                            color="primary"
                            startIcon={<FolderOpenIcon />}
                            onClick={() => handleFileSelect('pretrained_model2', 'openFile')}
                        ></Button>
                    </Box>
                )}
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

                            {/* Accumulated Batches */}
                            <TextField
                                label="Accumulated Batches"
                                type="number"
                                value={formData.acc_batches}
                                onChange={(e) => handleChange('acc_batches', e.target.value)}
                                fullWidth
                            />
                        </Box>

                        {formData.correct_CTF && (
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
                                    onChange={(e) =>
                                        handleChange('highpassnyquist', e.target.value)
                                    }
                                    fullWidth
                                />
                            </Box>
                        )}

                        <Box display="flex" alignItems="center" gap={2} marginY={2}>
                            {/* Loss Function */}
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

                        {/* Random Rotation */}
                        <FormControlLabel
                            control={
                                <Switch
                                    checked={formData.random_rotation}
                                    onChange={(e) =>
                                        handleChange('random_rotation', e.target.checked)
                                    }
                                />
                            }
                            label="Random Rotation"
                        />

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

                        <Box display="flex" alignItems="center" gap={2} marginY={2}>
                            <FormControlLabel
                                control={
                                    <Switch
                                        checked={formData.correct_between_tilts}
                                        onChange={(e) =>
                                            handleChange('correct_between_tilts', e.target.checked)
                                        }
                                    />
                                }
                                label="correct between tilts"
                            />
                            <TextField
                                label="start_bt_size"
                                type="number"
                                value={formData.start_bt_size}
                                onChange={(e) => handleChange('start_bt_size', e.target.value)}
                                // fullWidth
                            />
                        </Box>
                    </AccordionDetails>
                </Accordion>
                <Button
                    variant="contained"
                    color="primary"
                    fullWidth
                    sx={{ marginTop: 2 }}
                    onClick={() => handleSubmit({ onlyPrint: false, inqueue: true })}
                >
                    Submit (in queue)
                </Button>
                <Button
                    variant="contained"
                    color="primary"
                    fullWidth
                    sx={{ marginTop: 2 }}
                    onClick={() => handleSubmit({ onlyPrint: false, inqueue: false })}
                >
                    Submit (run immediately)
                </Button>
                <Button
                    variant="contained"
                    color="primary"
                    fullWidth
                    sx={{ marginTop: 2 }}
                    onClick={() => handleSubmit({ onlyPrint: true, inqueue: true })}
                >
                    Print Command
                </Button>
            </Box>
        </Drawer>
    )
}

export default DrawerRefine
