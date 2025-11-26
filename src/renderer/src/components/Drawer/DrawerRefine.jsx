import {
    Box,
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
import { useDrawerForm } from './useDrawerForm.js'
import DrawerBase from './DrawerBase.jsx'

const DrawerRefine = ({ open, onClose, onSubmit }) => {
    const {
        formData,
        handleChange,
        handleFileSelect,
        handleSubmit,
    } = useDrawerForm({
        type: 'refine',
        star_file: 'tomograms.star',
        name: 'refine',
        gpuID: 'None',
        ncpus: 16,
        method: 'isonet2-n2n',
        arch: 'unet-medium',
        pretrained_model: 'None',
        cube_size: 96,
        epochs: 50,
        input_column: 'rlnDeconvTomoName',
        batch_size: 'None',
        loss_func: 'L2',
        learning_rate: 3e-4,
        save_interval: 10,
        learning_rate_min: 3e-4,
        mw_weight: 20,
        apply_mw_x1: true,
        random_rot_weight: 0.2,
        mixed_precision: true,
        CTF_mode: 'None',
        isCTFflipped: false,
        do_phaseflip_input: true,
        clip_first_peak_mode: 1,
        bfactor: 0,
        noise_level: 0,
        noise_mode: 'nofilter',
        with_preview: true,
        prev_tomo_idx: 1,
        even_odd_input: true,
        snrfalloff: 0,
        deconvstrength: 1,
        highpassnyquist: 0.02
    })

    return (
        <DrawerBase
            formData={formData}
            open={open}
            onClose={onClose}
        >
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

            <TextField
                label="job name"
                type="string"
                value={formData.name}
                onChange={(e) => handleChange('name', e.target.value)}
                fullWidth
                margin="normal"
            />

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
                    </Select>
                </FormControl>
            </Box>
            <Box display="flex" alignItems="center" gap={2} marginY={2}>
                <FormControlLabel
                    control={
                        <Switch
                            checked={formData.with_preview}
                            onChange={(e) => handleChange('with_preview', e.target.checked)}
                        />
                    }
                    label="with preview"
                />
                {formData.with_preview && (
                    <TextField
                        label="preview tomo index"
                        type="str"
                        value={formData.prev_tomo_idx}
                        onChange={(e) => handleChange('prev_tomo_idx', e.target.value)}
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
                    </Box>

                    <Box display="flex" alignItems="center" gap={2} marginY={2}>
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

                    <Box display="flex" alignItems="center" gap={2} marginY={2}>
                        <TextField
                            label="random_rot_weight"
                            type="number"
                            value={formData.random_rot_weight}
                            onChange={(e) => handleChange('random_rot_weight', e.target.value)}
                            fullWidth
                        />
                        <FormControlLabel
                            control={
                                <Switch
                                    checked={formData.apply_mw_x1}
                                    onChange={(e) =>
                                        handleChange('apply_mw_x1', e.target.checked)
                                    }
                                />
                            }
                            label="apply_mw_x1"
                        />
                    </Box>
                </AccordionDetails>
            </Accordion>
            <Button
                variant="contained"
                color="primary"
                fullWidth
                sx={{ marginTop: 2 }}
                onClick={() => handleSubmit('inqueue', onClose, onSubmit)}
            >
                Submit (in queue)
            </Button>
            <Button
                variant="contained"
                color="primary"
                fullWidth
                sx={{ marginTop: 2 }}
                onClick={() => handleSubmit('running', onClose, onSubmit)}
            >
                Submit (run immediately)
            </Button>
        </DrawerBase>
    )
}

export default DrawerRefine
