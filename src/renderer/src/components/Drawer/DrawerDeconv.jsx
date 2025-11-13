import {
    Box,
    Typography,
    Divider,
    TextField,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    Button,
} from '@mui/material'
import FolderOpenIcon from '@mui/icons-material/FolderOpen'
import { useDrawerForm } from './useDrawerForm.js'
import DrawerBase from './DrawerBase.jsx'

const DrawerDeconv = ({ open, onClose, onSubmit }) => {
    const {
        formData,
        handleChange,
        handleFileSelect,
        handleSubmit,
    } = useDrawerForm({
        type: 'deconv',
        name: 'deconv',
        star_file: 'tomograms.star',
        input_column: 'rlnTomoName',
        snrfalloff: 1,
        deconvstrength: 1,
        highpassnyquist: 0.02,
        ncpus: 4,
        tomo_idx: 'all'
    })

    return (
        <DrawerBase
            formData={formData}
            open={open}
            onClose={onClose}
        >
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

export default DrawerDeconv
