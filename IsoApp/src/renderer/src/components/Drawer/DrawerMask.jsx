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

const DrawerMask = ({ open, onClose, onSubmit }) => {
    const {
        formData,
        handleChange,
        handleFileSelect,
        handleSubmit,
    } = useDrawerForm({
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

    return (
        <DrawerBase
            formData={formData}
            open={open}
            onClose={onClose}
        >
            <Typography variant="h6" gutterBottom>
                Create mask
            </Typography>
            <Typography variant="subtitle1" gutterBottom>
                Only supports full input
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

export default DrawerMask
