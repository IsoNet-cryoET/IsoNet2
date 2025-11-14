import { useState } from 'react'
import {
    Accordion,
    AccordionSummary,
    AccordionDetails,
    Box,
    Typography,
    Divider,
    TextField,
    FormControlLabel,
    Switch,
    Tabs,
    Tab,
    Button
} from '@mui/material'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import FolderOpenIcon from '@mui/icons-material/FolderOpen'
import { useDrawerForm } from './useDrawerForm.js'
import DrawerBase from './DrawerBase.jsx'

const DrawerPrepare = ({ open, onClose, onSubmit }) => {
    const [tabIndex, setTabIndex] = useState(0)
    const {
        formData,
        handleChange,
        handleFileSelect,
        handleSubmit,
    } = useDrawerForm({
        type: 'prepare_star',
        even: 'None',
        odd: 'None',
        full: 'None',
        star_name: 'tomograms.star',
        mask_folder: 'None',
        coordinate_folder: 'None',
        pixel_size: 1.0,
        cs: 2.7,
        voltage: 300,
        ac: 0.1,
        tilt_min: -60,
        tilt_max: 60,
        create_average: false,
        number_subtomos: 1000
    })

    const handleTabChange = (event, newValue) => {
        setTabIndex(newValue)
    }

    return (
        <DrawerBase
            formData={formData}
            open={open}
            onClose={onClose}
        >
            <Typography variant="h6" gutterBottom>
                Prepare Star
            </Typography>
            <Divider sx={{ marginBottom: 2 }} />

            {/* Tabs */}
            <Tabs value={tabIndex} onChange={handleTabChange} centered>
                <Tab label="Full Input" />
                <Tab label="Even/Odd Input" />
            </Tabs>

            {/* Tab Content */}
            {tabIndex === 0 && (
                <Box padding={2}>
                    {/* Full Input */}
                    <Box display="flex" alignItems="center" gap={2} marginY={2}>
                        <TextField
                            label="Full tomograms folder"
                            value={formData.full}
                            fullWidth
                            onChange={(e) => handleChange('full', e.target.value)}
                        />
                        <Button
                            variant="contained"
                            color="primary"
                            startIcon={<FolderOpenIcon />}
                            onClick={() => handleFileSelect('full', 'openDirectory')}
                        />
                    </Box>
                </Box>
            )}

            {tabIndex === 1 && (
                <Box padding={2}>
                    {/* Even Input */}
                    <Box display="flex" alignItems="center" gap={2} marginY={2}>
                        <TextField
                            label="Even tomograms folder"
                            value={formData.even}
                            fullWidth
                            onChange={(e) => handleChange('even', e.target.value)}
                        />
                        <Button
                            variant="contained"
                            color="primary"
                            startIcon={<FolderOpenIcon />}
                            onClick={() => handleFileSelect('even', 'openDirectory')}
                        />
                    </Box>

                    {/* Odd Input */}
                    <Box display="flex" alignItems="center" gap={2} marginY={2}>
                        <TextField
                            label="Odd tomograms folder"
                            value={formData.odd}
                            fullWidth
                            onChange={(e) => handleChange('odd', e.target.value)}
                        />
                        <Button
                            variant="contained"
                            color="primary"
                            startIcon={<FolderOpenIcon />}
                            onClick={() => handleFileSelect('odd', 'openDirectory')}
                        />
                    </Box>
                    <FormControlLabel
                        control={<Switch defaultChecked />}
                        label="create average"
                        value={formData.create_average}
                        onChange={(e) => handleChange('create_average', e.target.checked)}
                        fullWidth
                        margin="normal"
                    />
                </Box>
            )}

            <Box display="flex" gap={2} marginY={2}>
                <TextField
                    id="outlined-required"
                    label="output star file"
                    type="string"
                    value={formData.star_name}
                    onChange={(e) => handleChange('star_name', e.target.value)}
                    fullWidth
                />

                {/* Numeric Input */}
                <TextField
                    label="pixel size in A"
                    type="number"
                    value={formData.pixel_size}
                    onChange={(e) => handleChange('pixel_size', e.target.value)}
                    fullWidth
                />
            </Box>

            {/* Numeric Input */}
            <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                {/* Voltage Input */}
                <TextField
                    label="voltage in kV"
                    type="number"
                    value={formData.voltage}
                    onChange={(e) => handleChange('voltage', e.target.value)}
                    margin="normal"
                />

                {/* Spherical Aberration Input */}
                <TextField
                    label="spherical aberration in mm"
                    type="number"
                    value={formData.cs}
                    onChange={(e) => handleChange('cs', e.target.value)}
                    margin="normal"
                />

                {/* Amplitude Contrast Input */}
                <TextField
                    label="amplitude contrast"
                    type="number"
                    value={formData.ac}
                    onChange={(e) => handleChange('ac', e.target.value)}
                    margin="normal"
                />
            </div>

            <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                {/* Numeric Input */}
                <TextField
                    label="tilt min"
                    type="number"
                    value={formData.tilt_min}
                    onChange={(e) => handleChange('tilt_min', e.target.value)}
                    fullWidth
                    margin="normal"
                />

                {/* Numeric Input */}
                <TextField
                    label="tilt max"
                    type="number"
                    value={formData.tilt_max}
                    onChange={(e) => handleChange('tilt_max', e.target.value)}
                    fullWidth
                    margin="normal"
                />

                {/* Numeric Input */}
                {/* <TextField
                        label="tilt step"
                        type="number"
                        value={formData.tilt_step}
                        onChange={(e) => handleChange('tilt_step', e.target.value)}
                        fullWidth
                        margin="normal"
                    /> */}
            </div>

            {/* Numeric Input */}
            <TextField
                label="number subtomograms per tomo"
                type="number"
                value={formData.number_subtomos}
                onChange={(e) => handleChange('number_subtomos', e.target.value)}
                fullWidth
                margin="normal"
            />

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
                            label="mask folder"
                            type="string"
                            value={formData.mask_folder}
                            onChange={(e) => handleChange('mask_folder', e.target.value)}
                            fullWidth
                        />
                        <Button
                            variant="contained"
                            color="primary"
                            startIcon={<FolderOpenIcon />}
                            onClick={() => handleFileSelect('mask_folder', 'openDirectory')}
                        />
                    </Box>

                    {/* {formData.correct_CTF && ( */}
                    <Box display="flex" alignItems="center" gap={2} marginY={2}>
                        {/* Accumulated Batches */}
                        <TextField
                            label="coordinate folder"
                            type="string"
                            value={formData.coordinate_folder}
                            onChange={(e) => handleChange('coordinate_folder', e.target.value)}
                            fullWidth
                        />
                        <Button
                            variant="contained"
                            color="primary"
                            startIcon={<FolderOpenIcon />}
                            onClick={() =>
                                handleFileSelect('coordinate_folder', 'openDirectory')
                            }
                        />
                    </Box>
                </AccordionDetails>
            </Accordion>

            <Button
                variant="contained"
                color="primary"
                fullWidth
                sx={{ marginTop: 2 }}
                onClick={() => handleSubmit('running', onClose, onSubmit)}
            >
                Run
            </Button>
        </DrawerBase>
    )
}

export default DrawerPrepare
