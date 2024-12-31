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
    Tabs,
    Tab,
    InputLabel,
    Select,
    MenuItem,
    Button
} from '@mui/material'
import FolderOpenIcon from '@mui/icons-material/FolderOpen'

const DrawerPrepare = ({ open, onClose, onSubmit }) => {
    const [tabIndex, setTabIndex] = useState(0)

    const [formData, setFormData] = useState({
        command: 'prepare_star',
        even: 'EVN',
        odd: 'ODD',
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
        tilt_step: 3,
        create_average: true,
        number_subtomos: 1000,
        only_print: true,
        inqueue: true
    })

    const handleChange = (field, value) => {
        setFormData((prev) => ({ ...prev, [field]: value }))
    }

    // const handleFolderSelect = async () => {
    //     let folderPath = await api.selectFolder()
    //     setFormData((prevState) => ({
    //         ...prevState,
    //         folderPath: folderPath
    //     }))
    // }

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
    const handleTabChange = (event, newValue) => {
        setTabIndex(newValue)
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
                            onChange={(e) => handleChange('create_average', e.target.value)}
                            fullWidth
                            margin="normal"
                        />
                    </Box>
                )}

                {/* Folder Selector */}
                {/* <Box display="flex" alignItems="center" gap={2} marginY={2}>
                    <TextField label="Full tomograms folder" value={formData.full} fullWidth />
                    <Button
                        variant="contained"
                        color="primary"
                        startIcon={<FolderOpenIcon />}
                        onClick={() => handleFileSelect('full', 'openDirectory')}
                    ></Button>
                </Box>
                <Box display="flex" alignItems="center" gap={2} marginY={2}>
                    <TextField label="Even tomograms folder" value={formData.even} fullWidth />
                    <Button
                        variant="contained"
                        color="primary"
                        startIcon={<FolderOpenIcon />}
                        onClick={() => handleFileSelect('even', 'openDirectory')}
                    ></Button>
                </Box>
                <Box display="flex" alignItems="center" gap={2} marginY={2}>
                    <TextField label="Odd tomograms folder" value={formData.odd} fullWidth />
                    <Button
                        variant="contained"
                        color="primary"
                        startIcon={<FolderOpenIcon />}
                        onClick={() => handleFileSelect('odd', 'openDirectory')}
                    ></Button>
                </Box> */}
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
                    <TextField
                        label="tilt step"
                        type="number"
                        value={formData.tilt_step}
                        onChange={(e) => handleChange('tilt_step', e.target.value)}
                        fullWidth
                        margin="normal"
                    />
                </div>

                {/* Boolean Selector */}
                {/* <FormControl fullWidth margin="normal">
                    <InputLabel>create average</InputLabel>
                    <Select
                        value={formData.create_average}
                        onChange={(e) => handleChange('create_average', e.target.value)}
                    >
                        <MenuItem value="true">True</MenuItem>
                        <MenuItem value="false">False</MenuItem>
                    </Select>
                </FormControl> */}

                {/* Numeric Input */}
                <TextField
                    label="number subtomograms per tomo"
                    type="number"
                    value={formData.number_subtomos}
                    onChange={(e) => handleChange('number_subtomos', e.target.value)}
                    fullWidth
                    margin="normal"
                />

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

export default DrawerPrepare
