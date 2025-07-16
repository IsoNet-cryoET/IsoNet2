import React, { useState, useEffect, useCallback, useMemo } from 'react'
import {
    List,
    ListItem,
    ListItemText,
    IconButton,
    ListItemButton,
    CircularProgress
} from '@mui/material'
import DrawerPrepare from './components/DrawerPrepare'
import DrawerRefine from './components/DrawerRefine'
import DrawerRefine_v1 from './components/DrawerRefine_v1'
import DrawerDenoise from './components/DrawerDenoise'
import DrawerPredict from './components/DrawerPredict'
import DrawerDeconv from './components/DrawerDeconv'
import DrawerMask from './components/DrawerMask'
import PageRefine from './components/PageRefine'
import PageRefine_v1 from './components/PageRefine_v1'
import PageDenoise from './components/PageDenoise'
import PagePrepare from './components/PagePrepare'
import PagePredict from './components/PagePredict'
import PageMask from './components/PageMask'
import PageDeconv from './components/PageDeconv'
import PagePost from './components/PagePost'
import PageJobs from './components/PageJobs'
import { mergeMsg, processMessage } from './utils/utils'

import EditIcon from '@mui/icons-material/Edit'
import AppsIcon from '@mui/icons-material/Apps'
import CameraIcon from '@mui/icons-material/Camera'
import { ConstructionOutlined } from '@mui/icons-material'

const drawerComponents = {
    prepare_star: DrawerPrepare,
    denoise: DrawerDenoise,
    refine_v1: DrawerRefine_v1,
    refine: DrawerRefine,
    deconv: DrawerDeconv,
    make_mask: DrawerMask,
    predict: DrawerPredict
}

const App = () => {
    const [drawerState, setDrawerState] = useState({
        prepare_star: false,
        denoise: false,
        refine_v1: false,
        refine: false,
        deconv: false,
        make_mask: false,
        predict: false
    })
    const [messages, setMessages] = useState({
        prepare_star: [],
        denoise: [],
        refine_v1: [],
        refine: [],
        deconv: [],
        make_mask: [],
        predict: []
    })
    const [runningProcesses, setRunningProcesses] = useState({
        prepare_star: false,
        refine: false,
        refine_v1: false,
        denoise: false,
        deconv: false,
        make_mask: false,
        predict: false
    })
    const [starName, setStarName] = useState('')
    const [selectedPrimaryMenu, setSelectedPrimaryMenu] = useState(0)
    const [selectedSecondaryMenu, setSelectedSecondaryMenu] = useState(0)

    // const toggleDrawer = (drawer, open) => {
    //     setDrawerState((prev) => ({ ...prev, [drawer]: open }))
    // }
    const toggleDrawer = useCallback((drawer, open) => {
        setDrawerState((prev) => ({ ...prev, [drawer]: open }))
    }, [])

    // const handleSubmit = (type, data) => {
    //     try {
    //         if (type === 'prepare') setStarName(data.star_name)
    //         api.run(data)
    //     } catch (error) {
    //         console.error(`Error submitting ${type} form:`, error)
    //     }
    //     toggleDrawer(type, false)
    // }
    const handleSubmit = useCallback(
        (type, data) => {
            try {
                if (type === 'prepare_star') setStarName(data.star_name)
                api.run(data)
            } catch (error) {
                console.error(`Error submitting ${type} form:`, error)
            }
            toggleDrawer(type, false)
        },
        [toggleDrawer]
    )

    useEffect(() => {
        const handleIncomingMessage = (data) => {
            setMessages((prev) => ({
                ...prev,
                [data.cmd]: mergeMsg(prev[data.cmd] || [], processMessage(data))
            }))
        }
        window.api.onPythonStderr(handleIncomingMessage)
        window.api.onPythonStdout(handleIncomingMessage)

        const handleRunning = (data) =>
            setRunningProcesses((prev) => ({
                ...prev,
                [data.cmd]: true
            }))

        const handleClosed = (data) =>
            setRunningProcesses((prev) => ({
                ...prev,
                [data.cmd]: false
            }))

        window.api.onPythonRunning(handleRunning)
        window.api.onPythonClosed(handleClosed)

        return () => {
            window.api.offPythonStderr(handleIncomingMessage)
            window.api.offPythonStdout(handleIncomingMessage)
            window.api.offPythonRunning(handleRunning)
            window.api.offPythonClosed(handleClosed)
        }
    }, [])

    const primaryMenus = useMemo(
        () => [
            { icon: <CameraIcon sx={{ color: '#14446e' }} />, label: 'Camera' },
            { icon: <AppsIcon sx={{ color: '#14446e' }} />, label: 'Apps' }
        ],
        []
    )

    const secondaryMenus = useMemo(
        () => [
            { key: 'prepare_star', label: 'Prepare' },
            { key: 'denoise', label: 'Denoise' },
            { key: 'deconv', label: 'Deconvolve' },
            { key: 'make_mask', label: 'Create Mask' },
            { key: 'refine', label: 'Refine' },
            { key: 'refine_v1', label: 'Refine_v1' },
            { key: 'predict', label: 'Predict' },
            { key: 'postprocess', label: 'Postprocess' },
            { key: 'jobs_viewer', label: 'Jobs Viewer' }
        ],
        []
    )
    const Contents = [
        [
            PagePrepare,
            PageDenoise,
            PageDeconv,
            PageMask,
            PageRefine,
            PageRefine_v1,
            PagePredict,
            PagePost,
            PageJobs
        ],
        [null, null]
    ]

    const CurrentComponent = Contents[selectedPrimaryMenu][selectedSecondaryMenu]

    // const handlePrimaryMenuClick = (index) => {
    //     setSelectedPrimaryMenu(index)
    //     setSelectedSecondaryMenu(0) // Reset secondary menu selection
    // }

    // const handleSecondaryMenuClick = (event, index) => {
    //     setSelectedSecondaryMenu(index)
    // }

    // const selectedStyle = {
    //     '&.Mui-selected': {
    //         backgroundColor: '#D5E2F4',
    //         borderRadius: '24px'
    //     }
    // }

    return (
        <div className="outer-container">
            <div className="top-bar">IsoNet 2</div>
            <div className="main-content">
                <div className="primary-menu">
                    <List>
                        {primaryMenus.map((menu, index) => (
                            <ListItem disablePadding key={index}>
                                <ListItemButton
                                    onClick={() => setSelectedPrimaryMenu(index)}
                                    selected={selectedPrimaryMenu === index}
                                    sx={{
                                        '&.Mui-selected': {
                                            backgroundColor: '#D5E2F4',
                                            borderRadius: '24px'
                                        }
                                    }}
                                >
                                    {menu.icon}
                                </ListItemButton>
                            </ListItem>
                        ))}
                    </List>
                </div>

                <div className="secondary-menu">
                    {selectedPrimaryMenu === 0 && (
                        <List>
                            {secondaryMenus.map(({ key, label }, index) => (
                                <ListItem
                                    disablePadding
                                    key={index}
                                    sx={{ '&:hover': { backgroundColor: '#eaebef' } }}
                                >
                                    <ListItemButton
                                        selected={selectedSecondaryMenu === index}
                                        sx={{
                                            '&.Mui-selected': {
                                                backgroundColor: '#D5E2F4',
                                                borderRadius: '24px'
                                            }
                                        }}
                                        onClick={() => setSelectedSecondaryMenu(index)}
                                    >
                                        {runningProcesses[key] && (
                                            <CircularProgress size={20} color="inherit" />
                                        )}
                                        <ListItemText primary={label} />
                                        {index < 7 && (
                                            <IconButton
                                                onClick={() => toggleDrawer(key, true)}
                                                sx={{
                                                    // backgroundColor: '#eaf1fb',
                                                    backgroundColor:
                                                        index === 2 || index === 3 || index === 5
                                                            ? '#f8edeb'
                                                            : '#D5E2F4', // Replace 'defaultBackgroundColor' with your default color

                                                    '&:hover': { backgroundColor: '#e0e0e0' },
                                                    borderRadius: '50%',
                                                    width: 40,
                                                    height: 40,
                                                    display: 'flex',
                                                    justifyContent: 'center',
                                                    alignItems: 'center'
                                                }}
                                            >
                                                <EditIcon sx={{ color: '#14446e', fontSize: 24 }} />
                                            </IconButton>
                                        )}
                                    </ListItemButton>
                                </ListItem>
                            ))}
                        </List>
                    )}

                    {selectedPrimaryMenu === 1 && (
                        <List>
                            {['Page 1', 'Page 2'].map((label, index) => (
                                <ListItem disablePadding key={index}>
                                    <ListItemButton
                                        selected={selectedSecondaryMenu === index}
                                        sx={{
                                            '&.Mui-selected': {
                                                backgroundColor: '#D5E2F4',
                                                borderRadius: '24px'
                                            }
                                        }}
                                        onClick={() => setSelectedSecondaryMenu(index)}
                                    >
                                        <ListItemText primary={label} />
                                    </ListItemButton>
                                </ListItem>
                            ))}
                        </List>
                    )}
                </div>
                <div className="content-area">
                    {CurrentComponent ? (
                        <CurrentComponent
                            starName={starName}
                            setStarName={setStarName}
                            messages={messages}
                            setMessages={setMessages}
                        />
                    ) : (
                        <iframe
                            src="https://isonetcryoet.com/"
                            style={{ width: '100%', height: '100vh', border: 'none' }}
                            title="Embedded Webpage"
                        />
                    )}
                </div>

                {Object.keys(drawerState).map((key) => {
                    const DrawerComponent = drawerComponents[key]
                    return (
                        <DrawerComponent
                            key={key}
                            open={drawerState[key]}
                            onClose={() => toggleDrawer(key, false)}
                            onSubmit={(data) => handleSubmit(key, data)}
                        />
                    )
                })}
            </div>
        </div>
    )
}

export default App
