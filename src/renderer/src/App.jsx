import React, { useState, useEffect } from 'react'
import DrawerPrepare from './components/DrawerPrepare'
import {
    List,
    ListItem,
    ListItemText,
    IconButton,
    ListItemButton,
    Backdrop,
    Box,
    CircularProgress
} from '@mui/material'
import DrawerRefine from './components/DrawerRefine'
import DrawerDenoise from './components/DrawerDenoise'
import DrawerPredict from './components/DrawerPredict'
import DrawerDeconv from './components/DrawerDeconv'
import DrawerMask from './components/DrawerMask'

import EditIcon from '@mui/icons-material/Edit'
import AppsIcon from '@mui/icons-material/Apps'
import CameraIcon from '@mui/icons-material/Camera'

import PageRefine from './components/PageRefine'
import PagePrepare from './components/PagePrepare'
import PagePredict from './components/PagePredict'
import PageMask from './components/PageMask'
import PageDeconv from './components/PageDeconv'
import PagePost from './components/PagePost'
import PageJobs from './components/PageJobs'

import { mergeMsg, processMessage } from './utils/utils'
import { ConnectedTvOutlined } from '@mui/icons-material'

const App = () => {
    const [prepareDrawerOpen, setPrepareDrawerOpen] = useState(false)
    const [refineDrawerOpen, setRefineDrawerOpen] = useState(false)
    const [deconvDawerOpen, setDeconvDrawerOpen] = useState(false)
    const [maskDrawerOpen, setMaskDrawerOpen] = useState(false)
    const [predictDrawerOpen, setPredictDrawerOpen] = useState(false)

    const [prepareMessages, setPrepareMessages] = useState([])
    const [refineMessages, setRefineMessages] = useState([])
    const [deconvMessages, setDeconvMessages] = useState([])
    const [maskMessages, setMaskMessages] = useState([])
    const [predictMessages, setPredictMessages] = useState([])
    const [runningProcesses, setRunningProcesses] = useState({
        prepare_star: false,
        refine: false,
        deconv: false,
        make_mask: false,
        predict: false
    })

    const [starName, setStarName] = useState('')
    // const [JsonData, setJsonData] = useState('')

    const handleSubmitPrepare = (data) => {
        try {
            setStarName(data.star_name)
            // setPrepareMessages(() => [])
            api.run(data)
        } catch (error) {
            console.error('Error submitting form:', error)
        }
        setPrepareDrawerOpen(false)
    }

    const handleSubmitRefine = (data) => {
        try {
            // setRefineMessages(() => [])
            api.run(data)
        } catch (error) {
            console.error('Error submitting form:', error)
        }
        setRefineDrawerOpen(false)
    }

    const handleSubmitDeconv = (data) => {
        try {
            // setDeconvMessages(() => [])
            api.run(data)
        } catch (error) {
            console.error('Error submitting form:', error)
        }
        setDeconvDrawerOpen(false)
    }

    const handleSubmitPredict = (data) => {
        try {
            // setPredictMessages(() => [])
            api.run(data)
        } catch (error) {
            console.error('Error submitting form:', error)
        }
        setPredictDrawerOpen(false)
    }

    const handleSubmitMask = (data) => {
        try {
            // setMaskMessages(() => [])
            api.run(data)
        } catch (error) {
            console.error('Error submitting form:', error)
        }
        setMaskDrawerOpen(false)
    }
    window.api.onPythonRunning((data) => {
        setRunningProcesses((prevState) => {
            if (data.cmd in prevState) {
                return {
                    ...prevState,
                    [data.cmd]: true
                }
            }
            return prevState
        })
    })

    window.api.onPythonClosed((data) => {
        setRunningProcesses((prevState) => {
            if (data.cmd in prevState) {
                return {
                    ...prevState,
                    [data.cmd]: false
                }
            }
            return prevState
        })
    })
    useEffect(() => {
        const handleIncomingMessage = (data) => {
            let newMsg = processMessage(data)
            if (data.cmd === 'prepare_star') {
                setPrepareMessages((prevMessages) => mergeMsg(prevMessages, newMsg))
            } else if (data.cmd === 'refine') {
                setRefineMessages((prevMessages) => mergeMsg(prevMessages, newMsg))
            } else if (data.cmd === 'predict') {
                setPredictMessages((prevMessages) => mergeMsg(prevMessages, newMsg))
            } else if (data.cmd === 'make_mask') {
                setMaskMessages((prevMessages) => mergeMsg(prevMessages, newMsg))
            } else if (data.cmd === 'deconv') {
                setDeconvMessages((prevMessages) => mergeMsg(prevMessages, newMsg))
            }
        }
        window.api.onPythonStderr(handleIncomingMessage)
        window.api.onPythonStdout(handleIncomingMessage)

        // // Cleanup listeners on unmount
        // return () => {
        //     window.api.offPythonStderr(handleIncomingMessage)
        //     window.api.offPythonStdout(handleIncomingMessage)
        // }
    }, [])

    // selected menu index
    const [selectedPrimaryMenu, setSelectedPrimaryMenu] = useState(0)
    const [selectedSecondaryMenu, setSelectedSecondaryMenu] = useState(0)

    const Contents = [
        [PagePrepare, PageDeconv, PageMask, PageRefine, PagePredict, PagePost, PageJobs],
        [null, null]
    ]
    const CurrentComponent = Contents[selectedPrimaryMenu][selectedSecondaryMenu]

    const handlePrimaryMenuClick = (index) => {
        setSelectedPrimaryMenu(index)
        setSelectedSecondaryMenu(0) // Reset secondary menu selection
    }

    const handleSecondaryMenuClick = (event, index) => {
        setSelectedSecondaryMenu(index)
    }

    const selectedStyle = {
        '&.Mui-selected': {
            backgroundColor: '#D5E2F4',
            borderRadius: '24px'
        }
    }
    return (
        <div className="outer-container">
            <div className="top-bar">IsoNet 2</div>
            <div className="main-content">
                <div className="primary-menu">
                    <List>
                        <ListItem disablePadding>
                            <ListItemButton
                                onClick={() => handlePrimaryMenuClick(0)}
                                selected={selectedPrimaryMenu === 0}
                                sx={selectedStyle}
                            >
                                <CameraIcon sx={{ color: '#14446e' }} />
                            </ListItemButton>
                        </ListItem>
                        <ListItem disablePadding>
                            <ListItemButton
                                onClick={() => handlePrimaryMenuClick(1)}
                                selected={selectedPrimaryMenu === 1}
                                sx={selectedStyle}
                            >
                                <AppsIcon sx={{ color: '#14446e' }} />
                            </ListItemButton>
                        </ListItem>
                    </List>
                </div>

                <div className="secondary-menu">
                    {selectedPrimaryMenu === 0 && (
                        <List>
                            <ListItem
                                disablePadding
                                sx={{
                                    '&:hover': { backgroundColor: '#eaebef' }
                                }}
                            >
                                <ListItemButton
                                    selected={selectedSecondaryMenu === 0}
                                    sx={selectedStyle}
                                    onClick={(event) => handleSecondaryMenuClick(event, 0)}
                                >
                                    {runningProcesses['prepare_star'] === true ? (
                                        <CircularProgress size={20} color="inherit" />
                                    ) : null}
                                    <ListItemText primary="Prepare" />
                                    <IconButton
                                        onClick={() => setPrepareDrawerOpen(true)}
                                        sx={{
                                            backgroundColor: '#eaf1fb', // Set the background color
                                            '&:hover': {
                                                backgroundColor: '#e0e0e0' // Optional: Hover effect
                                            },
                                            borderRadius: '50%', // Optional: Keeps the background circular
                                            width: '40px', // Increase clickable area width
                                            height: '40px', // Increase clickable area height
                                            display: 'flex', // Center the icon within the button
                                            justifyContent: 'center',
                                            alignItems: 'center'
                                        }}
                                    >
                                        <EditIcon sx={{ color: '#14446e', fontSize: '24px' }} />{' '}
                                        {/* Keep the icon size fixed */}
                                    </IconButton>
                                </ListItemButton>
                            </ListItem>
                            <ListItem
                                disablePadding
                                sx={{
                                    '&:hover': { backgroundColor: '#eaebef' }
                                }}
                            >
                                <ListItemButton
                                    selected={selectedSecondaryMenu === 1}
                                    sx={selectedStyle}
                                    onClick={(event) => handleSecondaryMenuClick(event, 1)}
                                >
                                    {runningProcesses['deconv'] === true ? (
                                        <CircularProgress size={20} color="inherit" />
                                    ) : null}
                                    <ListItemText primary="Deconvolve" />
                                    <IconButton
                                        onClick={() => setDeconvDrawerOpen(true)}
                                        sx={{
                                            backgroundColor: '#eaf1fb', // Set the background color
                                            '&:hover': {
                                                backgroundColor: '#e0e0e0' // Optional: Hover effect
                                            },
                                            borderRadius: '50%', // Optional: Keeps the background circular
                                            width: '40px', // Increase clickable area width
                                            height: '40px', // Increase clickable area height
                                            display: 'flex', // Center the icon within the button
                                            justifyContent: 'center',
                                            alignItems: 'center'
                                        }}
                                    >
                                        <EditIcon sx={{ color: '#14446e', fontSize: '24px' }} />{' '}
                                        {/* Keep the icon size fixed */}
                                    </IconButton>
                                </ListItemButton>
                            </ListItem>
                            <ListItem
                                disablePadding
                                sx={{
                                    '&:hover': { backgroundColor: '#eaebef' }
                                }}
                            >
                                <ListItemButton
                                    selected={selectedSecondaryMenu === 2}
                                    sx={selectedStyle}
                                    onClick={(event) => handleSecondaryMenuClick(event, 2)}
                                >
                                    {runningProcesses['make_mask'] === true ? (
                                        <CircularProgress size={20} color="inherit" />
                                    ) : null}
                                    <ListItemText primary="Create Mask" />
                                    <IconButton
                                        onClick={() => setMaskDrawerOpen(true)}
                                        sx={{
                                            backgroundColor: '#eaf1fb', // Set the background color
                                            '&:hover': {
                                                backgroundColor: '#e0e0e0' // Optional: Hover effect
                                            },
                                            borderRadius: '50%', // Optional: Keeps the background circular
                                            width: '40px', // Increase clickable area width
                                            height: '40px', // Increase clickable area height
                                            display: 'flex', // Center the icon within the button
                                            justifyContent: 'center',
                                            alignItems: 'center'
                                        }}
                                    >
                                        <EditIcon sx={{ color: '#14446e', fontSize: '24px' }} />{' '}
                                        {/* Keep the icon size fixed */}
                                    </IconButton>
                                </ListItemButton>
                            </ListItem>
                            <ListItem disablePadding>
                                <ListItemButton
                                    selected={selectedSecondaryMenu === 3}
                                    sx={selectedStyle}
                                    onClick={(event) => handleSecondaryMenuClick(event, 3)}
                                >
                                    {runningProcesses['refine'] === true ? (
                                        <CircularProgress size={20} color="inherit" />
                                    ) : null}
                                    <ListItemText primary="Refine" />
                                    <IconButton
                                        onClick={() => setRefineDrawerOpen(true)}
                                        sx={{
                                            backgroundColor: '#eaf1fb', // Set the background color
                                            '&:hover': {
                                                backgroundColor: '#e0e0e0' // Optional: Hover effect
                                            },
                                            borderRadius: '50%', // Optional: Keeps the background circular
                                            width: '40px', // Increase clickable area width
                                            height: '40px', // Increase clickable area height
                                            display: 'flex', // Center the icon within the button
                                            justifyContent: 'center',
                                            alignItems: 'center'
                                        }}
                                    >
                                        <EditIcon sx={{ color: '#14446e', fontSize: '24px' }} />{' '}
                                        {/* Keep the icon size fixed */}
                                    </IconButton>
                                </ListItemButton>
                            </ListItem>
                            <ListItem
                                disablePadding
                                sx={{
                                    '&:hover': { backgroundColor: '#eaebef' }
                                }}
                            >
                                <ListItemButton
                                    selected={selectedSecondaryMenu === 4}
                                    sx={selectedStyle}
                                    onClick={(event) => handleSecondaryMenuClick(event, 4)}
                                >
                                    {runningProcesses['predict'] === true ? (
                                        <CircularProgress size={20} color="inherit" />
                                    ) : null}
                                    <ListItemText primary="Predict" />
                                    <IconButton
                                        onClick={() => setPredictDrawerOpen(true)}
                                        sx={{
                                            backgroundColor: '#eaf1fb', // Set the background color
                                            '&:hover': {
                                                backgroundColor: '#e0e0e0' // Optional: Hover effect
                                            },
                                            borderRadius: '50%', // Optional: Keeps the background circular
                                            width: '40px', // Increase clickable area width
                                            height: '40px', // Increase clickable area height
                                            display: 'flex', // Center the icon within the button
                                            justifyContent: 'center',
                                            alignItems: 'center'
                                        }}
                                    >
                                        <EditIcon sx={{ color: '#14446e', fontSize: '24px' }} />{' '}
                                        {/* Keep the icon size fixed */}
                                    </IconButton>
                                </ListItemButton>
                            </ListItem>
                            <ListItem disablePadding>
                                <ListItemButton
                                    selected={selectedSecondaryMenu === 5}
                                    sx={selectedStyle}
                                    onClick={(event) => handleSecondaryMenuClick(event, 5)}
                                >
                                    <ListItemText primary="Postprocess" />
                                </ListItemButton>
                            </ListItem>

                            <ListItem
                                disablePadding
                                // sx={{
                                //     '&:hover': { backgroundColor: '#eaebef' }
                                // }}
                            >
                                <ListItemButton
                                    selected={selectedSecondaryMenu === 6} // Unique index for the new item
                                    sx={{ ...selectedStyle, bottom: 0, position: 'static' }}
                                    onClick={(event) => handleSecondaryMenuClick(event, 6)} // Ensure the click handler works for the new item
                                >
                                    <ListItemText primary="Jobs viewer" />
                                </ListItemButton>
                            </ListItem>
                        </List>
                    )}
                    {selectedPrimaryMenu === 1 && (
                        <List>
                            <ListItem disablePadding>
                                <ListItemButton
                                    selected={selectedSecondaryMenu === 0}
                                    sx={selectedStyle}
                                    onClick={(event) => handleSecondaryMenuClick(event, 0)}
                                >
                                    <ListItemText primary="Page 1" />
                                </ListItemButton>
                            </ListItem>
                            <ListItem disablePadding>
                                <ListItemButton
                                    selected={selectedSecondaryMenu === 1}
                                    sx={selectedStyle}
                                    onClick={(event) => handleSecondaryMenuClick(event, 1)}
                                >
                                    <ListItemText primary="Page 2" />
                                </ListItemButton>
                            </ListItem>
                        </List>
                    )}
                </div>
                <div className="content-area">
                    {CurrentComponent !== null ? (
                        (() => {
                            // Determine props to pass based on the current component
                            let componentProps = { starName, setStarName }
                            if (CurrentComponent === PagePrepare) {
                                componentProps = {
                                    ...componentProps,
                                    prepareMessages,
                                    setPrepareMessages
                                }
                            } else if (CurrentComponent === PageRefine) {
                                componentProps = {
                                    ...componentProps,
                                    refineMessages,
                                    setRefineMessages
                                }
                            } else if (CurrentComponent === PagePredict) {
                                componentProps = {
                                    ...componentProps,
                                    predictMessages,
                                    setPredictMessages
                                }
                            } else if (CurrentComponent === PageMask) {
                                componentProps = {
                                    ...componentProps,
                                    maskMessages,
                                    setMaskMessages
                                }
                            } else if (CurrentComponent === PageDeconv) {
                                componentProps = {
                                    ...componentProps,
                                    deconvMessages,
                                    setDeconvMessages
                                }
                            }

                            return <CurrentComponent {...componentProps} />
                        })()
                    ) : (
                        <iframe
                            src="https://isonetcryoet.com/" // Replace with your desired URL
                            style={{
                                width: '100%',
                                height: '100vh',
                                border: 'none'
                            }}
                            title="Embedded Webpage"
                        />
                    )}
                </div>

                <DrawerPrepare
                    open={prepareDrawerOpen}
                    onClose={() => setPrepareDrawerOpen(false)}
                    onSubmit={handleSubmitPrepare}
                />
                <DrawerRefine
                    open={refineDrawerOpen}
                    onClose={() => setRefineDrawerOpen(false)}
                    onSubmit={handleSubmitRefine}
                />
                <DrawerDeconv
                    open={deconvDawerOpen}
                    onClose={() => setDeconvDrawerOpen(false)}
                    onSubmit={handleSubmitDeconv}
                />
                <DrawerMask
                    open={maskDrawerOpen}
                    onClose={() => setMaskDrawerOpen(false)}
                    onSubmit={handleSubmitMask}
                />
                <DrawerPredict
                    open={predictDrawerOpen}
                    onClose={() => setPredictDrawerOpen(false)}
                    onSubmit={handleSubmitPredict}
                />
            </div>
        </div>
    )
}

export default App
