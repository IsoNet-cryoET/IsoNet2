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
import PagePrepare from './components/PagePrepare'
import PageCommon from './components/PageCommon'

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

    //message is from the python stdout and stderr
    // we should clear message once the job is finished
    const [messages, setMessages] = useState([])

    //the list of all jobs
    const [jobs, setJobs] = useState([])

    // selectedJobs
    const [selectedJob, setSelectedJob] = useState(null)
    const [starName, setStarName] = useState('')
    const [selectedPrimaryMenu, setSelectedPrimaryMenu] = useState(0)

    const toggleDrawer = useCallback((drawer, open) => {
        setDrawerState((prev) => ({ ...prev, [drawer]: open }))
    }, [])

    const handleSubmit = useCallback(
        (type, data) => {
            try {
                const { inqueue, ...restData } = data
                const status = inqueue === true ? 'queued' : 'running'

                const newJob = {
                    id: jobs.length + 1,
                    type,
                    data: restData, // cleansed data
                    output_dir: data.output_dir,
                    status
                }

                setJobs((prev) => [...prev, newJob])
                if (type === 'prepare_star') setStarName(data.star_name)

                window.api.run(newJob) // Ensure this triggers ipcMain.handle
            } catch (error) {
                console.error(`Error submitting ${type} form:`, error)
            }

            toggleDrawer(type, false)
        },
        [toggleDrawer, jobs]
    )

    useEffect(() => {
        window.api.onJobStatusUpdated((data) => {
            console.log('Job status updated:', data)
            setJobs((prevJobs) =>
                prevJobs.map((job) => (job.id === data.id ? { ...job, status: data.status } : job))
            )
            console.log(jobs)
        })
    }, [])

    useEffect(() => {
        if (!selectedJob?.output_dir) return

        const logPath = `${selectedJob.output_dir}/log.txt`

        let isMounted = true

        const interval = setInterval(() => {
            window.api.readFile(logPath).then((fileContent) => {
                if (!isMounted) return

                if (!fileContent) {
                    setMessages([]) // Clear messages if log.txt is missing
                    return
                }

                const lines = fileContent.split(/\r?\n|\r/g).filter(Boolean)
                const newMessages = []

                for (const line of lines) {
                    const msg = { cmd: selectedJob.type, output: line }
                    const processed = processMessage(msg)
                    const merged = mergeMsg(newMessages, processed)
                    newMessages.length = 0
                    newMessages.push(...merged)
                }

                setMessages(newMessages)
            })
        }, 100) // poll every 100 ms

        return () => {
            isMounted = false
            clearInterval(interval)
        }
    }, [selectedJob])

    const primaryMenus = useMemo(
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
    let PageComponent = null

    if (selectedPrimaryMenu === 0) {
        PageComponent = PagePrepare
    } else if (selectedPrimaryMenu >= 1 && selectedPrimaryMenu <= 6) {
        PageComponent = PageCommon
    } else if (selectedPrimaryMenu === 8) {
        PageComponent = PageJobs
    }
    return (
        <div className="outer-container">
            <div className="top-bar">IsoNet 2</div>
            <div className="main-content">
                <div className="primary-menu">
                    <List>
                        {primaryMenus.map(({ key, label }, index) => (
                            <ListItem
                                disablePadding
                                key={index}
                                sx={{ '&:hover': { backgroundColor: '#eaebef' } }}
                            >
                                <ListItemButton
                                    selected={selectedPrimaryMenu === index}
                                    sx={{
                                        '&.Mui-selected': {
                                            backgroundColor: '#D5E2F4',
                                            borderRadius: '24px'
                                        }
                                    }}
                                    onClick={() => {
                                        setSelectedPrimaryMenu(index)
                                        setMessages([]) // ✅ Clear messages
                                        setSelectedJob(null) // ✅ Clear selected job
                                    }}
                                >
                                    <ListItemText primary={label} />

                                    {index < 7 && (
                                        <IconButton
                                            onClick={() => toggleDrawer(key, true)}
                                            sx={{
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
                </div>
                {['denoise', 'deconv', 'make_mask', 'refine', 'refine_v1', 'predict'].includes(
                    primaryMenus[selectedPrimaryMenu]?.key
                ) && (
                    <div className="secondary-menu">
                        <List>
                            {jobs
                                .filter(
                                    (job) => job.type === primaryMenus[selectedPrimaryMenu]?.key
                                )
                                .map((job) => (
                                    <ListItemButton
                                        key={job.id}
                                        selected={selectedJob?.id === job.id}
                                        onClick={() => setSelectedJob(job)}
                                    >
                                        {job.status === 'running' && <CircularProgress size={16} />}
                                        <ListItemText primary={`[${job.type}] Job ${job.id}`} />
                                    </ListItemButton>
                                ))}
                        </List>
                    </div>
                )}

                {PageComponent && (
                    <div className="content-area">
                        <PageComponent
                            starName={starName}
                            setStarName={setStarName}
                            messages={messages || []}
                            setMessages={setMessages}
                        />
                    </div>
                )}

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
