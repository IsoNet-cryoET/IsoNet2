import './index.css'
import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react'
import PageCommon from '../components/PageCommon'
import theme from '../theme.js'
import { mergeMsg, processMessage } from '../utils/utils.js'
import {
    List,
    ListItem,
    ListItemText,
    IconButton,
    ListItemButton,
    ThemeProvider,
    CircularProgress,
    Dialog,
    DialogTitle,
    DialogContent,
    Typography,
    DialogActions,
    Button,
    Box,
    Tooltip
} from '@mui/material'
import { keyframes } from '@mui/system'
import { alpha } from '@mui/material/styles'
import EditIcon from '@mui/icons-material/Edit'
import Backdrop from '@mui/material/Backdrop'
import { primaryMenuListinOrder, primaryMenuMapping } from './menuMapping.js'
import { useSelector, useDispatch } from 'react-redux'
import {
    fetchJobs,
    updateJobStatusAsync,
    updateJobPIDAsync,
    addJobAsync
} from '../store/jobSlice'

const App = () => {
    const [blocking, setBlocking] = useState(false)
    const [selectedDrawer, setSelectedDrawer] = useState('')
    const [selectedJob, setSelectedJob] = useState(null)
    const [messages, setMessages] = useState([])
    const [starName, setStarName] = useState('')
    const [selectedPrimaryMenu, setSelectedPrimaryMenu] = useState('prepare_star')
    const [confirmOpen, setConfirmOpen] = useState(false)
    const inflight = useRef(0)
    const intervalRef = useRef(null)

    const dispatch = useDispatch()

    // Load jobList from main process into Redux at startup
    useEffect(() => {
        dispatch(fetchJobs())
    }, [dispatch])

    const jobs = useSelector((state) => state.jobs.jobList)

    // Jobs for the currently selected primary menu (filtered)
    const visibleJobs = useMemo(
        () => jobs.filter((j) => j.type === selectedPrimaryMenu),
        [jobs, selectedPrimaryMenu]
    )

    // Listen to python runtime status updates and update job status/PID accordingly
    useEffect(() => {
        const off = window.api.on('python-status-change', ({ id, status, pid }) => {
            if (id > 0) {
                dispatch(updateJobStatusAsync({ id, status }))
                dispatch(updateJobPIDAsync({ id, pid }))
            }
        })
        return () => {
            try {
                off?.()
            } catch { }
        }
    }, [dispatch])

    // // When primary menu changes OR Redux job list changes:
    // // For PageCommon pages, automatically select the first job of that type
    // useEffect(() => {
    //     const mapping = primaryMenuMapping[selectedPrimaryMenu]
    //     if (!mapping || mapping.page !== PageCommon) return

    //     const first = jobs.find((o) => o.type === selectedPrimaryMenu)
    //     setSelectedJob(first || null)
    // }, [selectedPrimaryMenu, jobs])

    // Poll log file of the currently selected job
    useEffect(() => {
        if (!selectedJob) return
        if (selectedJob.status === 'inqueue') return

        const logPath = `${selectedJob.output_dir}/log.txt`
        let alive = true

        const interval = setInterval(() => {
            console.log('Polling logs...')
            window.api.call('readFile', logPath).then((fileContent) => {
                if (!alive) return
                if (!fileContent) {
                    setMessages([])
                    return
                }

                const lines = fileContent.split(/\r?\n|\r/g).filter(Boolean)
                const tmp = []
                for (const line of lines) {
                    const msg = { cmd: selectedJob.type, output: line }
                    const processed = processMessage(msg)
                    const merged = mergeMsg(tmp, processed)
                    tmp.length = 0
                    tmp.push(...merged)
                }
                setMessages(tmp)
            })
        }, 300)

        return () => {
            alive = false
            clearInterval(interval)
        }
    }, [selectedJob])

    // When main process sends app-close-request, open confirmation dialog
    useEffect(() => {
        const off = window.api.on('app-close-request', () => setConfirmOpen(true))
        return () => off?.()
    }, [])

    // Utility to show blocking UI when async tasks are running
    const withBlocking = async (fn) => {
        if (inflight.current === 0) setBlocking(true)
        inflight.current++
        try {
            await fn()
        } finally {
            inflight.current--
            if (inflight.current === 0) setBlocking(false)
        }
    }

    // Kill running jobs, mark queued jobs as completed, then close app
    const doCleanupThenClose = () =>
        withBlocking(async () => {
            setConfirmOpen(false)

            const running = jobs.filter((j) => j.status === 'running')
            await Promise.allSettled(
                running.map((j) => {
                    return (async () => {
                        await window.api.call('killJob', j.pid)
                        await dispatch(updateJobStatusAsync({ id: j.id, status: 'completed' }))
                    })()
                })
            )

            const queued = jobs.filter((j) => j.status === 'inqueue')
            await Promise.allSettled(
                queued.map((j) =>
                    dispatch(updateJobStatusAsync({ id: j.id, status: 'completed' }))
                )
            )

            await window.api.call('appClose', true)
        })

    const cancelClose = async () => {
        setConfirmOpen(false)
        await window.api.call('appClose', false)
    }

    // Submit job forms, create job, start backend, handle prepare_star logs
    const handleSubmit = useCallback(
        async (type, data) => {
            try {
                const id = await window.api.call('nextId')

                // Do not mutate input data
                const withOutputDir =
                    type !== 'prepare_star'
                        ? { ...data, output_dir: `${type}/job${id}` }
                        : { ...data }

                // Ensure "type" exists so visibleJobs works properly
                const payload = { ...withOutputDir, id, type }

                // Add job to Redux + save to settings.json
                dispatch(addJobAsync(payload))

                // Run associated backend Python process
                window.api.call('run', payload)

                // Special log polling for prepare_star
                if (type === 'prepare_star' && withOutputDir?.star_name) {
                    setStarName(withOutputDir.star_name)

                    if (intervalRef.current) {
                        clearInterval(intervalRef.current)
                        intervalRef.current = null
                    }

                    const logPath = 'prepare_log.txt'
                    intervalRef.current = setInterval(async () => {
                        const exists = await window.api.call('isFileExist', logPath)
                        if (!exists) {
                            setMessages([])
                            return
                        }

                        const fileContent = await window.api.call('readFile', logPath)
                        if (!fileContent) {
                            setMessages([])
                            return
                        }

                        const lines = fileContent.split(/\r?\n|\r/g).filter(Boolean)
                        const tmp = []

                        for (const line of lines) {
                            const msg = { cmd: 'prepare_star', output: line }
                            const processed = processMessage(msg)
                            const merged = mergeMsg(tmp, processed)
                            tmp.length = 0
                            tmp.push(...merged)

                            if (line.toLowerCase().includes('exit')) {
                                clearInterval(intervalRef.current)
                                intervalRef.current = null
                                break
                            }
                        }
                        setMessages(tmp)
                    }, 300)
                }
            } catch (error) {
                console.error(`Error submitting ${type} form:`, error)
            } finally {
                setSelectedDrawer('')
            }
        },
        [dispatch]
    )

    const DrawerComponent = primaryMenuMapping[selectedPrimaryMenu]?.drawer
    const PageComponent = primaryMenuMapping[selectedPrimaryMenu]?.page

    const togglePrimaryMenu = (key) => {
        setSelectedPrimaryMenu(key)
        setMessages([])

        const mapping = primaryMenuMapping[key]
        if (!mapping || mapping.page !== PageCommon) {
            // For non-PageCommon pages, clear selected job
            setSelectedJob(null)
            return
        }

        // Only when switching primary menu, pick the first job of that type
        const first = jobs.find((j) => j.type === key)
        setSelectedJob(first || null)
    }

    const innerGlowPulse = keyframes`
        0%, 100% {
            opacity: .25;
            box-shadow:
                inset 0 0 0 1px var(--ring-weak);
        }
        50% {
            opacity: 1;
            box-shadow:
                inset 0 0 14px var(--ring-strong),
                inset 0 0 0 2px var(--ring-strong);
        }
    `

    return (
        <ThemeProvider theme={theme}>
            {/* Full-screen blocking overlay */}
            {blocking && (
                <Backdrop
                    open={blocking}
                    sx={{
                        color: '#fff',
                        zIndex: (theme) => theme.zIndex.modal + 2,
                        bgcolor: 'rgba(0,0,0,0.35)'
                    }}
                >
                    <CircularProgress />
                </Backdrop>
            )}

            {/* Confirm quit dialog */}
            <Dialog open={confirmOpen} onClose={cancelClose}>
                <DialogTitle>Close and clean up?</DialogTitle>
                <DialogContent>
                    <Typography variant="body2">
                        This will kill all running jobs and remove all queued jobs. Proceed?
                    </Typography>
                </DialogContent>
                <DialogActions>
                    <Button onClick={cancelClose}>No</Button>
                    <Button
                        onClick={doCleanupThenClose}
                        autoFocus
                        variant="contained"
                        color="error"
                    >
                        Yes, close and clean up
                    </Button>
                </DialogActions>
            </Dialog>

            <div className="outer-container">
                <div className="main-content">
                    {/* Left primary menu */}
                    <div className="primary-menu">
                        <List>
                            {primaryMenuListinOrder.map((key) => (
                                <ListItem disablePadding key={key}>
                                    <ListItemButton
                                        selected={selectedPrimaryMenu === key}
                                        onClick={() => togglePrimaryMenu(key)}
                                    >
                                        {React.createElement(primaryMenuMapping[key].icon, {
                                            sx: { mr: 1, color: 'primary.main', fontSize: 16 }
                                        })}
                                        <ListItemText primary={primaryMenuMapping[key]?.label} />
                                        {primaryMenuMapping[key]?.drawer && (
                                            <IconButton
                                                onClick={() => setSelectedDrawer(key)}
                                            >
                                                <EditIcon
                                                    sx={{ fontSize: 16, color: 'primary.main' }}
                                                />
                                            </IconButton>
                                        )}
                                    </ListItemButton>
                                </ListItem>
                            ))}
                        </List>
                    </div>

                    {/* Right secondary menu: job list */}
                    {['denoise', 'deconv', 'make_mask', 'refine', 'predict'].includes(
                        selectedPrimaryMenu
                    ) &&
                        visibleJobs.length > 0 && (
                            <Box
                                className="secondary-menu"
                                sx={{
                                    height: 'calc(100vh)',
                                    overflowY: 'auto'
                                }}
                            >
                                <List>
                                    {visibleJobs.map((job) => (
                                        <ListItem key={job.id}>
                                            <ListItemButton
                                                selected={selectedJob?.id === job.id}
                                                onClick={() => setSelectedJob(job)}
                                                sx={{
                                                    position: 'relative',
                                                    '--ring-strong': (t) =>
                                                        alpha(t.palette.primary.main, 0.55),
                                                    '--ring-weak': (t) =>
                                                        alpha(t.palette.primary.main, 0.22),
                                                    ...(job.status === 'running' && {
                                                        '&::after': {
                                                            content: '""',
                                                            position: 'absolute',
                                                            inset: 0,
                                                            borderRadius: '30px',
                                                            pointerEvents: 'none',
                                                            animation: `${innerGlowPulse} 1.8s ease-in-out infinite`
                                                        },
                                                        '@media (prefers-reduced-motion: reduce)': {
                                                            '&::after': { animation: 'none' }
                                                        }
                                                    })
                                                }}
                                            >
                                                <Tooltip
                                                    title={`job ${job.id} ${job.name}`}
                                                    arrow
                                                    placement="right"
                                                >
                                                    <Typography className="secondary-menu-text">
                                                        {job.name}
                                                    </Typography>
                                                </Tooltip>
                                            </ListItemButton>
                                        </ListItem>
                                    ))}
                                </List>
                            </Box>
                        )}

                    {/* Main content area */}
                    {PageComponent && (
                        <div className="content-area">
                            <PageComponent
                                starName={starName}
                                setStarName={setStarName}
                                messages={messages || []}
                                setMessages={setMessages}
                                selectedJob={selectedJob}
                                setBlocking={setBlocking}
                            />
                        </div>
                    )}

                    {/* Drawer forms */}
                    {DrawerComponent && selectedDrawer === selectedPrimaryMenu && (
                        <DrawerComponent
                            key={selectedDrawer}
                            open={true}
                            onClose={() => setSelectedDrawer('')}
                            onSubmit={(data) => handleSubmit(selectedDrawer, data)}
                        />
                    )}
                </div>
            </div>
        </ThemeProvider>
    )
}

export default App
