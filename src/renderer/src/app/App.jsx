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

const App = () => {
    const [blocking, setBlocking] = useState(false)
    const [selectedDrawer, setSelectedDrawer] = useState('')
    const [jobs, setJobs] = useState([])
    const [selectedJob, setSelectedJob] = useState(null)
    const [messages, setMessages] = useState([])
    const [starName, setStarName] = useState('')
    const [selectedPrimaryMenu, setSelectedPrimaryMenu] = useState('prepare_star')
    const [confirmOpen, setConfirmOpen] = useState(false)
    const inflight = useRef(0)

    const visibleJobs = useMemo(
        () => jobs.filter((j) => j.type === selectedPrimaryMenu),
        [jobs, selectedPrimaryMenu]
    )

    // todo : delete or not
    useEffect(() => {
        const off = window.api.on('python-status-change', ({ id, status, pid }) => {
            if (id > 0) {
                window.api.call('updateJobStatus', id, status)
                window.api.call('updateJobPID', id, pid)
            }
        })
        return () => {
            try {
                off?.()
            } catch { }
        }
    }, []) // <-- empty deps: attach once

    useEffect(() => {
        if (primaryMenuMapping[selectedPrimaryMenu].page !== PageCommon) return
        window.api.call('getJobList').then((list) => {
            const first = list.find(o => o.type === selectedPrimaryMenu)
            setSelectedJob(first)
        })
        const interval = setInterval(() => {
            console.log("interval get jobs")
            window.api.call('getJobList').then((list) => {
                setJobs(() => list)
            })
        }, 500)
        return () => {
            clearInterval(interval)
        }
    }, [selectedPrimaryMenu])

    useEffect(() => {
        if (!selectedJob) return
        if (selectedJob.status === 'inqueue') return

        const logPath = `${selectedJob.output_dir}/log.txt`
        let alive = true

        const interval = setInterval(() => {
            console.log("interval get logs")
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
    }, [selectedJob, jobs])

    useEffect(() => {
        const off = window.api.on('app-close-request', () => setConfirmOpen(true))
        return () => off?.()
    }, [])

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

    const doCleanupThenClose = () =>
        withBlocking(async () => {
            setConfirmOpen(false)

            const all_jobs = await window.api.call('getJobList')

            const running = all_jobs.filter((j) => j.status === 'running')

            await Promise.allSettled(
                running.map((j) => {
                    window.api.call('killJob', j.pid)
                    window.api.call('updateJobStatus', j.id, 'completed')
                })
            )

            const queued = all_jobs.filter((j) => j.status === 'inqueue')
            await Promise.allSettled(
                queued.map((j) => window.api.call('updateJobStatus', j.id, 'completed'))
            )

            // 4) Tell main we’re safe to close
            await window.api.call('appClose', true)
        })

    const cancelClose = async () => {
        setConfirmOpen(false)
        await window.api.call('appClose', false)
    }

    const intervalRef = useRef(null)
    const handleSubmit = useCallback(async (type, data) => {
        try {
            const id = await window.api.call('nextId');
            // avoid mutating incoming data
            const withOutputDir =
                type !== "prepare_star"
                    ? { ...data, output_dir: `${type}/job${id}` }
                    : { ...data };
            const payload = { ...withOutputDir, id };
            await window.api.call('addJob', payload);
            window.api.call('run', payload);
            if (type === "prepare_star" && withOutputDir?.star_name) {
                setStarName(withOutputDir.star_name);
                // clear any previous polling interval
                if (intervalRef.current) {
                    clearInterval(intervalRef.current);
                    intervalRef.current = null;
                }
                const logPath = "prepare_log.txt";
                intervalRef.current = setInterval(async () => {
                    // 1) check existence first
                    const exists = await window.api.call('isFileExist', logPath);
                    if (!exists) {
                        setMessages([]);
                        return;
                    }

                    // 2) read content
                    const fileContent = await window.api.call('readFile', logPath);
                    if (!fileContent) {
                        setMessages([]);
                        return;
                    }

                    // 3) process lines
                    const lines = fileContent.split(/\r?\n|\r/g).filter(Boolean);
                    const tmp = [];

                    for (const line of lines) {
                        const msg = { cmd: "prepare_star", output: line };
                        const processed = processMessage(msg);
                        const merged = mergeMsg(tmp, processed);
                        tmp.length = 0;
                        tmp.push(...merged);

                        // stop condition
                        if (line.toLowerCase().includes("exit")) {
                            clearInterval(intervalRef.current);
                            intervalRef.current = null;
                            break;
                        }
                    }
                    setMessages(tmp);
                }, 300);
            }
        } catch (error) {
            console.error(`Error submitting ${type} form:`, error);
        } finally {
            setSelectedDrawer("");
        }
    }, [mergeMsg, processMessage, setSelectedDrawer, setStarName]);

    const DrawerComponent = primaryMenuMapping[selectedPrimaryMenu]?.drawer
    const PageComponent = primaryMenuMapping[selectedPrimaryMenu]?.page
    const togglePrimaryMenu = (key) => {
        setSelectedPrimaryMenu(key)
        setMessages([])
        setSelectedJob(null)
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
                    <div className="primary-menu">
                        <List>
                            {primaryMenuListinOrder.map((key) => (
                                <ListItem disablePadding key={key}>
                                    <ListItemButton
                                        selected={selectedPrimaryMenu === key}
                                        onClick={() => togglePrimaryMenu(key)}
                                    >
                                        {React.createElement(primaryMenuMapping[key].icon,
                                            { sx: { mr: 1, color: 'primary.main', fontSize: 16 } })}
                                        <ListItemText primary={primaryMenuMapping[key]?.label} />
                                        {primaryMenuMapping[key]?.drawer && (
                                            <IconButton
                                                onClick={() => {
                                                    // e.stopPropagation();
                                                    setSelectedDrawer(key)
                                                }}
                                            >
                                                <EditIcon sx={{ fontSize: 16, color: 'primary.main' }} />
                                            </IconButton>
                                        )}
                                    </ListItemButton>
                                </ListItem>
                            ))}
                        </List>
                    </div>

                    {['denoise', 'deconv', 'make_mask', 'refine', 'predict'].includes(
                        selectedPrimaryMenu
                    ) && visibleJobs.length > 0 && (
                            <Box
                                className="secondary-menu"
                                sx={{
                                    height: 'calc(100vh)',
                                    overflowY: 'auto'
                                }}
                            >
                                <List>
                                    {visibleJobs.map((job) => (
                                        <ListItem>
                                            <ListItemButton
                                                key={job.id}
                                                selected={selectedJob?.id === job.id}
                                                onClick={() => setSelectedJob(job)}
                                                // divider
                                                sx={{
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
                                                <Tooltip title={`job ${job.id} ${job.name}`} arrow placement='right'>
                                                    <Typography
                                                        className="secondary-menu-text"
                                                    >{`${job.name}`}</Typography>
                                                </Tooltip>
                                            </ListItemButton>
                                        </ListItem>
                                    ))}
                                </List>
                            </Box>
                        )}

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
