import React, { useState, useEffect, useCallback,useMemo,useRef } from 'react'
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
    Tooltip,
} from '@mui/material'
import { keyframes } from '@mui/system';
import { alpha } from '@mui/material/styles';
import EditIcon from '@mui/icons-material/Edit'
import Backdrop from '@mui/material/Backdrop'
import DrawerPrepare from './components/DrawerPrepare'
import DrawerRefine from './components/DrawerRefine'
import DrawerDenoise from './components/DrawerDenoise'
import DrawerPredict from './components/DrawerPredict'
import DrawerDeconv from './components/DrawerDeconv'
import DrawerMask from './components/DrawerMask'

import PagePrepare from './components/PagePrepare'
import PageCommon from './components/PageCommon'
import PageJobs from './components/PageJobs'

import theme from './theme';
import JobsList from './backup/SecondaryMenu'
import { mergeMsg, processMessage } from './utils/utils'
import toCommand from './utils/handle_json'
import { ConstructionOutlined } from '@mui/icons-material'
// import {BlockingProvider} from './BlockingContext'


const primaryMenuListinOrder = [
    'prepare_star','denoise','deconv','make_mask','refine','predict','jobs_viewer'
]
const primaryMenuMapping = {
    prepare_star: {
        label: 'Prepare',
        drawer: DrawerPrepare,
        page: PagePrepare
    },
    denoise: {
        label: 'Denoise',
        drawer: DrawerDenoise,
        page: PageCommon
    },
    deconv: {
        label: 'Deconvolve',
        drawer: DrawerDeconv,
        page: PageCommon
    },
    make_mask: {
        label: 'Create Mask',
        drawer: DrawerMask,
        page: PageCommon
    },
    refine: {
        label: 'Refine',
        drawer: DrawerRefine,
        page: PageCommon
    },
    predict: {
        label: 'Predict',
        drawer: DrawerPredict,
        page: PageCommon
    },
    jobs_viewer: {
        label: 'Jobs Viewer',
        page: PageJobs
    }
}

const App = () => {
    const [blocking, setBlocking] = useState(false)
    const [selectedDrawer, setSelectedDrawer] = useState('')
    const [jobs, setJobs] = useState([])
    const [selectedJob, setSelectedJob] = useState(null)
    const [messages, setMessages] = useState([])
    const [starName, setStarName] = useState('')
    const [selectedPrimaryMenu, setSelectedPrimaryMenu] = useState('prepare_star')
    const [confirmOpen, setConfirmOpen] = useState(false);
    const inflight = useRef(0);

    const visibleJobs = useMemo(
        () => jobs.filter((j) => j.type === selectedPrimaryMenu),
        [jobs, selectedPrimaryMenu]
    )

    useEffect(() => {
        const off = window.api.onPythonUpdateStatus(({ id, status, pid }) => {
            console.log('python-status-changed:', id, status)
            window.jobList.updateStatus({ id, status })  
            window.jobList.updatePID({ id, pid })  
        })
        return () => { try { off?.() } catch {} } 
    }, []) // <-- empty deps: attach once

    useEffect(() => {
        const interval = setInterval(() => {
            window.jobList.get().then(list => {
                setJobs(()=>list)
              })
        }, 500)
        return () => {clearInterval(interval) }
    }, [])

    useEffect(() => {

        if (!selectedJob) return
        const output_dir = `${selectedJob.type}/job${selectedJob.id}_${String(selectedJob.name).replace(/\s+/g, '_')}`

        const logPath = `${output_dir}/log.txt`
        let alive = true

        const interval = setInterval(() => {
        window.api.readFile(logPath).then((fileContent) => {
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

        return () => { alive = false; clearInterval(interval) }
    }, [selectedJob, jobs])

    useEffect(() => {
        if (selectedPrimaryMenu !== 'prepare_star') { setMessages([]); return }
      
        const logPath = 'prepare_log.txt' // 确保主进程按预期解析
        let alive = true
        const interval = setInterval(() => {
            window.api.readFile(logPath).then((fileContent) => {
                if (!alive) return
                if (!fileContent) {
                setMessages([])
                return
                }
    
                const lines = fileContent.split(/\r?\n|\r/g).filter(Boolean)
                const tmp = []
                for (const line of lines) {
                const msg = { cmd: "prepare_star", output: line }
                const processed = processMessage(msg)
                const merged = mergeMsg(tmp, processed)
                tmp.length = 0
                tmp.push(...merged)
                }
                setMessages(tmp)
            })
            }, 300)
      
        return () => { alive = false; clearInterval(interval) }
      }, [selectedPrimaryMenu])
      
    useEffect(() => {
        const off = window.appClose?.onRequest?.(() => setConfirmOpen(true));
        return () => { try { off?.(); } catch {}
        };
    }, []);


  
    const withBlocking = async (fn) => {
      if (inflight.current === 0) setBlocking(true);
      inflight.current++;
      try { await fn(); }
      finally {
        inflight.current--;
        if (inflight.current === 0) setBlocking(false);
      }
    };

    const doCleanupThenClose = () =>
        withBlocking(async () => {
            setConfirmOpen(false);
        
            const all_jobs = await window.jobList.get()

            const running = all_jobs.filter(j => j.status === 'running');
            console.log(running)

            await Promise.allSettled(
                running.map(j => {
                    api.killJob(j.pid)
                    console.log(j)
                    window.jobList.updateStatus({ id: j.id, status: 'completed' })
                })
            )

            const queued = all_jobs.filter(j => j.status === 'inqueue');
            await Promise.allSettled(
                queued.map(j => window.jobList.updateStatus({ id: j.id, status: 'completed' }))
            )
      
          // 4) Tell main we’re safe to close
          await window.appClose.reply(true);
        });

    const cancelClose = async () => {
        setConfirmOpen(false);
        await window.appClose.reply(false);
    };

    const handleSubmit = useCallback(async (type, data) => {
        console.log("submit in app.jsx")
        console.log(type)
        console.log(data)
        try {
            const id = await window.count.next()
            if (data.type !== 'prepare_star'){
                data.output_dir = `${data.type}/job${id}_${String(data.name).replace(/\s+/g, '_')}`
            }
            const payload = {
                ...data,
                id,
              }              
            // setJobs((prev) => {prev, payload})
            await window.jobList.add(payload)
            window.api.run(payload)

            if (type === 'prepare_star' && data?.star_name) {
                setStarName(data.star_name)
            }
        } catch (error) {
                console.error(`Error submitting ${type} form:`, error)
        }finally {
            setSelectedDrawer('')
        }
    },
    []
    )


    const DrawerComponent = primaryMenuMapping[selectedPrimaryMenu]?.drawer;
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
    `;
    return (
        <ThemeProvider theme={theme}>
            {blocking && <Backdrop
                open={blocking}
                sx={{
                color: '#fff',
                zIndex: (theme) => theme.zIndex.modal + 2, // 确保足够高
                bgcolor: 'rgba(0,0,0,0.35)',
                }}
            >
                <CircularProgress />
            </Backdrop>}

            <Dialog open={confirmOpen} onClose={cancelClose}>
                <DialogTitle>Close and clean up?</DialogTitle>
                <DialogContent>
                <Typography variant="body2">
                    This will kill all running jobs and remove all queued jobs. Proceed?
                </Typography>
                </DialogContent>
                <DialogActions>
                <Button onClick={cancelClose}>No</Button>
                <Button onClick={doCleanupThenClose} autoFocus variant="contained" color="error">
                    Yes, close and clean up
                </Button>
                </DialogActions>
            </Dialog>
            <div className="outer-container">
                {/* customerized top bar -- delete temperally */}
                {/* <div className="top-bar">IsoNet 2</div> */}
                <div className="main-content">
                    <div className="primary-menu">
                        <List>
                            {primaryMenuListinOrder.map((key) => (
                                <ListItem
                                    disablePadding
                                    key={key}
                                >
                                    <ListItemButton
                                        selected={selectedPrimaryMenu === key}
                                        onClick={() => togglePrimaryMenu(key)}
                                    >
                                        <ListItemText primary={primaryMenuMapping[key]?.label} />
                                        {primaryMenuMapping[key]?.drawer &&
                                          (
                                            <IconButton
                                                onClick={() => {
                                                    // e.stopPropagation();
                                                    setSelectedDrawer(key);
                                                }}
                                                sx={{
                                                    backgroundColor:'#D5E2F4',
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

                    {['denoise','deconv','make_mask','refine','predict'].includes(selectedPrimaryMenu) && (
                        <Box 
                            className="secondary-menu"
                            sx={{
                                height: 'calc(100vh)',
                                overflowY: 'auto',
                            }}                            
                            >
                            <List>
                                {visibleJobs.map((job) => (
                                    <ListItem                                >
                                    <ListItemButton
                                        key={job.id}
                                        selected={selectedJob?.id === job.id}
                                        onClick={() => setSelectedJob(job)}
                                        // divider
                                        sx={{
                                            '--ring-strong': (t) => alpha(t.palette.primary.main, 0.55),
                                            '--ring-weak':   (t) => alpha(t.palette.primary.main, 0.22),
                                            ...(job.status === 'running' && {
                                              '&::after': {
                                                content: '""',
                                                position: 'absolute',
                                                inset: 0,       
                                                borderRadius: 1,
                                                pointerEvents: 'none',
                                                animation: `${innerGlowPulse} 1.8s ease-in-out infinite`,
                                              },
                                              '@media (prefers-reduced-motion: reduce)': {
                                                '&::after': { animation: 'none' },
                                              },
                                            }),
                                          }}
                                    >
                                        <Tooltip title={`job ${job.id} ${job.name}`} arrow>
                                            <Typography className="secondary-menu-text"
                                                // variant="body2"
                                            >{`Job ${job.id}`}</Typography>
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
                    onClose={() => setSelectedDrawer("")}
                    onSubmit={(data) => handleSubmit(selectedDrawer, data)}
                />
                )}
                </div>
            </div>          
        </ThemeProvider>
    )
}

export default App
