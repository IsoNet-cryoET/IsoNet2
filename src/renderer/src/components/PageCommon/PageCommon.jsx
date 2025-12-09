import './index.css'
import { useEffect, useRef, useState } from 'react'
import { renderContent } from '../LogHandler/log_handler'
import { Box, Button, Snackbar, TextField, Typography } from '@mui/material'
import CleaningServicesIcon from '@mui/icons-material/CleaningServices'
import { useDispatch } from 'react-redux'
import { removeJobAsync, updateJobNameAsync } from '../../store/jobSlice'
import Placeholder from '../PlaceHolder/Placeholder'

const PageCommon = (props) => {
    const job = props?.selectedJob
    const [isDeleting, setIsDeleting] = useState(false)
    const [isSaving, setIsSaving] = useState(false)
    const [snack, setSnack] = useState({ open: false, message: '' })
    const dispatch = useDispatch()

    // --- NEW CODE START: Setup Ref for auto-scrolling ---
    const messagesEndRef = useRef(null)
    const prevLenRef = useRef(0)
    const prevLastRef = useRef(null)

    const scrollToBottom = () => {
        // You can change behavior to 'auto' for instant scrolling if logs come in very fast
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }

    // Only auto-scroll when content actually changed (avoids scrolling on parent re-renders
    // that recreate the same messages array reference)
    // Only auto-scroll when content actually changed AND the user is already at the bottom.
    // If the user has scrolled up, preserve their position and do not force-scroll.
    useEffect(() => {
        const getScrollContainer = (el) => {
            if (!el) return document.scrollingElement || document.documentElement
            let node = el
            while (node) {
                if (node === document.body || node === document.documentElement) {
                    return document.scrollingElement || document.documentElement
                }
                const s = window.getComputedStyle(node)
                if (s.overflowY === 'auto' || s.overflowY === 'scroll') return node
                node = node.parentElement
            }
            return document.scrollingElement || document.documentElement
        }

        const isScrolledToBottom = (threshold = 50) => {
            const end = messagesEndRef.current
            if (!end) return true
            const container = getScrollContainer(end)
            if (!container) return true
            if (container === document.scrollingElement || container === document.documentElement) {
                const scrollTop = window.scrollY || window.pageYOffset || document.documentElement.scrollTop
                const clientHeight = window.innerHeight
                const scrollHeight = document.documentElement.scrollHeight
                return scrollHeight - (scrollTop + clientHeight) <= threshold
            } else {
                return container.scrollHeight - (container.scrollTop + container.clientHeight) <= threshold
            }
        }

        const msgs = props.messages || []
        const len = msgs.length
        const last = len ? msgs[len - 1] : null

        // only scroll when content changed AND the user is currently at/bottom (visible end)
        if (len !== prevLenRef.current || last !== prevLastRef.current) {
            const wasAtBottom = isScrolledToBottom()
            if (wasAtBottom) scrollToBottom()
            prevLenRef.current = len
            prevLastRef.current = last
        }
    }, [props.messages])
    // --- NEW CODE END ---

    // Local name states, so typing is responsive even before Redux updates
    const [name, setName] = useState(job?.name || '')
    useEffect(() => {
        // When a different job is selected, reset input to that job's name
        setName(job?.name || '')
    }, [job?.id])

    // Placeholder for future debounce logic (currently only used to clear timer)
    const saveTimerRef = useRef(null)

    const handleDelete = async () => {
        if (!job?.id) return
        try {
            setIsDeleting(true)
            // Remove job from Redux + disk
            await dispatch(removeJobAsync(job.id))
            setSnack({ open: true, message: `Deleted job ${job.id}` })
            // NOTE: parent still holds selectedJob state; if you want to clear
            // selection, pass down a callback from parent and call it here.
        } catch (e) {
            console.error('Delete job failed:', e)
            setSnack({ open: true, message: 'Failed to delete job' })
        } finally {
            setIsDeleting(false)
        }
    }

    if (!job) {
        return (
            <Placeholder
                src="figures/no_data.svg"
                title="Nothing here yet"
                subtitle="Create a new job from the left menu to get started"
            />
        )
    }

    return (
        <div>
            <Box
                display="flex"
                alignItems="center"
                gap={2}
                marginY={2}
                sx={{ flexGrow: 1, height: '100%' }}
            >
                {job?.status === 'completed' && (
                    <Button
                        variant="outlined"
                        color="primary"
                        startIcon={<CleaningServicesIcon />}
                        onClick={handleDelete}
                        disabled={!job?.id || isDeleting || job?.status === 'running'}
                        sx={{ height: '56px' }}
                    >
                        permanently remove job files
                    </Button>
                )}

                <TextField
                    label="job name"
                    value={name}
                    onChange={(e) => {
                        const v = e.target.value
                        setName(v) // update UI immediately

                        // If you later add debounce, you can use saveTimerRef here,
                        // e.g. clearTimeout + setTimeout to call updateJobNameAsync.
                    }}
                    onBlur={() => {
                        if (!job?.id) return
                        if (saveTimerRef.current) clearTimeout(saveTimerRef.current)
                        ;(async () => {
                            try {
                                setIsSaving(true)
                                await dispatch(updateJobNameAsync({ id: job.id, name }))
                                setSnack({
                                    open: true,
                                    message: 'Saved job name'
                                })
                            } catch (e) {
                                console.error('Failed to save job name:', e)
                                setSnack({
                                    open: true,
                                    message: 'Failed to save name'
                                })
                            } finally {
                                setIsSaving(false)
                            }
                        })()
                    }}
                    fullWidth
                    sx={{ height: '56px' }}
                    helperText={isSaving ? 'Saving…' : ' '}
                />
            </Box>

            {renderContent(props.messages, job?.id)}
            <div ref={messagesEndRef} />

            {job?.status === 'inqueue' && (
                <Placeholder src="figures/queue.svg" title="" subtitle="job is in waiting list" />
            )}

            <Snackbar
                open={snack.open}
                autoHideDuration={1500}
                onClose={() => setSnack({ open: false, message: '' })}
                message={snack.message}
                anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
            />
        </div>
    )
}

export default PageCommon
