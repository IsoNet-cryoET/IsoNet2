import './index.css'
import { useEffect, useRef, useState } from 'react'
import { renderContent } from '../LogHandler/log_handler'
import { Box, Button, Snackbar, TextField } from '@mui/material'
import CleaningServicesIcon from '@mui/icons-material/CleaningServices'
import Nodata from '../NoData/Nodata'
import { useDispatch } from 'react-redux'
import { removeJobAsync, updateJobNameAsync } from '../../store/jobSlice'

const PageCommon = (props) => {
    const job = props?.selectedJob
    const [isDeleting, setIsDeleting] = useState(false)
    const [isSaving, setIsSaving] = useState(false)
    const [snack, setSnack] = useState({ open: false, message: '' })
    const dispatch = useDispatch()

    // Local name state, so typing is responsive even before Redux updates
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
            <Nodata
                message="Nothing here yet"
                sub="Create a new job from the left menu to get started"
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
                            ; (async () => {
                                try {
                                    setIsSaving(true)
                                    await dispatch(
                                        updateJobNameAsync({ id: job.id, name })
                                    )
                                    setSnack({
                                        open: true,
                                        message: 'Saved job name',
                                    })
                                } catch (e) {
                                    console.error('Failed to save job name:', e)
                                    setSnack({
                                        open: true,
                                        message: 'Failed to save name',
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
