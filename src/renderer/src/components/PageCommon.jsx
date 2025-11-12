import { useEffect, useRef, useState } from 'react'
import { renderContent } from '../utils/log_handler'
import { Box, Button, Snackbar, TextField } from '@mui/material'
import CleaningServicesIcon from '@mui/icons-material/CleaningServices'
import Nodata from './Nodata'

const PageCommon = (props) => {
    const job = props?.selectedJob
    const [isDeleting, setIsDeleting] = useState(false)
    const [isSaving, setIsSaving] = useState(false)
    const [snack, setSnack] = useState({ open: false, message: '' })

    // local name for responsive input
    const [name, setName] = useState(job?.name || '')
    useEffect(() => {
        setName(job?.name || '')
    }, [job?.id]) // when a new job is selected, reset input to its name

    // simple debounce without extra deps
    const saveTimerRef = useRef(null)

    const handleDelete = async () => {
        if (!job?.id) return
        try {
            setIsDeleting(true)
            const ok = await window.api.call('removeJob', job.id)
            if (ok) setSnack({ open: true, message: `Deleted job ${job.id}` })
        } catch (e) {
            console.error('Delete job failed:', e)
            setSnack({ open: true, message: 'Failed to delete job' })
        } finally {
            setIsDeleting(false)
        }
    }

    if (!job) {
        return (
            <Nodata message="Nothing here yet" sub="Create a new job from the left menu to get started" />
        )
    }
    return (
        <div>
            <Box display="flex" alignItems="center" gap={2} marginY={2} sx={{ flexGrow: 1, height: "100%" }}>
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
                        //   debouncedSaveName(v)    // save after debounce
                    }}
                    onBlur={() => {
                        // ensure a final save on blur (immediate, no debounce)
                        if (!job?.id) return
                        if (saveTimerRef.current) clearTimeout(saveTimerRef.current)
                            ; (async () => {
                                try {
                                    setIsSaving(true)
                                    const ok = await window.api.call('updateJobName', job.id, name)

                                    if (ok) setSnack({ open: true, message: 'Saved job name' })
                                } catch (e) {
                                    console.error('Update name failed:', e)
                                    setSnack({ open: true, message: 'Failed to save name' })
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
