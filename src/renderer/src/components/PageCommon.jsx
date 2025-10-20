import { useEffect, useState } from 'react'
import { renderContent } from '../utils/log_handler'
import { IconButton, Box, Button, Snackbar } from '@mui/material'
import AddCardIcon from '@mui/icons-material/AddCard'
import CleaningServicesIcon from '@mui/icons-material/CleaningServices'
import { mergeMsg, processMessage } from '../utils/utils'
import { DeleteRounded } from '@mui/icons-material'

const PageCommon = (props) => {
    const [deleting, setDeleting] = useState(false)
    const [snackOpen, setSnackOpen] = useState(false)
    
    // const [showButton, setShowButton] = useState(false)

    const handleDelete = async () => {
        if (!props?.selectedJob?.id) return

        try {
            setDeleting(true)
            const ok = await window.jobList.remove(props.selectedJob.id)
            console.log(ok)
            if (ok) {
              setSnackOpen(true)
            }
          } catch (e) {
            console.error('Delete job failed:', e)
          } finally {
            setDeleting(false)
            // setShowButton(false)
          }
        }
    // useEffect(()=>{
    //     if (props?.selectedJob?.status === "completed")
    //         setShowButton(true)
    // }
    // , [props?.selectedJob?.status])

    return (
        <div>
            {props?.selectedJob?.status === "completed" &&
            <Box display="flex" alignItems="center" gap={2} marginY={2}>
                <Button
                    variant="outlined"
                    color="primary"
                    startIcon={<CleaningServicesIcon />}
                    onClick={() => handleDelete()}
                    disabled={!props?.selectedJob?.id || deleting || props?.selectedJob?.status == "running"}
                    sx={{ height: '56px' }} // Ensure the button has a height
                >
                    permanently remove job files
                </Button>
            </Box>
            }
            {renderContent(props.messages, props?.selectedJob?.id)}
            <Snackbar
                    open={snackOpen}
                    autoHideDuration={1500}
                    onClose={() => setSnackOpen(false)}
                    message={`Deleted Job ${props?.selectedJob?.id}!`}
                    anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
                />
        </div>
    )
}
export default PageCommon


// const handleFileSelect = async (field, property) => {
//     const filePath = await window.api.selectFile(property)
//     if (!filePath) return

//     const fileContent = await window.api.readFile(filePath)
//     if (!fileContent) return

//     const lines = fileContent
//         .split(/\r?\n|\r/g) // handles both \n and ^M
//         .filter((line) => line.trim() !== '') // remove empty lines
//     const newMessages = []
//     for (const line of lines) {
//         const msg = { cmd: 'denoise', output: line }
//         const processed = processMessage(msg)
//         const merged = mergeMsg(newMessages, processed)
//         newMessages.length = 0
//         newMessages.push(...merged)
//     }

//     props.setMessages(newMessages)
// }

// const handleClear = () => {
//     props.setMessages([])
// }