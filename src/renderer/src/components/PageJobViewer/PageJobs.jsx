import './index.css'
import { useState, useEffect, useCallback } from 'react'
import {
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Paper,
    Button,
} from '@mui/material'
import { useDispatch } from 'react-redux'
import { updateJobStatusAsync, updateJobPIDAsync } from '../../store/jobSlice'

const PageJobs = ({ setBlocking }) => {
    const dispatch = useDispatch()

    const [jobList, setJobList] = useState({
        inQueueList: [],
        notInQueueList: [],
    })

    // Fetch queue info from main process (separate from Redux's jobList)
    const fetchJobLists = useCallback(async () => {
        try {
            const data = await window.api.call('jobsQueue')
            setJobList(
                data || { inQueueList: [], notInQueueList: [] }
            )
        } catch (e) {
            console.error('Failed to fetch jobsQueue:', e)
            setJobList({ inQueueList: [], notInQueueList: [] })
        }
    }, [])

    // Initial load
    useEffect(() => {
        fetchJobLists()
    }, [fetchJobLists])

    // Listen to python-status-change to update Redux + refresh queue view
    useEffect(() => {
        const off = window.api.on(
            'python-status-change',
            async ({ id, status, pid }) => {
                if (!id) return
                // Update Redux global jobList
                dispatch(updateJobStatusAsync({ id, status }))
                dispatch(updateJobPIDAsync({ id, pid }))
                // Refresh queue grouping view
                await fetchJobLists()
                setBlocking(false)
            }
        )

        return () => {
            try {
                off?.()
            } catch {
                // ignore
            }
        }
    }, [dispatch, fetchJobLists, setBlocking])

    // Kill a running job by PID and mark it completed
    const handleKillJob = (pid, id) => {
        if (!pid || !id) return
        setBlocking(true)
        window.api
            .call('killJob', pid)
            .then(async () => {
                await dispatch(
                    updateJobStatusAsync({ id, status: 'completed' })
                )
                await fetchJobLists()
            })
            .catch((err) => {
                console.error('Failed to kill job:', err)
            })
            .finally(() => {
                setBlocking(false)
            })
    }

    // Remove a job from queue (not necessarily killing the PID)
    const handleRemoveJob = (id) => {
        if (!id) return
        setBlocking(true)
        window.api
            .call('removeJobFromQueue', id)
            .then(async () => {
                await dispatch(
                    updateJobStatusAsync({ id, status: 'completed' })
                )
                await fetchJobLists()
            })
            .catch((err) => {
                console.error('Failed to remove job from queue:', err)
            })
            .finally(() => {
                setBlocking(false)
            })
    }

    const inQueue = jobList.inQueueList || []
    const notInQueue = jobList.notInQueueList || []

    return (
        <div>
            {/* Jobs in queue */}
            <div>
                <h4>Jobs in Queue</h4>
                <TableContainer component={Paper}>
                    <Table>
                        <TableHead>
                            <TableRow>
                                <TableCell>id</TableCell>
                                <TableCell>status</TableCell>
                                <TableCell>name</TableCell>
                                <TableCell>command</TableCell>
                                <TableCell>PID</TableCell>
                                <TableCell>actions</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {inQueue.length > 0 ? (
                                inQueue.map((job) => (
                                    <TableRow key={job.id ?? `queue-${job.pid}`}>
                                        <TableCell>{job.id ?? 'N/A'}</TableCell>
                                        <TableCell>{job.status}</TableCell>
                                        <TableCell>{job.name}</TableCell>
                                        <TableCell>{job.command_line}</TableCell>
                                        <TableCell>{job.pid ?? 'N/A'}</TableCell>
                                        <TableCell>
                                            <Button
                                                variant="contained"
                                                color={job.pid ? 'error' : 'secondary'}
                                                onClick={() =>
                                                    job.pid
                                                        ? handleKillJob(job.pid, job.id)
                                                        : handleRemoveJob(job.id)
                                                }
                                            >
                                                {job.pid ? 'Kill Job' : 'Remove from Queue'}
                                            </Button>
                                        </TableCell>
                                    </TableRow>
                                ))
                            ) : (
                                <TableRow>
                                    {/* 6 columns total */}
                                    <TableCell colSpan={6}>no jobs</TableCell>
                                </TableRow>
                            )}
                        </TableBody>
                    </Table>
                </TableContainer>
            </div>

            {/* Running jobs that are not in queue */}
            <div style={{ marginTop: 24 }}>
                <h4>Jobs Running but Not in Queue</h4>
                <TableContainer component={Paper}>
                    <Table>
                        <TableHead>
                            <TableRow>
                                <TableCell>id</TableCell>
                                <TableCell>status</TableCell>
                                <TableCell>name</TableCell>
                                <TableCell>command</TableCell>
                                <TableCell>PID</TableCell>
                                <TableCell>actions</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {notInQueue.length > 0 ? (
                                notInQueue.map((job) => (
                                    <TableRow key={job.id ?? `notqueue-${job.pid}`}>
                                        <TableCell>{job.id ?? 'N/A'}</TableCell>
                                        <TableCell>{job.status}</TableCell>
                                        <TableCell>{job.name}</TableCell>
                                        <TableCell>{job.command_line}</TableCell>
                                        <TableCell>{job.pid ?? 'N/A'}</TableCell>
                                        <TableCell>
                                            <Button
                                                variant="contained"
                                                color={job.pid ? 'error' : 'primary'}
                                                onClick={() =>
                                                    job.pid
                                                        ? handleKillJob(job.pid, job.id)
                                                        : handleRemoveJob(job.id)
                                                }
                                            >
                                                {job.pid ? 'Kill Job' : 'Remove from Queue'}
                                            </Button>
                                        </TableCell>
                                    </TableRow>
                                ))
                            ) : (
                                <TableRow>
                                    <TableCell colSpan={6}>no jobs</TableCell>
                                </TableRow>
                            )}
                        </TableBody>
                    </Table>
                </TableContainer>
            </div>
        </div>
    )
}

export default PageJobs
