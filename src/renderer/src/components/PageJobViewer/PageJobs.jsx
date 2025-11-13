import "./index.css";
import { useState, useEffect } from 'react'
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

const PageJobs = ({ setBlocking }) => {
    const [jobList, setJobList] = useState({ inQueueList: [], notInQueueList: [] })

    const fetchJobLists = async () => {
        let data = await window.api.call('jobsQueue')
        setJobList(data)
    }

    useEffect(() => {
        fetchJobLists()
    }, [])

    useEffect(() => {
        const off = window.api.on('python-status-change', async ({ id, status, pid }) => {
            await window.api.call('updateJobStatus', id, status).catch(() => { })
            await window.api.call('updateJobPID', id, pid).catch(() => { })
            await fetchJobLists()
            setBlocking(false)
        })
        return () => { try { off?.() } catch { } }
    }, [])

    const handleKillJob = (pid, id) => {
        setBlocking(true)                                  // 开遮罩
        window.api.call('killJob', pid)
            .then(async () => {
                await window.api.call('updateJobStatus', id, 'completed')
                await fetchJobLists()
            })
            .catch(console.error)
            .finally()               // 关遮罩
    }

    const handleRemoveJob = (id) => {
        setBlocking(true)                                  // 开遮罩
        window.api.call('removeJobFromQueue', id)
            .then(() => {
                window.api.call('updateJobStatus', id, 'completed')
                fetchJobLists()
            }).then(() => setBlocking(false))
            .catch(console.error)
            .finally()               // 关遮罩
    }
    return (
        <div>
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
                            {(jobList.inQueueList || []).length > 0 ? (
                                jobList.inQueueList.map((job, index) => (
                                    <TableRow key={index}>
                                        <TableCell>{job.id || 'N/A'}</TableCell>
                                        <TableCell>{job.status}</TableCell>
                                        <TableCell>{job.name}</TableCell>
                                        <TableCell>{job.command_line}</TableCell>
                                        <TableCell>{job.pid || 'N/A'}</TableCell>
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
                                    <TableCell colSpan={4}>no jobs</TableCell>
                                </TableRow>
                            )}
                        </TableBody>
                    </Table>
                </TableContainer>
            </div>

            <div>
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
                            {(jobList.notInQueueList || []).length > 0 ? (
                                jobList.notInQueueList.map((job, index) => (
                                    <TableRow key={index}>
                                        <TableCell>{job.id || 'N/A'}</TableCell>
                                        <TableCell>{job.status}</TableCell>
                                        <TableCell>{job.name}</TableCell>
                                        <TableCell>{job.command_line}</TableCell>
                                        <TableCell>{job.pid || 'N/A'}</TableCell>
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
                                    <TableCell colSpan={4}>no jobs</TableCell>
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
