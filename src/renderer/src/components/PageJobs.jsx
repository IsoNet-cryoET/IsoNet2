import React, { useState, useEffect } from 'react'
import {
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Paper,
    Button,
    Checkbox
} from '@mui/material'
const PageJobs = () => {
    const [jobList, setJobList] = useState({ inQueueList: [], notInQueueList: [] })

    const fetchJobLists = async () => {
        let data = await api.onFetchJobs()
        console.log('jobs', data)
        setJobList(data)
    }

    useEffect(() => {
        fetchJobLists()
    }, [])

    window.api.onPythonClosed(() => {
        fetchJobLists()
    })

    const handleKillJob = (pid) => {
        console.log(`Attempting to kill Job with PID: ${pid}`)
        api.killJob(pid)
            .then(() => {
                console.log(`Job with PID ${pid} killed`)
                fetchJobLists() // Refresh job list after killing the job
            })
            .catch((error) => {
                console.error('Error killing job:', error) // Handle failure
            })
    }

    const handleRemoveJob = (result) => {
        console.log(`Attempt to remove Job ${result} from queue`)
        api.removeJobFromQueue(result).then(() => {
            console.log(`Job ${result} removed from queue`)
            fetchJobLists() // Refresh job list after removing the job
        })
    }
    return (
        <div>
            <div>
                <h4>Jobs in Queue</h4>
                <TableContainer component={Paper}>
                    <Table>
                        <TableHead>
                            <TableRow>
                                <TableCell>Status</TableCell>
                                <TableCell>Result</TableCell>
                                <TableCell>PID</TableCell>
                                <TableCell>Actions</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {(jobList.inQueueList || []).length > 0 ? (
                                jobList.inQueueList.map((job, index) => (
                                    <TableRow key={index}>
                                        <TableCell>{job.status}</TableCell>
                                        <TableCell>{job.result}</TableCell>
                                        <TableCell>{job.pid || 'N/A'}</TableCell>
                                        <TableCell>
                                            <Button
                                                variant="contained"
                                                color={job.pid ? 'error' : 'secondary'}
                                                onClick={() =>
                                                    job.pid
                                                        ? handleKillJob(job.pid)
                                                        : handleRemoveJob(job.result)
                                                }
                                            >
                                                {job.pid ? 'Kill Job' : 'Remove from Queue'}
                                            </Button>
                                        </TableCell>
                                    </TableRow>
                                ))
                            ) : (
                                <TableRow>
                                    <TableCell colSpan={4}>No jobs</TableCell>
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
                                <TableCell>Status</TableCell>
                                <TableCell>Result</TableCell>
                                <TableCell>PID</TableCell>
                                <TableCell>Actions</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {(jobList.notInQueueList || []).length > 0 ? (
                                jobList.notInQueueList.map((job, index) => (
                                    <TableRow key={index}>
                                        <TableCell>{job.status}</TableCell>
                                        <TableCell>{job.result}</TableCell>
                                        <TableCell>{job.pid || 'N/A'}</TableCell>
                                        <TableCell>
                                            <Button
                                                variant="contained"
                                                color={job.pid ? 'error' : 'primary'}
                                                onClick={() =>
                                                    job.pid
                                                        ? handleKillJob(job.pid)
                                                        : handleRemoveJob(job.pid)
                                                }
                                            >
                                                {job.pid ? 'Kill Job' : 'Remove from Queue'}
                                            </Button>
                                        </TableCell>
                                    </TableRow>
                                ))
                            ) : (
                                <TableRow>
                                    <TableCell colSpan={4}>No jobs</TableCell>
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
