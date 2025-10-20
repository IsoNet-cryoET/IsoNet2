import { useEffect, useState, useMemo } from 'react'
import { List, ListItemButton, ListItemText, CircularProgress } from '@mui/material'

function JobsList({ selectedPrimaryMenu, jobs, setJobs, selectedJob, setSelectedJob }) {

  // Filter for current menu
  const visibleJobs = useMemo(
    () => jobs.filter((j) => j.type === selectedPrimaryMenu),
    [jobs, selectedPrimaryMenu]
  )

  return (
    <List>
      {visibleJobs.map((job) => (
        <ListItemButton
          key={job.id}
          selected={selectedJob?.id === job.id}
          onClick={() => setSelectedJob(job)}
        >
          {job.status === 'running' && <CircularProgress size={16} />}
          <ListItemText primary={`job ${job.id} ${job.name}`} />
        </ListItemButton>
      ))}
    </List>
  )
}

export default JobsList
 