import { createSlice, createAsyncThunk } from '@reduxjs/toolkit'

export const fetchJobs = createAsyncThunk('jobs/fetchJobs', async () => {
    const jobs = await window.api.call('getJobList')
    return jobs || []
})

export const addJobAsync = createAsyncThunk(
    'jobs/addJob',
    async (job) => {
        const jobs = await window.api.call('addJob', job)
        return jobs || []
    }
)

export const updateJobAsync = createAsyncThunk(
    'jobs/updateJob',
    async (data) => {
        const jobs = await window.api.call('updateJob', data)
        return jobs || []
    }
)

export const updateJobStatusAsync = createAsyncThunk(
    'jobs/updateJobStatus',
    async ({ id, status }) => {
        const jobs = await window.api.call('updateJobStatus', id, status)
        return jobs || []
    }
)

export const updateJobPIDAsync = createAsyncThunk(
    'jobs/updateJobPID',
    async ({ id, pid }) => {
        const jobs = await window.api.call('updateJobPID', id, pid)
        return jobs || []
    }
)

export const updateJobNameAsync = createAsyncThunk(
    'jobs/updateJobName',
    async ({ id, name }) => {
        const jobs = await window.api.call('updateJobName', id, name)
        return jobs || []
    }
)

export const removeJobAsync = createAsyncThunk(
    'jobs/removeJob',
    async (id) => {
        const jobs = await window.api.call('removeJob', id)
        return jobs || []
    }
)

const jobSlice = createSlice({
    name: 'jobs',
    initialState: {
        jobList: [],
        status: 'idle',
        error: null,
    },
    reducers: {},
    extraReducers: (builder) => {
        builder
            .addCase(fetchJobs.fulfilled, (state, action) => {
                state.jobList = action.payload
            })
            .addCase(addJobAsync.fulfilled, (state, action) => {
                state.jobList = action.payload
            })
            .addCase(updateJobAsync.fulfilled, (state, action) => {
                state.jobList = action.payload
            })
            .addCase(updateJobStatusAsync.fulfilled, (state, action) => {
                state.jobList = action.payload
            })
            .addCase(updateJobPIDAsync.fulfilled, (state, action) => {
                state.jobList = action.payload
            })
            .addCase(updateJobNameAsync.fulfilled, (state, action) => {
                state.jobList = action.payload
            })
            .addCase(removeJobAsync.fulfilled, (state, action) => {
                state.jobList = action.payload
            })
    },
})

export default jobSlice.reducer
