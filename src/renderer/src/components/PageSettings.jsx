import React, { useEffect, useState } from 'react';
import {
  Box,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
} from '@mui/material';
import FolderOpenIcon from '@mui/icons-material/FolderOpen';

const PageSettings = () => {

  console.log("render setting page")
  const [condaEnv, setCondaEnv] = useState('');
  const [isoNetPath, setIsoNetPath] = useState('');
  const [envOptions, setEnvOptions] = useState([]); // [{name, path, active}]

  const envApi = typeof window !== 'undefined' ? window.environment : undefined;
  const appApi = typeof window !== 'undefined' ? window.api : undefined;

  // Load saved values + available envs once
  useEffect(() => {
    console.log("setting page first useEffect")
    const run = async () => {
      try {
        const [savedEnv, savedIsoNetPath, envListResp] = await Promise.all([
          envApi?.getCondaEnv?.(),
          envApi?.getIsoNetPath?.(),
          envApi?.getAvailableCondaEnv?.(),
        ]);

        if (typeof savedEnv === 'string') setCondaEnv(savedEnv);
        if (typeof savedIsoNetPath === 'string') setIsoNetPath(savedIsoNetPath);

        if (envListResp?.success && Array.isArray(envListResp.envs)) {
          setEnvOptions(envListResp.envs);
          // If nothing saved (empty string means "None", so don't auto-pick)
          if (savedEnv === undefined || savedEnv === null) {
            const active = envListResp.envs.find((e) => e.active);
            if (active?.name) setCondaEnv(active.name);
          }
        }
      } catch (e) {
        console.error('Failed to init PageSettings:', e);
      }
    };
    run();
  }, []);

  const handleCondaChange = async (value) => {
    setCondaEnv(value);
    try {
      await envApi?.setCondaEnv?.(value); // value can be '' for None
    } catch (e) {
      console.error('setCondaEnv failed:', e);
    }
  };

  const handleIsoNetPathChange = (v) => setIsoNetPath(v);

  const handleIsoNetPathBlur = async (v) => {
    try {
      await envApi?.setIsoNetPath?.(v);
    } catch (e) {
      console.error('setIsoNetPath failed:', e);
    }
  };

  const handleFileSelect = async (property) => {
    try {
      const filePath = await appApi?.selectFile?.(property);
      if (!filePath) return; // canceled
      setIsoNetPath(filePath);
      await envApi?.setIsoNetPath?.(filePath);
    } catch (error) {
      console.error('Error selecting file:', error);
    }
  };

  const labelId = 'conda-env-label';
  const selectId = 'conda-env-select';

  return (
    <div>
      <Box display="flex" alignItems="center" gap={2} my={2}>
        <FormControl fullWidth variant="standard" sx={{ minWidth: 240 }}>
          <InputLabel id={labelId}>Conda environment</InputLabel>
          <Select
            labelId={labelId}
            id={selectId}
            value={condaEnv ?? ''}
            onChange={(e) => handleCondaChange(e.target.value)}
            label="Conda environment"
            renderValue={(v) => (v ? v : 'None')}
          >
            {/* None option */}
            <MenuItem value="">
              <em>None</em>
            </MenuItem>

            {/* Actual envs */}
            {envOptions.map((env) => (
              <MenuItem key={env.name} value={env.name}>
                {env.name} — {env.path}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Box>

      <Box display="flex" alignItems="center" gap={2} my={2}>
        <TextField
          label="IsoNet install path"
          value={isoNetPath}
          onChange={(e) => handleIsoNetPathChange(e.target.value)}
          onBlur={(e) => handleIsoNetPathBlur(e.target.value)}
          fullWidth
          sx={{
            '& .MuiInputBase-root': { height: 56 },
            '& .MuiInputBase-input': { padding: '0 14px' },
          }}
        />
        <Button
          variant="outlined"
          color="primary"
          startIcon={<FolderOpenIcon />}
          onClick={() => handleFileSelect('openDirectory')}
          sx={{
            height: '56px',
            transition: 'all 0.2s ease', // fixed typo
          }}
        >
          Open
        </Button>
      </Box>
    </div>
  );
};

export default PageSettings;
