import "./index.css";
import { useEffect, useState } from 'react';
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
	const [condaEnv, setCondaEnv] = useState('');
	const [isoNetPath, setIsoNetPath] = useState('');
	const [envOptions, setEnvOptions] = useState([]); // [{name, path, active}]

	// Load saved values + available envs once
	useEffect(() => {
		const run = async () => {
			try {
				const [savedEnv, savedIsoNetPath, envListResp] = await Promise.all([
					window.api.call('getCondaEnv'),
					window.api.call('getIsoNetPath'),
					window.api.call('getAvailableCondaEnv'),
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
			await window.api.call('setCondaEnv', value); // value can be '' for None
		} catch (e) {
			console.error('setCondaEnv failed:', e);
		}
	};

	const handleIsoNetPathChange = (v) => setIsoNetPath(v);

	const handleIsoNetPathBlur = async (v) => {
		try {
			await window.api.call('setIsoNetPath', v);
		} catch (e) {
			console.error('setIsoNetPath failed:', e);
		}
	};

	const handleFileSelect = async (property) => {
		try {
			const filePath = await window.api.call('selectFile', property);
			if (!filePath) return; // canceled
			setIsoNetPath(filePath);
			await window.api.call('setIsoNetPath', filePath);
		} catch (error) {
			console.error('Error selecting file:', error);
		}
	};

	const labelId = 'conda-env-label';
	const selectId = 'conda-env-select';

	return (
		<div>
			<Box className="settings-row">
				<FormControl
					fullWidth
					variant="standard"
					className="conda-select-form"
				>
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

			<Box className="settings-row">
				<TextField
					label="IsoNet install path"
					value={isoNetPath}
					onChange={(e) => handleIsoNetPathChange(e.target.value)}
					onBlur={(e) => handleIsoNetPathBlur(e.target.value)}
					fullWidth
					className="isonet-path-input"
				/>
				<Button
					disableFocusRipple
					variant="outlined"
					color="primary"
					startIcon={<FolderOpenIcon />}
					onClick={() => handleFileSelect('openDirectory')}
					className="open-folder-button"
				>
					Open
				</Button>
			</Box>
		</div>
	);
};

export default PageSettings;
