import { useMemo, useState, useEffect } from 'react';
import {
  Accordion, AccordionSummary, AccordionDetails,
  Typography, Box, IconButton, Tooltip, Snackbar
} from '@mui/material';
import ArrowDownwardIcon from '@mui/icons-material/ArrowDownward';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import toCommand from '../../utils/handle_json';

export default function CommandAccordion({ formData }) {
  const [copied, setCopied] = useState(false);
  const [commandLine, setCommandLine] = useState(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const id = await window.api.call('getCurrentId') + 1;
      const cmd = toCommand(formData, id);
      if (!cancelled) setCommandLine(cmd);
    })();
    return () => { cancelled = true; };
  }, [formData]);

  const textToCopy = useMemo(
    () => (typeof commandLine === 'string'
      ? commandLine
      : JSON.stringify(commandLine, null, 2)),
    [commandLine]
  );

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(textToCopy);
      setCopied(true);
    } catch {
      const ta = document.createElement('textarea');
      ta.value = textToCopy;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
      setCopied(true);
    }
  };

  return (
    <>
      <Accordion className="cmd-accordion">
        <AccordionSummary
          expandIcon={<ArrowDownwardIcon />}
          className="cmd-accordion-summary"
        >
          <Typography component="span">Show command</Typography>
        </AccordionSummary>

        <AccordionDetails>
          <Box className="cmd-container">
            <Box className="cmd-copy-btn-wrapper">
              <Tooltip title="Copy to clipboard">
                <IconButton size="small" onClick={handleCopy} aria-label="Copy">
                  <ContentCopyIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </Box>

            <Typography component="pre" className="cmd-pre">
              {textToCopy}
            </Typography>
          </Box>
        </AccordionDetails>
      </Accordion>


      <Snackbar
        open={copied}
        autoHideDuration={1200}
        onClose={() => setCopied(false)}
        message="Copied!"
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      />
    </>
  );
}
