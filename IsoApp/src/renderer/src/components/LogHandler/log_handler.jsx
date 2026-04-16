import "./log.css";
import ProgressBar from './ProgressBar'
import VisibilityIcon from '@mui/icons-material/Visibility'
import { Button } from '@mui/material'
import ImageFromPath from "../ImageFromPath";

const ERROR_PAT =
    /(ERROR\b|Exception\b|AssertionError|RuntimeError|ValueError|TypeError|OSError|Segmentation fault|Process exited with code (?!0|null)\S*|KeyboardInterrupt)/i
const WARN_PAT = /\b(WARNING|WARN|UserWarning|DeprecationWarning)\b/
const INFO_PAT = /\b(INFO|DEBUG)\b/
const NEW_BLOCK_PAT =
    /^(\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\s*(INFO|DEBUG|WARNING|ERROR)\b)|^(INFO|DEBUG|WARNING|ERROR)\b/

const isBlank = (s) => !s || /^\s*$/.test(s)

function classifyLine(line, traceState) {
    if (traceState === 'trace') {
        if (isBlank(line) || NEW_BLOCK_PAT.test(line)) {
            return { level: 'normal', nextTraceState: 'none' }
        }
        return { level: 'trace', nextTraceState: 'trace' }
    }

    if (/^Traceback\b/i.test(line)) {
        return { level: 'traceHead', nextTraceState: 'trace' }
    }

    if (ERROR_PAT.test(line)) return { level: 'error', nextTraceState: 'none' }
    if (WARN_PAT.test(line)) return { level: 'warning', nextTraceState: 'none' }
    if (INFO_PAT.test(line)) return { level: 'info', nextTraceState: 'none' }
    if (/^Command: /.test(line)) return { level: 'meta', nextTraceState: 'none' }

    return { level: 'info', nextTraceState: 'none' }
}

const lineStyleByLevel = (themeMode = 'light') => ({
    error: {
        background: themeMode === 'dark' ? 'rgba(244, 67, 54, 0.15)' : 'rgba(244, 67, 54, 0.08)',
        borderLeft: '3px solid #f44336',
        color: themeMode === 'dark' ? '#ffcdd2' : '#b71c1c'
    },
    warning: {
        background: themeMode === 'dark' ? 'rgba(255, 193, 7, 0.18)' : 'rgba(255, 193, 7, 0.16)',
        borderLeft: '3px solid #FFC107',
        color: themeMode === 'dark' ? '#ffe082' : '#7a5d00'
    },
    info: {
        background: themeMode === 'dark' ? 'rgba(33, 150, 243, 0.18)' : 'rgba(33, 150, 243, 0.10)',
        borderLeft: '3px solid #2196F3',
        color: themeMode === 'dark' ? '#bbdefb' : '#0d47a1'
    },
    meta: {
        background: themeMode === 'dark' ? 'rgba(156, 39, 176, 0.20)' : 'rgba(156, 39, 176, 0.10)',
        borderLeft: '3px solid #9C27B0',
        color: themeMode === 'dark' ? '#e1bee7' : '#4a148c',
        fontWeight: 600
    },
    traceHead: {
        background: themeMode === 'dark' ? 'rgba(244, 67, 54, 0.15)' : 'rgba(244, 67, 54, 0.08)',
        borderLeft: '3px solid #f44336',
        color: themeMode === 'dark' ? '#ffcdd2' : '#b71c1c'
    },
    trace: {
        background: themeMode === 'dark' ? 'rgba(244, 67, 54, 0.15)' : 'rgba(244, 67, 54, 0.08)',
        borderLeft: '3px solid #f44336',
        color: themeMode === 'dark' ? '#ffcdd2' : '#b71c1c'
    },
    normal: {
        borderLeft: '3px solid transparent',
        color: 'inherit'
    }
})

export const renderContent = (messageList, id, themeMode = 'light') => {
    if (!messageList) return null
    const styles = lineStyleByLevel(themeMode)
    const handleView = (file) => {
        window.api.call('view', file)
    }
    let traceState = 'none' // 'none' | 'trace'

    return messageList.map((msg, i) => {
        if (msg.type === 'bar') {
            return <ProgressBar key={`bar-${id}-${i}`} currentProgress={msg} />
        }

        if (msg.type === 'power_spectrum') {
            let { epoch, folder, volume_file } = msg
            console.log(volume_file)
            const volumeName = volume_file?.split('/').pop().replace(/\.[^/.]+$/, '');

            const basePath = `${folder}/`
            if (!epoch) epoch = ""
            return (
                <div key={`ps-${i}`} className="epoch-block">
                    <div className="epoch-top-row">
                        <ImageFromPath
                            relativePath={`${basePath}${volumeName}_xy_epoch_${epoch}.png`}
                            imgStyle={{ width: '100%', display: 'block' }}
                        />

                        <div className="epoch-view-btn-column">
                            <Button
                                size="large"
                                color="secondary"
                                onClick={() => handleView(volume_file)}
                                className="epoch-icon-btn"
                            >
                                <VisibilityIcon className="epoch-icon" />
                            </Button>
                        </div>
                    </div>

                    <div className="epoch-caption">XY Slice</div>

                    <div className="epoch-bottom-row">
                        <div className="epoch-bottom-item">
                            <ImageFromPath
                                relativePath={`${basePath}${volumeName}_xz_epoch_${epoch}.png`}
                                imgStyle={{ width: '100%', display: 'block' }}
                            />
                            <div className="epoch-caption">XZ Slice</div>
                        </div>

                        <div className="epoch-bottom-item">
                            <ImageFromPath
                                relativePath={`${basePath}${volumeName}_power_epoch_${epoch}.png`}
                                imgStyle={{ width: '100%', display: 'block' }}
                            />
                            <div className="epoch-caption">Power Spectrum</div>
                        </div>
                    </div>
                </div>
            )
        }

        if (msg.type === 'text') {
            const lines = String(msg.output ?? '').split(/\r?\n/)
            return (
                <div key={`text-${i}`} style={{ marginBottom: 4 }}>
                    {lines.map((line, j) => {
                        const res = classifyLine(line, traceState)
                        traceState = res.nextTraceState
                        const style = {
                            ...styles[res.level]
                        }
                        const content = line === '' ? '\u00A0' : line
                        return (
                            <div key={`line-${i}-${j}`} className="log-line" style={style}>
                                {content}
                            </div>
                        )
                    })}
                </div>
            )
        }

        traceState = 'none'
        return null
    })
}
