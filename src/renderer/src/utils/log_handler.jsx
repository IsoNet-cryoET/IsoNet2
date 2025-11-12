import ProgressBar from './ProgressBar'
import React, { useEffect, useState } from 'react'
import VisibilityIcon from '@mui/icons-material/Visibility'
import { Button } from '@mui/material'
// —— 级别识别：支持跨消息 Traceback 状态 —— //
const ERROR_PAT =
    /(ERROR\b|Exception\b|AssertionError|RuntimeError|ValueError|TypeError|OSError|Segmentation fault|Process exited with code (?!0|null)\S*|KeyboardInterrupt)/i
const WARN_PAT = /\b(WARNING|WARN|UserWarning|DeprecationWarning)\b/
const INFO_PAT = /\b(INFO|DEBUG)\b/

// 新段落（时间戳或等级）判定
const NEW_BLOCK_PAT =
    /^(\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\s*(INFO|DEBUG|WARNING|ERROR)\b)|^(INFO|DEBUG|WARNING|ERROR)\b/

const isBlank = (s) => !s || /^\s*$/.test(s)

function classifyLine(line, traceState /* 'none' | 'trace' */) {
    if (traceState === 'trace') {
        if (isBlank(line) || NEW_BLOCK_PAT.test(line)) {
            // 空行或新块 -> 结束 traceback 详情
            return { level: 'normal', nextTraceState: 'none' }
        }
        return { level: 'trace', nextTraceState: 'trace' }
    }

    // Traceback 头
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

/** —— 你的图片小组件：支持传入自定义 img props —— */
const ImageFromPath = ({ label, relativePath, imgStyle, ...imgProps }) => {
    const [imgData, setImgData] = useState(null)
    useEffect(() => {
        let mounted = true
        window.api.call('getImageData', relativePath).then((res) => {
            if (!mounted) return
            if (res.success) setImgData(res.content)
            else console.error(`Error loading image ${relativePath}: ${res.error}`)
        })
        return () => {
            mounted = false
        }
    }, [relativePath])

    return (
        <div style={{ textAlign: 'center' }}>
            {label && <div style={{ fontSize: '12px', marginBottom: '4px' }}>{label}</div>}
            {imgData ? (
                <img
                    src={imgData}
                    alt={label || relativePath}
                    style={{
                        width: '100%',
                        border: '1px solid #ccc',
                        borderRadius: '4px',
                        ...imgStyle
                    }}
                    {...imgProps}
                />
            ) : (
                <div
                    style={{
                        width: '180px',
                        height: '135px',
                        background: '#eee',
                        display: 'inline-flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        borderRadius: 4,
                        border: '1px solid #ddd',
                        color: '#777'
                    }}
                >
                    Loading...
                </div>
            )}
        </div>
    )
}

/** —— 页面内容渲染 —— */
export const renderContent = (messageList, id, themeMode = 'light') => {
    if (!messageList) return null
    const styles = lineStyleByLevel(themeMode)
    const handleView = (file) => {
        window.api.call('view', file)
    }
    let traceState = 'none' // 跨消息状态：'none' | 'trace'

    return messageList.map((msg, i) => {
        if (msg.type === 'bar') {
            return <ProgressBar key={`bar-${id}-${i}`} currentProgress={msg} />
        }

        if (msg.type === 'power_spectrum') {
            const { epoch, folder, volume_file } = msg
            console.log(volume_file)
            const basePath = `${folder}/`
            return (
                <div
                    key={`ps-${i}`}
                    style={{
                        display: 'flex',
                        flexDirection: 'column',
                        gap: '10px',
                        alignItems: 'center',
                        marginBottom: '20px'
                    }}
                >
                    <div
                        style={{
                            display: 'flex',
                            flexDirection: 'row',
                            justifyContent: 'center',
                            gap: '10px',
                            width: '60%'
                        }}
                    >
                        <ImageFromPath
                            relativePath={`${basePath}xy_epoch_${epoch}.png`}
                            imgStyle={{ width: '100%', display: 'block' }}
                        />
                        <div
                            style={{
                                display: 'flex',
                                flexDirection: 'column',
                                // justifyContent: 'center',
                                gap: '10px',
                                width: '5%'
                            }}
                        >
                            <Button
                                size="large"
                                color="secondary"
                                onClick={() => handleView(volume_file)}
                                sx={{
                                    padding: 0, // Remove padding
                                    margin: 0, // Remove margin
                                    minWidth: 0 // Ensure the button occupies minimal space
                                }}
                            >
                                <VisibilityIcon
                                    sx={{
                                        color: '#14446e',
                                        fontSize: 'large'
                                    }}
                                />
                            </Button>
                        </div>
                    </div>
                    <div style={{ textAlign: 'center' }}>XY Slice</div>
                    <div
                        style={{
                            display: 'flex',
                            flexDirection: 'row',
                            justifyContent: 'center',
                            gap: '10px',
                            width: '60%'
                        }}
                    >
                        <div style={{ flex: 1 }}>
                            <ImageFromPath
                                relativePath={`${basePath}xz_epoch_${epoch}.png`}
                                imgStyle={{ width: '100%', display: 'block' }}
                            />
                            <div style={{ textAlign: 'center' }}>XZ Slice</div>
                        </div>
                        <div style={{ flex: 1 }}>
                            <ImageFromPath
                                relativePath={`${basePath}power_epoch_${epoch}.png`}
                                imgStyle={{ width: '100%', display: 'block' }}
                            />
                            <div style={{ textAlign: 'center' }}>Power Spectrum</div>
                        </div>
                    </div>
                </div>
            )
        }

        if (msg.type === 'text') {
            // 按行渲染，并跨消息更新 traceState
            const lines = String(msg.output ?? '').split(/\r?\n/)
            return (
                <div key={`text-${i}`} style={{ marginBottom: 4 }}>
                    {lines.map((line, j) => {
                        const res = classifyLine(line, traceState)
                        traceState = res.nextTraceState
                        const style = {
                            ...styles[res.level]
                        }
                        // 空行也要渲染一个高度（用不间断空格）
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

        // 碰到非 text 的消息时，保险起见结束 traceback 块
        traceState = 'none'
        return null
    })
}
