import ProgressBar from './ProgressBar'
import React, { useEffect, useState } from 'react'

const ImageFromPath = ({ label, relativePath }) => {
    const [imgData, setImgData] = useState(null)

    useEffect(() => {
        window.api.call('getImageData', relativePath).then((res) => {
            if (res.success) setImgData(res.content)
            else console.error(`Error loading image ${relativePath}: ${res.error}`)
        })
    }, [relativePath])

    return (
        <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '12px', marginBottom: '4px' }}>{label}</div>
            {imgData ? (
                <img
                    src={imgData}
                    alt={label}
                    style={{
                        width: '100%',
                        border: '1px solid #ccc',
                        borderRadius: '4px'
                    }}
                />
            ) : (
                <div style={{ width: '180px', height: '135px', background: '#eee' }}>
                    Loading...
                </div>
            )}
        </div>
    )
}
export const renderContent = (messageList) => {
    if (!messageList) return null
    return messageList.map((msg) => {
        // if (msg.cmd !== page) return null

        if (msg.type === 'bar') {
            return <ProgressBar currentProgress={msg} />
        } else if (msg.type === 'power_spectrum') {
            const { epoch, folder } = msg
            const basePath = `${folder}/`

            return (
                <div
                    style={{
                        display: 'flex',
                        flexDirection: 'column',
                        gap: '10px',
                        alignItems: 'center',
                        marginBottom: '20px'
                    }}
                >
                    {/* XY Slice - Full width or 2x relative size */}
                    <div style={{ width: '50%' }}>
                        <ImageFromPath
                            relativePath={`${basePath}xy_epoch_${epoch}.png`}
                            style={{ width: '100%', display: 'block' }}
                        />
                        <div style={{ textAlign: 'center' }}>XY Slice</div>
                    </div>

                    {/* XZ + Power Spectrum Row */}
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
                                style={{ width: '100%', display: 'block' }}
                            />
                            <div style={{ textAlign: 'center' }}>XZ Slice</div>
                        </div>
                        <div style={{ flex: 1 }}>
                            <ImageFromPath
                                relativePath={`${basePath}power_epoch_${epoch}.png`}
                                style={{ width: '100%', display: 'block' }}
                            />
                            <div style={{ textAlign: 'center' }}>Power Spectrum</div>
                        </div>
                    </div>
                </div>
            )
        } else if (msg.type === 'text') {
            return (
                <div style={{ marginBottom: '10px' }}>
                    <p style={{ fontFamily: 'monospace' }}>{msg.output}</p>
                </div>
            )
        }
    })
}
