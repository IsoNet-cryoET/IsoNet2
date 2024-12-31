import React from 'react'
import ProgressBar from './ProgressBar'

export const renderContent = (messageList) => {
    if (!messageList) return null
    // console.log('messageList', messageList)
    return messageList.map((msg) => {
        // if (msg.cmd !== page) return null

        if (msg.type === 'bar') {
            return <ProgressBar currentProgress={msg} />
        } else if (msg.type === 'text') {
            return (
                <div style={{ marginBottom: '10px' }}>
                    <pre style={{ margin: 0 }}>{msg.output}</pre>
                </div>
            )
        }
    })
}
