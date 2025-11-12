import React, { useState, useEffect } from 'react'
import {
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Paper,
    TextField,
    Button,
    Box
} from '@mui/material'
import FileOpenIcon from '@mui/icons-material/FileOpen'
import VisibilityIcon from '@mui/icons-material/Visibility'

const DataTable = ({ jsonData, star_name }) => {
    if (!jsonData || jsonData.length === 0) return null

    const [rows, setRows] = useState(flattenData(jsonData))
    const [focusedColumn, setFocusedColumn] = useState(null)

    const [columnWidths, setColumnWidths] = useState({})
    const [editingCell, setEditingCell] = useState({
        rowIndex: null,
        col: null,
        originalValue: null
    })

    useEffect(() => {
        setRows(flattenData(jsonData))
    }, [jsonData])

    // useEffect(() => {
    //     convertToJson()
    // }, [rows])

    const columns = Object.keys(jsonData[0]).filter((col) => col !== 'index')

    // const handleCellBlur = () => {
    //     convertToJson()
    // }
    const isNumeric = (value) => {
        if (typeof value === 'number') return true
        if (typeof value !== 'string') return false
        return value.trim() !== '' && !isNaN(Number(value))
    }
    const handleCellChange = (rowIndex, columnName, value) => {
        const updatedRows = [...rows]
        if (isNumeric(value)) {
            updatedRows[rowIndex][columnName] = Number(value)
        } else {
            updatedRows[rowIndex][columnName] = value
        }
        setRows(updatedRows)
    }

    const handleOpen = async (rowIndex, columnName) => {
        try {
            const filePath = await window.api.call('selectFile', 'openFile')
            if (filePath) {
                handleCellChange(rowIndex, columnName, filePath)
            }
        } catch (error) {
            console.error('Error selecting file:', error)
        }
    }

    const handleView = (file) => {
        window.api.call('view', file)
    }

    const convertToJson = () => {
        const convertedJson = {}
        columns.forEach((column) => {
            convertedJson[column] = {}
        })

        rows.forEach((row, rowIndex) => {
            columns.forEach((col) => {
                convertedJson[col][rowIndex] = row[col]
            })
        })
        window.api.call('updateStar', { convertedJson, star_name })
    }
    const measureTextWidth = (text, font = 'inherit') => {
        const canvas = document.createElement('canvas')
        const context = canvas.getContext('2d')
        context.font = font
        return context.measureText(text).width + 75 // Add some padding
    }

    const getCellWidth = (col) => {
        const firstValue = rows[0]?.[col]
        const valueStr = firstValue != null ? firstValue.toString() : ''
        const isShort = valueStr.length < 10

        const defaultWidth = isShort ? '75px' : '250px'
        const expandedWidth = columnWidths[col] || defaultWidth
        return {
            minWidth: focusedColumn === col ? expandedWidth : defaultWidth,
            maxWidth: focusedColumn === col ? expandedWidth : defaultWidth
        }
    }
    return (
        <TableContainer component={Paper} style={{ maxHeight: 500 }}>
            <Table stickyHeader>
                <TableHead>
                    <TableRow>
                        {columns.map((col) => {
                            // const isString = typeof rows[0][col] === 'string'
                            return (
                                <TableCell
                                    key={col}
                                    title={col}
                                    sx={{
                                        ...getCellWidth(col),
                                        whiteSpace: 'nowrap',
                                        textAlign: 'center',
                                        overflow: 'hidden',
                                        textOverflow: 'ellipsis',
                                        transition: 'min-width 0.3s ease, max-width 0.3s ease' // ✨ Smooth transition
                                    }}
                                >
                                    {col}
                                </TableCell>
                            )
                        })}
                    </TableRow>
                </TableHead>
                <TableBody>
                    {rows.map((row, rowIndex) => (
                        <TableRow key={rowIndex}>
                            {columns.map((col) => (
                                <TableCell
                                    key={col}
                                    sx={{
                                        ...getCellWidth(col),
                                        whiteSpace: 'nowrap',
                                        overflow: 'hidden',
                                        textOverflow: 'ellipsis',
                                        padding: '8px',
                                        transition: 'min-width 0.3s ease, max-width 0.3s ease' // ✨ Smooth transition
                                    }}
                                >
                                    <Box display="flex" alignItems="center" gap={1}>
                                        <TextField
                                            variant="outlined"
                                            value={row[col]}
                                            onChange={(e) =>
                                                handleCellChange(rowIndex, col, e.target.value)
                                            }
                                            onFocus={(e) => {
                                                setFocusedColumn(col)
                                                const font = getComputedStyle(e.target).font
                                                const width = measureTextWidth(e.target.value, font)
                                                setColumnWidths((prev) => ({
                                                    ...prev,
                                                    [col]: width
                                                }))
                                                setEditingCell({
                                                    rowIndex,
                                                    col,
                                                    originalValue: row[col]
                                                })
                                            }}
                                            onBlur={() => {
                                                setFocusedColumn(null)

                                                const editedValue = rows[rowIndex][col]
                                                if (
                                                    editingCell.rowIndex === rowIndex &&
                                                    editingCell.col === col &&
                                                    editingCell.originalValue !== editedValue
                                                ) {
                                                    convertToJson()
                                                }

                                                setEditingCell({
                                                    rowIndex: null,
                                                    col: null,
                                                    originalValue: null
                                                })
                                            }}
                                            size="small"
                                            inputProps={{
                                                title: row[col],
                                                style: {
                                                    overflow: 'auto',
                                                    textOverflow: 'ellipsis',
                                                    whiteSpace: 'nowrap'
                                                }
                                            }}
                                            fullWidth
                                        />

                                        {typeof row[col] === 'string' && (
                                            <Box
                                                display="flex"
                                                flexDirection="column"
                                                alignItems="center"
                                                justifyContent="center"
                                                sx={{
                                                    width: 'auto',
                                                    padding: 0, // Remove extra padding
                                                    margin: 0 // Remove extra margin
                                                }}
                                            >
                                                <Button
                                                    size="small"
                                                    color="primary"
                                                    onClick={() => handleOpen(rowIndex, col)}
                                                    sx={{
                                                        padding: 0, // Remove padding
                                                        margin: 0, // Remove margin
                                                        minWidth: 0 // Ensure the button occupies minimal space
                                                    }}
                                                >
                                                    <FileOpenIcon
                                                        sx={{
                                                            color: '#14446e',
                                                            fontSize: 'medium'
                                                        }}
                                                    />
                                                </Button>
                                                <Button
                                                    size="small"
                                                    color="secondary"
                                                    onClick={() => handleView(row[col])}
                                                    sx={{
                                                        padding: 0, // Remove padding
                                                        margin: 0, // Remove margin
                                                        minWidth: 0 // Ensure the button occupies minimal space
                                                    }}
                                                >
                                                    <VisibilityIcon
                                                        sx={{
                                                            color: '#14446e',
                                                            fontSize: 'medium'
                                                        }}
                                                    />
                                                </Button>
                                            </Box>
                                        )}
                                    </Box>
                                </TableCell>
                            ))}
                        </TableRow>
                    ))}
                </TableBody>
            </Table>
        </TableContainer>
    )
}

const flattenData = (jsonData) => {
    return Object.keys(jsonData[0].rlnIndex).map((index) => {
        const row = {}
        Object.keys(jsonData[0]).forEach((key) => {
            row[key] = jsonData[0][key][index]
        })
        return row
    })
}

export default DataTable
