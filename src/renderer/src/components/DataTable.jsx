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

    useEffect(() => {
        setRows(flattenData(jsonData))
    }, [jsonData])

    // useEffect(() => {
    //     convertToJson()
    // }, [rows])

    const columns = Object.keys(jsonData[0]).filter((col) => col !== 'index')

    const handleCellBlur = () => {
        convertToJson()
    }
    const handleCellChange = (rowIndex, columnName, value) => {
        const updatedRows = [...rows]
        updatedRows[rowIndex][columnName] = value
        setRows(updatedRows)
    }

    const handleOpen = async (rowIndex, columnName) => {
        try {
            const filePath = await api.selectFile('openFile')
            if (filePath) {
                handleCellChange(rowIndex, columnName, filePath)
            }
        } catch (error) {
            console.error('Error selecting file:', error)
        }
    }

    const handleView = (file) => {
        api.view(file)
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
        api.updateStar({ convertedJson, star_name })
    }

    return (
        <TableContainer component={Paper}>
            <Table stickyHeader>
                <TableHead>
                    <TableRow>
                        {columns.map((col) => {
                            const isString = typeof rows[0][col] === 'string'
                            return (
                                <TableCell
                                    key={col}
                                    sx={{
                                        whiteSpace: 'normal',
                                        wordWrap: 'break-word',
                                        textAlign: 'center',
                                        minWidth: isString ? '250px' : '75px', // Adjust width based on content type
                                        maxWidth: isString ? '250px' : '75px' // Keep the maxWidth consistent
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
                                <TableCell key={col}>
                                    <Box display="flex" alignItems="center" gap={1}>
                                        <TextField
                                            value={row[col]}
                                            onChange={(e) =>
                                                handleCellChange(rowIndex, col, e.target.value)
                                            }
                                            onBlur={handleCellBlur} // Save on blur
                                            variant="outlined"
                                            size="small"
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
