import { useState, useCallback } from 'react'

export function useDrawerForm(initialValues) {
    const [formData, setFormData] = useState(initialValues)

    const handleChange = useCallback((field, value) => {
        setFormData((prev) => ({ ...prev, [field]: value }))
    }, [])

    const handleFileSelect = useCallback(async (field, property) => {
        let folderPath = await window.api.call('selectFile', property)
        setFormData((prevState) => ({
            ...prevState,
            [field]: folderPath
        }))
    }, [])

    const handleSubmit = useCallback((status, onClose, onSubmit) => {
        const updatedFormData = {
            ...formData,
            status
        }
        onSubmit(updatedFormData)
        onClose()
    })

    return {
        formData,
        setFormData,
        handleChange,
        handleFileSelect,
        handleSubmit,
    }
}
