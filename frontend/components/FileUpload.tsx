'use client'

import React, { useState, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Upload, 
  File, 
  Image, 
  FileText, 
  Music, 
  Video, 
  X, 
  Check,
  AlertCircle,
  Download,
  Eye,
  Trash2
} from 'lucide-react'
import { useUIActions } from '../store'

// ===== TYPES =====

interface FileUploadProps {
  onFileSelect?: (files: File[]) => void
  onFileUpload?: (files: UploadedFile[]) => void
  maxFiles?: number
  maxFileSize?: number // in MB
  acceptedTypes?: string[]
  showPreview?: boolean
  className?: string
}

interface UploadedFile {
  id: string
  file: File
  name: string
  size: number
  type: string
  status: 'uploading' | 'completed' | 'error'
  progress: number
  preview?: string
  metadata?: {
    dimensions?: { width: number; height: number }
    duration?: number
    pages?: number
    extractedText?: string
  }
  error?: string
}

// ===== FILE TYPE DETECTION =====

const getFileIcon = (type: string) => {
  if (type.startsWith('image/')) return <Image className="h-5 w-5" />
  if (type.startsWith('video/')) return <Video className="h-5 w-5" />
  if (type.startsWith('audio/')) return <Music className="h-5 w-5" />
  if (type.includes('pdf') || type.includes('document') || type.includes('text')) {
    return <FileText className="h-5 w-5" />
  }
  return <File className="h-5 w-5" />
}

const getFileCategory = (type: string): string => {
  if (type.startsWith('image/')) return 'image'
  if (type.startsWith('video/')) return 'video'
  if (type.startsWith('audio/')) return 'audio'
  if (type.includes('pdf')) return 'pdf'
  if (type.includes('document') || type.includes('text')) return 'document'
  return 'other'
}

const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes'
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

// ===== FILE UPLOAD COMPONENT =====

export const FileUpload: React.FC<FileUploadProps> = ({
  onFileSelect,
  onFileUpload,
  maxFiles = 10,
  maxFileSize = 50, // 50MB default
  acceptedTypes = [
    'image/*',
    'video/*',
    'audio/*',
    'application/pdf',
    'text/*',
    '.doc,.docx,.txt,.md'
  ],
  showPreview = true,
  className = ''
}) => {
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([])
  const [isDragOver, setIsDragOver] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const { addNotification } = useUIActions()

  // ===== FILE VALIDATION =====

  const validateFile = (file: File): string | null => {
    // Check file size
    if (file.size > maxFileSize * 1024 * 1024) {
      return `File size exceeds ${maxFileSize}MB limit`
    }

    // Check file type
    const isAccepted = acceptedTypes.some(type => {
      if (type.includes('*')) {
        return file.type.startsWith(type.replace('*', ''))
      }
      if (type.startsWith('.')) {
        return file.name.toLowerCase().endsWith(type.toLowerCase())
      }
      return file.type === type
    })

    if (!isAccepted) {
      return 'File type not supported'
    }

    return null
  }

  // ===== FILE PROCESSING =====

  const processFile = async (file: File): Promise<UploadedFile> => {
    const uploadedFile: UploadedFile = {
      id: `file-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      file,
      name: file.name,
      size: file.size,
      type: file.type,
      status: 'uploading',
      progress: 0,
    }

    // Generate preview for images
    if (file.type.startsWith('image/')) {
      uploadedFile.preview = await generateImagePreview(file)
      uploadedFile.metadata = await getImageMetadata(file)
    }

    // Extract text from documents
    if (file.type.includes('text') || file.name.endsWith('.txt') || file.name.endsWith('.md')) {
      uploadedFile.metadata = { extractedText: await extractTextFromFile(file) }
    }

    return uploadedFile
  }

  const generateImagePreview = (file: File): Promise<string> => {
    return new Promise((resolve) => {
      const reader = new FileReader()
      reader.onload = (e) => resolve(e.target?.result as string)
      reader.readAsDataURL(file)
    })
  }

  const getImageMetadata = (file: File): Promise<{ dimensions: { width: number; height: number } }> => {
    return new Promise((resolve) => {
      if (typeof window !== 'undefined') {
        const img = window.Image ? new window.Image() : new (globalThis as any).Image()
        img.onload = () => {
          resolve({
            dimensions: { width: img.width, height: img.height }
          })
        }
        img.src = URL.createObjectURL(file)
      } else {
        resolve({ dimensions: { width: 0, height: 0 } })
      }
    })
  }

  const extractTextFromFile = (file: File): Promise<string> => {
    return new Promise((resolve) => {
      const reader = new FileReader()
      reader.onload = (e) => {
        const text = e.target?.result as string
        resolve(text.substring(0, 1000)) // Limit to first 1000 characters
      }
      reader.readAsText(file)
    })
  }

  // ===== FILE UPLOAD SIMULATION =====

  const simulateUpload = async (uploadedFile: UploadedFile): Promise<void> => {
    return new Promise((resolve) => {
      let progress = 0
      const interval = setInterval(() => {
        progress += Math.random() * 20
        if (progress >= 100) {
          progress = 100
          clearInterval(interval)
          resolve()
        }
        
        setUploadedFiles(prev => 
          prev.map(f => 
            f.id === uploadedFile.id 
              ? { ...f, progress }
              : f
          )
        )
      }, 200)
    })
  }

  // ===== EVENT HANDLERS =====

  const handleFileSelect = useCallback(async (files: FileList) => {
    const fileArray = Array.from(files)
    
    // Check max files limit
    if (uploadedFiles.length + fileArray.length > maxFiles) {
      addNotification({
        type: 'warning',
        title: 'Too Many Files',
        message: `Maximum ${maxFiles} files allowed`,
        duration: 3000,
      })
      return
    }

    // Validate and process files
    const validFiles: File[] = []
    const errors: string[] = []

    for (const file of fileArray) {
      const error = validateFile(file)
      if (error) {
        errors.push(`${file.name}: ${error}`)
      } else {
        validFiles.push(file)
      }
    }

    // Show validation errors
    if (errors.length > 0) {
      addNotification({
        type: 'error',
        title: 'File Validation Errors',
        message: errors.join(', '),
        duration: 5000,
      })
    }

    if (validFiles.length === 0) return

    setIsUploading(true)

    try {
      // Process files
      const processedFiles = await Promise.all(
        validFiles.map(file => processFile(file))
      )

      // Add to uploaded files list
      setUploadedFiles(prev => [...prev, ...processedFiles])

      // Simulate upload for each file
      await Promise.all(
        processedFiles.map(async (uploadedFile) => {
          try {
            await simulateUpload(uploadedFile)
            
            setUploadedFiles(prev => 
              prev.map(f => 
                f.id === uploadedFile.id 
                  ? { ...f, status: 'completed', progress: 100 }
                  : f
              )
            )
          } catch (error) {
            setUploadedFiles(prev => 
              prev.map(f => 
                f.id === uploadedFile.id 
                  ? { ...f, status: 'error', error: 'Upload failed' }
                  : f
              )
            )
          }
        })
      )

      // Notify callbacks
      onFileSelect?.(validFiles)
      onFileUpload?.(processedFiles)

      addNotification({
        type: 'success',
        title: 'Files Uploaded',
        message: `${validFiles.length} file(s) uploaded successfully`,
        duration: 3000,
      })

    } catch (error) {
      addNotification({
        type: 'error',
        title: 'Upload Failed',
        message: 'Failed to upload files. Please try again.',
        duration: 5000,
      })
    } finally {
      setIsUploading(false)
    }
  }, [uploadedFiles.length, maxFiles, onFileSelect, onFileUpload, addNotification])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
    
    const files = e.dataTransfer.files
    if (files.length > 0) {
      handleFileSelect(files)
    }
  }, [handleFileSelect])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
  }, [])

  const removeFile = (fileId: string) => {
    setUploadedFiles(prev => prev.filter(f => f.id !== fileId))
  }

  const openFileDialog = () => {
    fileInputRef.current?.click()
  }

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Upload Area */}
      <motion.div
        className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
          isDragOver
            ? 'border-indigo-500 bg-indigo-50'
            : 'border-gray-300 hover:border-gray-400'
        }`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        whileHover={{ scale: 1.01 }}
        whileTap={{ scale: 0.99 }}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept={acceptedTypes.join(',')}
          onChange={(e) => e.target.files && handleFileSelect(e.target.files)}
          className="hidden"
        />

        <div className="space-y-4">
          <div className="flex justify-center">
            <Upload className={`h-12 w-12 ${isDragOver ? 'text-indigo-500' : 'text-gray-400'}`} />
          </div>
          
          <div>
            <p className="text-lg font-medium text-gray-900">
              {isDragOver ? 'Drop files here' : 'Upload files'}
            </p>
            <p className="text-sm text-gray-500 mt-1">
              Drag and drop files here, or{' '}
              <button
                onClick={openFileDialog}
                className="text-indigo-600 hover:text-indigo-700 font-medium"
              >
                browse
              </button>
            </p>
          </div>

          <div className="text-xs text-gray-400">
            <p>Supported: Images, Videos, Audio, Documents, PDFs</p>
            <p>Max file size: {maxFileSize}MB • Max files: {maxFiles}</p>
          </div>
        </div>

        {isUploading && (
          <div className="absolute inset-0 bg-white bg-opacity-75 flex items-center justify-center rounded-lg">
            <div className="text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600 mx-auto"></div>
              <p className="text-sm text-gray-600 mt-2">Processing files...</p>
            </div>
          </div>
        )}
      </motion.div>

      {/* Uploaded Files List */}
      <AnimatePresence>
        {uploadedFiles.length > 0 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="space-y-2"
          >
            <h3 className="text-sm font-medium text-gray-700">
              Uploaded Files ({uploadedFiles.length})
            </h3>
            
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {uploadedFiles.map((file) => (
                <FileItem
                  key={file.id}
                  file={file}
                  onRemove={removeFile}
                  showPreview={showPreview}
                />
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

// ===== FILE ITEM COMPONENT =====

interface FileItemProps {
  file: UploadedFile
  onRemove: (fileId: string) => void
  showPreview: boolean
}

const FileItem: React.FC<FileItemProps> = ({ file, onRemove, showPreview }) => {
  const [showFullPreview, setShowFullPreview] = useState(false)

  const getStatusIcon = () => {
    switch (file.status) {
      case 'completed':
        return <Check className="h-4 w-4 text-green-500" />
      case 'error':
        return <AlertCircle className="h-4 w-4 text-red-500" />
      default:
        return (
          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-indigo-600"></div>
        )
    }
  }

  const getStatusColor = () => {
    switch (file.status) {
      case 'completed':
        return 'border-green-200 bg-green-50'
      case 'error':
        return 'border-red-200 bg-red-50'
      default:
        return 'border-blue-200 bg-blue-50'
    }
  }

  return (
    <>
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -10 }}
        className={`border rounded-lg p-3 ${getStatusColor()}`}
      >
        <div className="flex items-center space-x-3">
          {/* File Icon */}
          <div className="flex-shrink-0">
            {getFileIcon(file.type)}
          </div>

          {/* File Info */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center justify-between">
              <p className="text-sm font-medium text-gray-900 truncate">
                {file.name}
              </p>
              <div className="flex items-center space-x-2">
                {getStatusIcon()}
                <button
                  onClick={() => onRemove(file.id)}
                  className="text-gray-400 hover:text-red-500 transition-colors"
                  title="Remove file"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            </div>

            <div className="flex items-center justify-between mt-1">
              <p className="text-xs text-gray-500">
                {formatFileSize(file.size)} • {getFileCategory(file.type)}
              </p>

              {showPreview && file.preview && (
                <button
                  onClick={() => setShowFullPreview(true)}
                  className="text-xs text-indigo-600 hover:text-indigo-700 flex items-center"
                >
                  <Eye className="h-3 w-3 mr-1" />
                  Preview
                </button>
              )}
            </div>

            {/* Progress Bar */}
            {file.status === 'uploading' && (
              <div className="mt-2">
                <div className="bg-gray-200 rounded-full h-1">
                  <div
                    className="bg-indigo-600 h-1 rounded-full transition-all duration-300"
                    style={{ width: `${file.progress}%` }}
                  />
                </div>
                <p className="text-xs text-gray-500 mt-1">
                  {Math.round(file.progress)}% uploaded
                </p>
              </div>
            )}

            {/* Error Message */}
            {file.status === 'error' && file.error && (
              <p className="text-xs text-red-600 mt-1">{file.error}</p>
            )}

            {/* Metadata */}
            {file.metadata && (
              <div className="mt-2 text-xs text-gray-500">
                {file.metadata.dimensions && (
                  <span>
                    {file.metadata.dimensions.width} × {file.metadata.dimensions.height}
                  </span>
                )}
                {file.metadata.extractedText && (
                  <p className="mt-1 italic">
                    "{file.metadata.extractedText.substring(0, 100)}..."
                  </p>
                )}
              </div>
            )}
          </div>

          {/* Preview Thumbnail */}
          {showPreview && file.preview && (
            <div className="flex-shrink-0">
              <img
                src={file.preview}
                alt={file.name}
                className="w-12 h-12 object-cover rounded cursor-pointer"
                onClick={() => setShowFullPreview(true)}
              />
            </div>
          )}
        </div>
      </motion.div>

      {/* Full Preview Modal */}
      <AnimatePresence>
        {showFullPreview && file.preview && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-75"
            onClick={() => setShowFullPreview(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="max-w-4xl max-h-4xl p-4"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="bg-white rounded-lg overflow-hidden">
                <div className="flex items-center justify-between p-4 border-b">
                  <h3 className="text-lg font-semibold text-gray-900">{file.name}</h3>
                  <button
                    onClick={() => setShowFullPreview(false)}
                    className="text-gray-400 hover:text-gray-600"
                  >
                    <X className="h-6 w-6" />
                  </button>
                </div>

                <div className="p-4">
                  <img
                    src={file.preview}
                    alt={file.name}
                    className="max-w-full max-h-96 mx-auto"
                  />

                  {file.metadata && (
                    <div className="mt-4 text-sm text-gray-600">
                      <p>Size: {formatFileSize(file.size)}</p>
                      {file.metadata.dimensions && (
                        <p>
                          Dimensions: {file.metadata.dimensions.width} × {file.metadata.dimensions.height}
                        </p>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}

export default FileUpload
