'use client'

import React, { useMemo, useRef, useEffect, forwardRef, useImperativeHandle, useCallback } from 'react'
import { VariableSizeList as List } from 'react-window'
import { motion, AnimatePresence } from 'framer-motion'
import { messageAnimations } from '../lib/animations'
import MarkdownRenderer from './MarkdownRenderer'
import { Bot, User, Loader2 } from 'lucide-react'
import { formatTimestamp } from '../lib/api'

// ===== TYPES =====

interface Message {
  id: string
  content: string
  sender: 'user' | 'ai'
  timestamp: Date
  isStreaming?: boolean
}

interface VirtualizedMessageListProps {
  messages: Message[]
  isLoading: boolean
  className?: string
}

interface MessageItemProps {
  index: number
  style: React.CSSProperties
  data: {
    messages: Message[]
    isLoading: boolean
    getItemHeight: (index: number) => number
  }
}

export interface VirtualizedMessageListRef {
  scrollToBottom: () => void
  scrollToMessage: (messageId: string) => void
}

// ===== CONSTANTS =====

const ESTIMATED_MESSAGE_HEIGHT = 120 // Base height for estimation
const AVATAR_SIZE = 40
const MESSAGE_PADDING = 24
const MESSAGE_SPACING = 24

// ===== UTILITY FUNCTIONS =====

const estimateMessageHeight = (message: Message): number => {
  // More sophisticated height estimation
  const baseHeight = AVATAR_SIZE + MESSAGE_PADDING * 2

  // Calculate content height based on content analysis
  let contentHeight = 40 // Minimum content height

  if (message.content) {
    // Count actual line breaks
    const explicitLines = message.content.split('\n').length

    // Estimate wrapped lines (assuming 80 characters per line)
    const wrappedLines = Math.ceil(message.content.length / 80)

    // Use the larger of the two estimates
    const estimatedLines = Math.max(explicitLines, wrappedLines)

    // Line height varies by message type
    const lineHeight = message.sender === 'ai' ? 28 : 24
    contentHeight = Math.max(estimatedLines * lineHeight, 40)

    // Add extra height for AI messages with potential markdown elements
    if (message.sender === 'ai') {
      // Check for markdown elements that add height
      const codeBlocks = (message.content.match(/```/g) || []).length / 2
      const lists = (message.content.match(/^[\s]*[-*+]/gm) || []).length
      const headers = (message.content.match(/^#+/gm) || []).length

      // Add extra height for special elements
      contentHeight += codeBlocks * 60 + lists * 8 + headers * 16
    }
  }

  // Add timestamp height
  const timestampHeight = 20

  // Calculate total with spacing
  const totalHeight = baseHeight + contentHeight + timestampHeight + MESSAGE_SPACING

  return Math.ceil(totalHeight)
}

// ===== MESSAGE ITEM COMPONENT =====

const MessageItem = React.memo(({ index, style, data }: MessageItemProps) => {
  const { messages, isLoading } = data
  const message = messages[index]
  
  if (!message) return null

  return (
    <div style={style}>
      <motion.div
        className={`flex space-x-4 px-6 py-3 ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
        variants={messageAnimations}
        initial="hidden"
        animate="visible"
        layout
      >
        {message.sender === 'ai' && (
          <motion.div 
            className="w-10 h-10 bg-gradient-to-br from-indigo-600 to-purple-700 rounded-2xl flex items-center justify-center shadow-lg flex-shrink-0"
            initial={{ scale: 0, rotate: -180 }}
            animate={{ scale: 1, rotate: 0 }}
            transition={{ delay: 0.1, type: "spring", stiffness: 300, damping: 30 }}
          >
            <Bot className="h-5 w-5 text-white" />
          </motion.div>
        )}
        
        <motion.div className={`max-w-3xl ${
          message.sender === 'user' 
            ? 'bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-2xl rounded-br-md' 
            : 'bg-white border border-slate-200 rounded-2xl rounded-bl-md'
        } px-6 py-4 shadow-sm`}>
          <div className={`${message.sender === 'user' ? 'text-white' : 'text-slate-900'}`}>
            {message.sender === 'ai' ? (
              <MarkdownRenderer content={message.content} />
            ) : (
              <div className="whitespace-pre-wrap">{message.content}</div>
            )}
          </div>
          <div className={`text-xs mt-2 ${message.sender === 'user' ? 'text-indigo-100' : 'text-slate-500'}`}>
            {formatTimestamp(message.timestamp)}
          </div>
        </motion.div>
        
        {message.sender === 'user' && (
          <motion.div 
            className="w-10 h-10 bg-gradient-to-br from-slate-600 to-slate-700 rounded-2xl flex items-center justify-center shadow-lg flex-shrink-0"
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.1, type: "spring", stiffness: 300, damping: 30 }}
          >
            <User className="h-5 w-5 text-white" />
          </motion.div>
        )}
      </motion.div>
    </div>
  )
})

MessageItem.displayName = 'MessageItem'

// ===== LOADING ITEM COMPONENT =====

const LoadingItem = React.memo(({ style }: { style: React.CSSProperties }) => (
  <div style={style}>
    <motion.div 
      className="flex space-x-4 justify-start px-6 py-3"
      variants={messageAnimations}
      initial="hidden"
      animate="visible"
      exit="exit"
    >
      <motion.div 
        className="w-10 h-10 bg-gradient-to-br from-indigo-600 to-purple-700 rounded-2xl flex items-center justify-center shadow-lg flex-shrink-0"
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        transition={{ type: "spring", stiffness: 300, damping: 30 }}
      >
        <Bot className="h-5 w-5 text-white" />
      </motion.div>
      <motion.div 
        className="bg-white border border-slate-200 rounded-2xl rounded-bl-md px-6 py-4 shadow-sm"
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.1, type: "spring", stiffness: 300, damping: 30 }}
      >
        <div className="flex items-center space-x-2">
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
          >
            <Loader2 className="h-4 w-4 text-indigo-600" />
          </motion.div>
          <span className="text-slate-600">AI is thinking...</span>
        </div>
      </motion.div>
    </motion.div>
  </div>
))

LoadingItem.displayName = 'LoadingItem'

// ===== MAIN COMPONENT =====

const VirtualizedMessageList = forwardRef<VirtualizedMessageListRef, VirtualizedMessageListProps>(
  ({ messages, isLoading, className = '' }, ref) => {
    const listRef = useRef<List>(null)
    const heightCache = useRef<Map<number, number>>(new Map())
    const performanceRef = useRef({ renderCount: 0, lastRenderTime: 0 })

    // Calculate total items (messages + loading indicator)
    const totalItems = messages.length + (isLoading ? 1 : 0)

    // Performance monitoring
    useEffect(() => {
      performanceRef.current.renderCount++
      performanceRef.current.lastRenderTime = Date.now()

      // Log performance metrics in development
      if (process.env.NODE_ENV === 'development' && performanceRef.current.renderCount % 10 === 0) {
        console.log(`VirtualizedMessageList: ${performanceRef.current.renderCount} renders, ${messages.length} messages`)
      }
    }, [messages.length])
    
    // Memoized height calculation
    const getItemHeight = useMemo(() => {
      return (index: number): number => {
        // Check cache first
        if (heightCache.current.has(index)) {
          return heightCache.current.get(index)!
        }
        
        let height: number
        
        if (index < messages.length) {
          // Regular message
          height = estimateMessageHeight(messages[index])
        } else {
          // Loading indicator
          height = 80
        }
        
        // Cache the height
        heightCache.current.set(index, height)
        return height
      }
    }, [messages])
    
    // Memoized item data
    const itemData = useMemo(() => ({
      messages,
      isLoading,
      getItemHeight
    }), [messages, isLoading, getItemHeight])
    
    // Clear height cache when messages change significantly
    useEffect(() => {
      heightCache.current.clear()
    }, [messages.length])
    
    // Smooth scroll to bottom function with momentum
    const scrollToBottom = useCallback(() => {
      if (listRef.current && totalItems > 0) {
        // Use smooth scrolling behavior
        listRef.current.scrollToItem(totalItems - 1, 'end')

        // Add a small delay to ensure smooth animation
        setTimeout(() => {
          if (listRef.current) {
            listRef.current.scrollToItem(totalItems - 1, 'end')
          }
        }, 50)
      }
    }, [totalItems])

    // Scroll to specific message with smooth animation
    const scrollToMessage = useCallback((messageId: string) => {
      const messageIndex = messages.findIndex(msg => msg.id === messageId)
      if (messageIndex !== -1 && listRef.current) {
        listRef.current.scrollToItem(messageIndex, 'center')
      }
    }, [messages])
    
    // Auto-scroll to bottom when new messages arrive
    useEffect(() => {
      const timer = setTimeout(() => {
        scrollToBottom()
      }, 100) // Small delay to ensure rendering is complete
      
      return () => clearTimeout(timer)
    }, [messages.length, isLoading])
    
    // Expose methods via ref
    useImperativeHandle(ref, () => ({
      scrollToBottom,
      scrollToMessage
    }))
    
    // Render item function
    const renderItem = ({ index, style }: { index: number; style: React.CSSProperties }) => {
      if (index < messages.length) {
        return <MessageItem index={index} style={style} data={itemData} />
      } else {
        return <LoadingItem style={style} />
      }
    }
    
    if (totalItems === 0) {
      return <div className={className} />
    }
    
    return (
      <div className={`${className} relative`}>
        <List
          ref={listRef}
          height={600} // This will be overridden by parent container
          width="100%"
          itemCount={totalItems}
          itemSize={getItemHeight}
          itemData={itemData}
          overscanCount={8} // Render 8 extra items for smoother scrolling
          className="scrollbar-hide"
          useIsScrolling={true} // Enable scroll state optimization
          direction="vertical"
          layout="vertical"
          style={{
            // Smooth scrolling CSS
            scrollBehavior: 'smooth',
            // Hardware acceleration
            willChange: 'transform',
            // Optimize for performance
            contain: 'layout style paint'
          }}
        >
          {renderItem}
        </List>
      </div>
    )
  }
)

VirtualizedMessageList.displayName = 'VirtualizedMessageList'

export default VirtualizedMessageList
