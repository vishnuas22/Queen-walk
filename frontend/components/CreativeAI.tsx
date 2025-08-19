'use client'

import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Lightbulb, 
  Palette, 
  Wand2, 
  Sparkles,
  Image,
  Music,
  FileText,
  Video,
  Layers,
  Shuffle,
  Download,
  Share2,
  Heart,
  Star,
  Zap,
  Brain,
  Rocket
} from 'lucide-react'
import { useUIActions } from '../store'

// ===== TYPES =====

interface CreativeAIProps {
  onIdeaGenerated?: (idea: CreativeIdea) => void
  onWorkflowCreated?: (workflow: CreativeWorkflow) => void
  className?: string
}

interface CreativeIdea {
  id: string
  title: string
  description: string
  category: 'design' | 'content' | 'strategy' | 'innovation'
  confidence: number
  tags: string[]
  inspirationSources: string[]
  implementationSteps: string[]
  estimatedTime: string
  difficulty: 'easy' | 'medium' | 'hard'
  createdAt: Date
}

interface MoodBoard {
  id: string
  title: string
  theme: string
  colors: string[]
  images: string[]
  keywords: string[]
  mood: string
  style: string
}

interface CreativeWorkflow {
  id: string
  name: string
  steps: WorkflowStep[]
  category: string
  estimatedTime: string
  tools: string[]
  outputs: string[]
}

interface WorkflowStep {
  id: string
  title: string
  description: string
  duration: string
  tools: string[]
  inputs: string[]
  outputs: string[]
}

// ===== CREATIVE AI COMPONENT =====

export const CreativeAI: React.FC<CreativeAIProps> = ({
  onIdeaGenerated,
  onWorkflowCreated,
  className = ''
}) => {
  const [activeTab, setActiveTab] = useState<'ideas' | 'moodboard' | 'workflow' | 'inspiration'>('ideas')
  const [isGenerating, setIsGenerating] = useState(false)
  const [generatedIdeas, setGeneratedIdeas] = useState<CreativeIdea[]>([])
  const [currentMoodBoard, setCurrentMoodBoard] = useState<MoodBoard | null>(null)
  const [workflows, setWorkflows] = useState<CreativeWorkflow[]>([])
  const [inspirationPrompt, setInspirationPrompt] = useState('')
  const [selectedCategory, setSelectedCategory] = useState<'design' | 'content' | 'strategy' | 'innovation'>('design')

  const { addNotification } = useUIActions()

  // ===== IDEA GENERATION =====

  const generateIdeas = async () => {
    setIsGenerating(true)

    try {
      // Simulate AI idea generation
      await new Promise(resolve => setTimeout(resolve, 2000))

      const newIdeas: CreativeIdea[] = [
        {
          id: `idea-${Date.now()}-1`,
          title: 'Adaptive UI Personality System',
          description: 'Create an AI interface that adapts its personality and communication style based on user preferences, context, and emotional state.',
          category: selectedCategory,
          confidence: 0.87,
          tags: ['AI', 'Personalization', 'UX', 'Emotion Recognition'],
          inspirationSources: ['Human psychology', 'Chatbot evolution', 'Gaming NPCs'],
          implementationSteps: [
            'Research personality frameworks',
            'Design personality trait system',
            'Implement emotion detection',
            'Create adaptive response engine',
            'Test with user groups'
          ],
          estimatedTime: '6-8 weeks',
          difficulty: 'hard',
          createdAt: new Date()
        },
        {
          id: `idea-${Date.now()}-2`,
          title: 'Collaborative Brainstorming Canvas',
          description: 'A real-time collaborative space where multiple users can brainstorm with AI assistance, featuring mind mapping, idea clustering, and creative prompts.',
          category: selectedCategory,
          confidence: 0.92,
          tags: ['Collaboration', 'Brainstorming', 'Real-time', 'Mind Mapping'],
          inspirationSources: ['Miro/Figma collaboration', 'Design thinking workshops', 'AI creativity research'],
          implementationSteps: [
            'Design canvas interface',
            'Implement real-time sync',
            'Add AI suggestion engine',
            'Create idea clustering algorithms',
            'Build export functionality'
          ],
          estimatedTime: '4-6 weeks',
          difficulty: 'medium',
          createdAt: new Date()
        },
        {
          id: `idea-${Date.now()}-3`,
          title: 'Contextual Micro-Learning System',
          description: 'An AI system that provides just-in-time learning suggestions and micro-tutorials based on user actions and current context.',
          category: selectedCategory,
          confidence: 0.79,
          tags: ['Learning', 'Context-aware', 'Micro-interactions', 'Education'],
          inspirationSources: ['Just-in-time learning', 'Contextual help systems', 'Adaptive learning'],
          implementationSteps: [
            'Analyze user interaction patterns',
            'Create learning content database',
            'Implement context detection',
            'Design micro-tutorial UI',
            'Add progress tracking'
          ],
          estimatedTime: '3-4 weeks',
          difficulty: 'medium',
          createdAt: new Date()
        }
      ]

      setGeneratedIdeas(prev => [...newIdeas, ...prev])
      
      addNotification({
        type: 'success',
        title: 'Ideas Generated',
        message: `${newIdeas.length} creative ideas have been generated`,
        duration: 3000,
      })

      // Notify parent component
      newIdeas.forEach(idea => onIdeaGenerated?.(idea))

    } catch (error) {
      addNotification({
        type: 'error',
        title: 'Generation Failed',
        message: 'Failed to generate ideas. Please try again.',
        duration: 5000,
      })
    } finally {
      setIsGenerating(false)
    }
  }

  // ===== MOOD BOARD GENERATION =====

  const generateMoodBoard = async (theme: string) => {
    setIsGenerating(true)

    try {
      await new Promise(resolve => setTimeout(resolve, 1500))

      const moodBoard: MoodBoard = {
        id: `mood-${Date.now()}`,
        title: `${theme} Inspiration Board`,
        theme,
        colors: ['#6366f1', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981'],
        images: [
          'https://images.unsplash.com/photo-1558655146-9f40138edfeb?w=300',
          'https://images.unsplash.com/photo-1557804506-669a67965ba0?w=300',
          'https://images.unsplash.com/photo-1573164713714-d95e436ab8d6?w=300',
          'https://images.unsplash.com/photo-1551650975-87deedd944c3?w=300'
        ],
        keywords: ['Innovation', 'Creativity', 'Technology', 'Future', 'Inspiration'],
        mood: 'Innovative and forward-thinking',
        style: 'Modern minimalist with vibrant accents'
      }

      setCurrentMoodBoard(moodBoard)

      addNotification({
        type: 'success',
        title: 'Mood Board Created',
        message: `Inspiration board for "${theme}" has been generated`,
        duration: 3000,
      })

    } catch (error) {
      addNotification({
        type: 'error',
        title: 'Generation Failed',
        message: 'Failed to create mood board. Please try again.',
        duration: 5000,
      })
    } finally {
      setIsGenerating(false)
    }
  }

  // ===== WORKFLOW AUTOMATION =====

  const createWorkflow = async (workflowType: string) => {
    setIsGenerating(true)

    try {
      await new Promise(resolve => setTimeout(resolve, 1000))

      const workflow: CreativeWorkflow = {
        id: `workflow-${Date.now()}`,
        name: `${workflowType} Creative Process`,
        category: workflowType,
        estimatedTime: '2-4 hours',
        tools: ['AI Assistant', 'Design Tools', 'Collaboration Platform'],
        outputs: ['Concept Document', 'Visual Mockups', 'Implementation Plan'],
        steps: [
          {
            id: 'step-1',
            title: 'Research & Discovery',
            description: 'Gather inspiration and analyze requirements',
            duration: '30 minutes',
            tools: ['Research Tools', 'AI Analysis'],
            inputs: ['Project Brief', 'Target Audience'],
            outputs: ['Research Summary', 'Inspiration Collection']
          },
          {
            id: 'step-2',
            title: 'Ideation & Brainstorming',
            description: 'Generate and explore creative concepts',
            duration: '45 minutes',
            tools: ['Brainstorming Canvas', 'AI Idea Generator'],
            inputs: ['Research Summary', 'Creative Brief'],
            outputs: ['Concept Ideas', 'Initial Sketches']
          },
          {
            id: 'step-3',
            title: 'Concept Development',
            description: 'Refine and develop selected concepts',
            duration: '60 minutes',
            tools: ['Design Software', 'Prototyping Tools'],
            inputs: ['Selected Concepts', 'Design Guidelines'],
            outputs: ['Detailed Concepts', 'Prototypes']
          },
          {
            id: 'step-4',
            title: 'Review & Iteration',
            description: 'Gather feedback and refine concepts',
            duration: '30 minutes',
            tools: ['Collaboration Tools', 'Feedback System'],
            inputs: ['Prototypes', 'Stakeholder Feedback'],
            outputs: ['Final Concept', 'Implementation Plan']
          }
        ]
      }

      setWorkflows(prev => [workflow, ...prev])
      onWorkflowCreated?.(workflow)

      addNotification({
        type: 'success',
        title: 'Workflow Created',
        message: `Creative workflow for "${workflowType}" has been generated`,
        duration: 3000,
      })

    } catch (error) {
      addNotification({
        type: 'error',
        title: 'Creation Failed',
        message: 'Failed to create workflow. Please try again.',
        duration: 5000,
      })
    } finally {
      setIsGenerating(false)
    }
  }

  // ===== RENDER HELPERS =====

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'easy': return 'text-green-600 bg-green-100'
      case 'medium': return 'text-yellow-600 bg-yellow-100'
      case 'hard': return 'text-red-600 bg-red-100'
      default: return 'text-gray-600 bg-gray-100'
    }
  }

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'design': return <Palette className="h-4 w-4" />
      case 'content': return <FileText className="h-4 w-4" />
      case 'strategy': return <Brain className="h-4 w-4" />
      case 'innovation': return <Rocket className="h-4 w-4" />
      default: return <Lightbulb className="h-4 w-4" />
    }
  }

  return (
    <div className={`bg-white rounded-lg shadow-lg border border-gray-200 ${className}`}>
      {/* Header */}
      <div className="bg-gradient-to-r from-pink-500 to-purple-600 px-6 py-4 rounded-t-lg">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Wand2 className="h-6 w-6 text-white" />
            <h2 className="text-xl font-semibold text-white">Creative AI Studio</h2>
            {isGenerating && (
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                className="flex items-center space-x-1 text-white"
              >
                <Sparkles className="h-4 w-4" />
                <span className="text-sm">Generating...</span>
              </motion.div>
            )}
          </div>
          
          <div className="flex items-center space-x-2">
            <select
              value={selectedCategory}
              onChange={(e) => setSelectedCategory(e.target.value as any)}
              className="px-3 py-1 bg-white bg-opacity-20 text-white rounded text-sm border border-white border-opacity-30"
            >
              <option value="design">Design</option>
              <option value="content">Content</option>
              <option value="strategy">Strategy</option>
              <option value="innovation">Innovation</option>
            </select>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-200">
        {[
          { id: 'ideas', label: 'Idea Generation', icon: <Lightbulb className="h-4 w-4" /> },
          { id: 'moodboard', label: 'Mood Board', icon: <Palette className="h-4 w-4" /> },
          { id: 'workflow', label: 'Workflow', icon: <Layers className="h-4 w-4" /> },
          { id: 'inspiration', label: 'Inspiration', icon: <Star className="h-4 w-4" /> },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={`flex items-center space-x-2 px-6 py-3 text-sm font-medium transition-colors ${
              activeTab === tab.id
                ? 'text-purple-600 border-b-2 border-purple-600'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            {tab.icon}
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="p-6">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.2 }}
          >
            {/* Ideas Tab */}
            {activeTab === 'ideas' && (
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-gray-900">AI Idea Generation</h3>
                  <button
                    onClick={generateIdeas}
                    disabled={isGenerating}
                    className="flex items-center space-x-2 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors disabled:opacity-50"
                  >
                    <Zap className="h-4 w-4" />
                    <span>Generate Ideas</span>
                  </button>
                </div>

                <div className="grid gap-4">
                  {generatedIdeas.map((idea) => (
                    <motion.div
                      key={idea.id}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow"
                    >
                      <div className="flex items-start justify-between mb-3">
                        <div className="flex items-center space-x-2">
                          {getCategoryIcon(idea.category)}
                          <h4 className="font-semibold text-gray-900">{idea.title}</h4>
                        </div>
                        <div className="flex items-center space-x-2">
                          <span className={`px-2 py-1 text-xs rounded-full ${getDifficultyColor(idea.difficulty)}`}>
                            {idea.difficulty}
                          </span>
                          <span className="text-sm text-gray-500">
                            {Math.round(idea.confidence * 100)}%
                          </span>
                        </div>
                      </div>

                      <p className="text-gray-600 mb-3">{idea.description}</p>

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                        <div>
                          <span className="font-medium text-gray-700">Implementation:</span>
                          <p className="text-gray-600">{idea.estimatedTime}</p>
                        </div>
                        <div>
                          <span className="font-medium text-gray-700">Tags:</span>
                          <div className="flex flex-wrap gap-1 mt-1">
                            {idea.tags.map((tag, index) => (
                              <span key={index} className="px-2 py-1 bg-purple-100 text-purple-700 text-xs rounded">
                                {tag}
                              </span>
                            ))}
                          </div>
                        </div>
                      </div>

                      <div className="mt-3 flex justify-end space-x-2">
                        <button className="p-2 text-gray-400 hover:text-red-500 transition-colors">
                          <Heart className="h-4 w-4" />
                        </button>
                        <button className="p-2 text-gray-400 hover:text-blue-500 transition-colors">
                          <Share2 className="h-4 w-4" />
                        </button>
                        <button className="p-2 text-gray-400 hover:text-green-500 transition-colors">
                          <Download className="h-4 w-4" />
                        </button>
                      </div>
                    </motion.div>
                  ))}
                </div>

                {generatedIdeas.length === 0 && (
                  <div className="text-center py-12 text-gray-500">
                    <Lightbulb className="h-12 w-12 mx-auto mb-4 text-gray-300" />
                    <p>No ideas generated yet. Click "Generate Ideas" to start!</p>
                  </div>
                )}
              </div>
            )}

            {/* Mood Board Tab */}
            {activeTab === 'moodboard' && (
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-gray-900">Inspiration Mood Board</h3>
                  <div className="flex items-center space-x-2">
                    <input
                      type="text"
                      placeholder="Enter theme (e.g., 'futuristic AI')"
                      value={inspirationPrompt}
                      onChange={(e) => setInspirationPrompt(e.target.value)}
                      className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                    />
                    <button
                      onClick={() => generateMoodBoard(inspirationPrompt || 'Creative AI')}
                      disabled={isGenerating}
                      className="flex items-center space-x-2 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors disabled:opacity-50"
                    >
                      <Palette className="h-4 w-4" />
                      <span>Generate</span>
                    </button>
                  </div>
                </div>

                {currentMoodBoard ? (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="space-y-4"
                  >
                    <div className="bg-gray-50 rounded-lg p-4">
                      <h4 className="font-semibold text-gray-900 mb-2">{currentMoodBoard.title}</h4>
                      <p className="text-gray-600 mb-4">{currentMoodBoard.mood}</p>
                      
                      {/* Color Palette */}
                      <div className="mb-4">
                        <span className="text-sm font-medium text-gray-700 mb-2 block">Color Palette:</span>
                        <div className="flex space-x-2">
                          {currentMoodBoard.colors.map((color, index) => (
                            <div
                              key={index}
                              className="w-12 h-12 rounded-lg shadow-sm border border-gray-200"
                              style={{ backgroundColor: color }}
                              title={color}
                            />
                          ))}
                        </div>
                      </div>

                      {/* Keywords */}
                      <div className="mb-4">
                        <span className="text-sm font-medium text-gray-700 mb-2 block">Keywords:</span>
                        <div className="flex flex-wrap gap-2">
                          {currentMoodBoard.keywords.map((keyword, index) => (
                            <span key={index} className="px-3 py-1 bg-purple-100 text-purple-700 text-sm rounded-full">
                              {keyword}
                            </span>
                          ))}
                        </div>
                      </div>

                      {/* Style Description */}
                      <div>
                        <span className="text-sm font-medium text-gray-700 mb-2 block">Style:</span>
                        <p className="text-gray-600">{currentMoodBoard.style}</p>
                      </div>
                    </div>
                  </motion.div>
                ) : (
                  <div className="text-center py-12 text-gray-500">
                    <Palette className="h-12 w-12 mx-auto mb-4 text-gray-300" />
                    <p>No mood board created yet. Enter a theme and generate one!</p>
                  </div>
                )}
              </div>
            )}

            {/* Workflow Tab */}
            {activeTab === 'workflow' && (
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-gray-900">Creative Workflow Automation</h3>
                  <div className="flex space-x-2">
                    <button
                      onClick={() => createWorkflow('Design')}
                      disabled={isGenerating}
                      className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors disabled:opacity-50"
                    >
                      Design Workflow
                    </button>
                    <button
                      onClick={() => createWorkflow('Content')}
                      disabled={isGenerating}
                      className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50"
                    >
                      Content Workflow
                    </button>
                  </div>
                </div>

                <div className="space-y-4">
                  {workflows.map((workflow) => (
                    <motion.div
                      key={workflow.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="border border-gray-200 rounded-lg p-4"
                    >
                      <div className="flex items-center justify-between mb-4">
                        <h4 className="font-semibold text-gray-900">{workflow.name}</h4>
                        <span className="text-sm text-gray-500">{workflow.estimatedTime}</span>
                      </div>

                      <div className="space-y-3">
                        {workflow.steps.map((step, index) => (
                          <div key={step.id} className="flex items-start space-x-3">
                            <div className="flex-shrink-0 w-8 h-8 bg-purple-100 text-purple-600 rounded-full flex items-center justify-center text-sm font-medium">
                              {index + 1}
                            </div>
                            <div className="flex-1">
                              <h5 className="font-medium text-gray-900">{step.title}</h5>
                              <p className="text-sm text-gray-600">{step.description}</p>
                              <div className="flex items-center space-x-4 mt-1 text-xs text-gray-500">
                                <span>Duration: {step.duration}</span>
                                <span>Tools: {step.tools.join(', ')}</span>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </motion.div>
                  ))}
                </div>

                {workflows.length === 0 && (
                  <div className="text-center py-12 text-gray-500">
                    <Layers className="h-12 w-12 mx-auto mb-4 text-gray-300" />
                    <p>No workflows created yet. Generate a creative workflow to get started!</p>
                  </div>
                )}
              </div>
            )}

            {/* Inspiration Tab */}
            {activeTab === 'inspiration' && (
              <div className="space-y-6">
                <h3 className="text-lg font-semibold text-gray-900">Creative Inspiration</h3>
                
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {[
                    { title: 'Design Trends 2024', category: 'Design', icon: <Palette className="h-5 w-5" /> },
                    { title: 'AI Innovation Patterns', category: 'Technology', icon: <Brain className="h-5 w-5" /> },
                    { title: 'User Experience Evolution', category: 'UX', icon: <Star className="h-5 w-5" /> },
                    { title: 'Creative Collaboration', category: 'Process', icon: <Heart className="h-5 w-5" /> },
                    { title: 'Future Interfaces', category: 'Innovation', icon: <Rocket className="h-5 w-5" /> },
                    { title: 'Emotional Design', category: 'Psychology', icon: <Sparkles className="h-5 w-5" /> },
                  ].map((item, index) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.1 }}
                      className="bg-gradient-to-br from-purple-50 to-pink-50 border border-purple-200 rounded-lg p-4 hover:shadow-md transition-shadow cursor-pointer"
                    >
                      <div className="flex items-center space-x-2 mb-2">
                        <div className="text-purple-600">{item.icon}</div>
                        <span className="text-sm text-purple-600 font-medium">{item.category}</span>
                      </div>
                      <h4 className="font-semibold text-gray-900">{item.title}</h4>
                    </motion.div>
                  ))}
                </div>
              </div>
            )}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  )
}

export default CreativeAI
