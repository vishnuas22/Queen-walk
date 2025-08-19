'use client'

import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  GitBranch, 
  Target, 
  TrendingUp, 
  AlertTriangle,
  CheckCircle,
  XCircle,
  BarChart3,
  Layers,
  Zap,
  Brain,
  Scale,
  Clock,
  DollarSign,
  Users,
  Shield,
  Lightbulb,
  Play,
  Pause
} from 'lucide-react'
import { useUIActions } from '../store'

// ===== TYPES =====

interface DecisionSupportProps {
  decision?: string
  onDecisionMade?: (decision: DecisionPath) => void
  className?: string
}

interface DecisionNode {
  id: string
  type: 'decision' | 'criteria' | 'option' | 'outcome'
  title: string
  description: string
  weight: number
  score?: number
  confidence: number
  children: string[]
  parent?: string
  position: { x: number; y: number }
  metadata: {
    cost?: number
    time?: string
    risk?: 'low' | 'medium' | 'high'
    impact?: 'low' | 'medium' | 'high'
    stakeholders?: string[]
  }
}

interface DecisionCriteria {
  id: string
  name: string
  weight: number
  description: string
  type: 'cost' | 'time' | 'quality' | 'risk' | 'impact' | 'feasibility'
}

interface DecisionOption {
  id: string
  title: string
  description: string
  scores: { [criteriaId: string]: number }
  totalScore: number
  pros: string[]
  cons: string[]
  risks: string[]
  opportunities: string[]
  implementation: {
    timeline: string
    resources: string[]
    dependencies: string[]
    milestones: string[]
  }
}

interface DecisionPath {
  id: string
  selectedOption: DecisionOption
  reasoning: string[]
  confidence: number
  alternatives: DecisionOption[]
  nextSteps: string[]
}

interface ScenarioAnalysis {
  id: string
  name: string
  probability: number
  impact: 'positive' | 'negative' | 'neutral'
  description: string
  mitigations?: string[]
}

// ===== DECISION SUPPORT COMPONENT =====

export const DecisionSupport: React.FC<DecisionSupportProps> = ({
  decision = "Should we implement advanced AI features in the next quarter?",
  onDecisionMade,
  className = ''
}) => {
  const [activeTab, setActiveTab] = useState<'tree' | 'matrix' | 'scenarios' | 'analysis'>('tree')
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [decisionNodes, setDecisionNodes] = useState<DecisionNode[]>([])
  const [criteria, setCriteria] = useState<DecisionCriteria[]>([])
  const [options, setOptions] = useState<DecisionOption[]>([])
  const [scenarios, setScenarios] = useState<ScenarioAnalysis[]>([])
  const [selectedOption, setSelectedOption] = useState<string | null>(null)
  const [analysisComplete, setAnalysisComplete] = useState(false)

  const { addNotification } = useUIActions()

  // Initialize decision analysis
  useEffect(() => {
    initializeDecisionAnalysis()
  }, [decision])

  const initializeDecisionAnalysis = async () => {
    setIsAnalyzing(true)

    // Initialize criteria
    const decisionCriteria: DecisionCriteria[] = [
      {
        id: 'cost',
        name: 'Implementation Cost',
        weight: 0.25,
        description: 'Total cost including development, infrastructure, and maintenance',
        type: 'cost'
      },
      {
        id: 'time',
        name: 'Time to Market',
        weight: 0.20,
        description: 'How quickly can this be implemented and delivered',
        type: 'time'
      },
      {
        id: 'impact',
        name: 'Business Impact',
        weight: 0.30,
        description: 'Expected positive impact on business metrics',
        type: 'impact'
      },
      {
        id: 'risk',
        name: 'Implementation Risk',
        weight: 0.15,
        description: 'Technical and business risks involved',
        type: 'risk'
      },
      {
        id: 'feasibility',
        name: 'Technical Feasibility',
        weight: 0.10,
        description: 'How technically feasible is the implementation',
        type: 'feasibility'
      }
    ]

    // Initialize options
    const decisionOptions: DecisionOption[] = [
      {
        id: 'option-1',
        title: 'Full AI Feature Suite',
        description: 'Implement comprehensive AI features including advanced reasoning, creative tools, and decision support',
        scores: {
          cost: 3, // Lower score = higher cost
          time: 2, // Lower score = longer time
          impact: 9, // Higher score = better impact
          risk: 4, // Lower score = higher risk
          feasibility: 7 // Higher score = more feasible
        },
        totalScore: 0,
        pros: [
          'Comprehensive feature set',
          'Strong competitive advantage',
          'High user engagement potential',
          'Future-proof architecture'
        ],
        cons: [
          'High development cost',
          'Extended timeline',
          'Complex implementation',
          'Resource intensive'
        ],
        risks: [
          'Technical complexity',
          'Market timing',
          'Resource allocation',
          'User adoption'
        ],
        opportunities: [
          'Market leadership',
          'Premium pricing',
          'User retention',
          'Platform differentiation'
        ],
        implementation: {
          timeline: '6-8 months',
          resources: ['Senior developers', 'AI specialists', 'UX designers', 'QA engineers'],
          dependencies: ['AI infrastructure', 'Data pipeline', 'Security framework'],
          milestones: ['MVP completion', 'Beta testing', 'Performance optimization', 'Production release']
        }
      },
      {
        id: 'option-2',
        title: 'Phased AI Implementation',
        description: 'Implement AI features in phases, starting with core functionality and expanding over time',
        scores: {
          cost: 6,
          time: 7,
          impact: 7,
          risk: 7,
          feasibility: 8
        },
        totalScore: 0,
        pros: [
          'Manageable implementation',
          'Faster initial delivery',
          'Lower initial investment',
          'Iterative improvement'
        ],
        cons: [
          'Limited initial impact',
          'Fragmented user experience',
          'Potential technical debt',
          'Competitive disadvantage'
        ],
        risks: [
          'Feature inconsistency',
          'User confusion',
          'Technical debt accumulation',
          'Delayed competitive advantage'
        ],
        opportunities: [
          'User feedback integration',
          'Gradual market education',
          'Resource optimization',
          'Risk mitigation'
        ],
        implementation: {
          timeline: '3-4 months per phase',
          resources: ['Core development team', 'Part-time AI specialist', 'UX designer'],
          dependencies: ['Basic AI infrastructure', 'User feedback system'],
          milestones: ['Phase 1 release', 'User feedback analysis', 'Phase 2 planning', 'Iterative releases']
        }
      },
      {
        id: 'option-3',
        title: 'AI Integration Partnership',
        description: 'Partner with existing AI providers to integrate their capabilities into our platform',
        scores: {
          cost: 8,
          time: 9,
          impact: 6,
          risk: 8,
          feasibility: 9
        },
        totalScore: 0,
        pros: [
          'Rapid implementation',
          'Lower development cost',
          'Proven technology',
          'Reduced technical risk'
        ],
        cons: [
          'Limited customization',
          'Vendor dependency',
          'Ongoing licensing costs',
          'Less differentiation'
        ],
        risks: [
          'Vendor lock-in',
          'Service availability',
          'Cost escalation',
          'Limited control'
        ],
        opportunities: [
          'Quick market entry',
          'Focus on core competencies',
          'Reduced development overhead',
          'Faster user validation'
        ],
        implementation: {
          timeline: '2-3 months',
          resources: ['Integration team', 'API developers', 'QA engineers'],
          dependencies: ['Vendor selection', 'API documentation', 'Security review'],
          milestones: ['Vendor selection', 'Integration development', 'Testing', 'Launch']
        }
      }
    ]

    // Calculate total scores
    decisionOptions.forEach(option => {
      option.totalScore = decisionCriteria.reduce((total, criteria) => {
        return total + (option.scores[criteria.id] * criteria.weight)
      }, 0)
    })

    // Initialize scenarios
    const scenarioAnalysis: ScenarioAnalysis[] = [
      {
        id: 'scenario-1',
        name: 'Market Leadership',
        probability: 0.7,
        impact: 'positive',
        description: 'AI features become a key differentiator and drive significant user growth',
        mitigations: ['Continuous innovation', 'User feedback integration', 'Performance monitoring']
      },
      {
        id: 'scenario-2',
        name: 'Technical Challenges',
        probability: 0.4,
        impact: 'negative',
        description: 'Implementation faces significant technical hurdles causing delays',
        mitigations: ['Prototype validation', 'Expert consultation', 'Phased rollback plan']
      },
      {
        id: 'scenario-3',
        name: 'Competitive Response',
        probability: 0.6,
        impact: 'neutral',
        description: 'Competitors quickly match AI capabilities, reducing advantage',
        mitigations: ['Continuous innovation', 'Patent protection', 'User experience focus']
      },
      {
        id: 'scenario-4',
        name: 'Resource Constraints',
        probability: 0.3,
        impact: 'negative',
        description: 'Limited resources impact implementation quality or timeline',
        mitigations: ['Resource planning', 'Phased approach', 'External partnerships']
      }
    ]

    // Create decision tree nodes
    const nodes: DecisionNode[] = [
      {
        id: 'root',
        type: 'decision',
        title: decision,
        description: 'Primary decision to be made',
        weight: 1.0,
        confidence: 1.0,
        children: ['criteria-1', 'criteria-2', 'criteria-3'],
        position: { x: 400, y: 100 },
        metadata: {}
      },
      {
        id: 'criteria-1',
        type: 'criteria',
        title: 'Business Impact',
        description: 'Evaluate potential business outcomes',
        weight: 0.4,
        confidence: 0.85,
        children: ['option-1', 'option-2'],
        parent: 'root',
        position: { x: 200, y: 250 },
        metadata: { impact: 'high' }
      },
      {
        id: 'criteria-2',
        type: 'criteria',
        title: 'Implementation Feasibility',
        description: 'Assess technical and resource feasibility',
        weight: 0.35,
        confidence: 0.90,
        children: ['option-2', 'option-3'],
        parent: 'root',
        position: { x: 400, y: 250 },
        metadata: { risk: 'medium' }
      },
      {
        id: 'criteria-3',
        type: 'criteria',
        title: 'Cost-Benefit Analysis',
        description: 'Evaluate financial implications',
        weight: 0.25,
        confidence: 0.80,
        children: ['option-1', 'option-3'],
        parent: 'root',
        position: { x: 600, y: 250 },
        metadata: { cost: 50000 }
      }
    ]

    // Add option nodes
    decisionOptions.forEach((option, index) => {
      nodes.push({
        id: option.id,
        type: 'option',
        title: option.title,
        description: option.description,
        weight: option.totalScore / 10,
        confidence: option.totalScore / 10,
        children: [],
        position: { x: 150 + (index * 200), y: 400 },
        metadata: {
          cost: option.implementation.resources.length * 10000,
          time: option.implementation.timeline,
          risk: option.risks.length > 3 ? 'high' : option.risks.length > 1 ? 'medium' : 'low',
          impact: option.totalScore > 7 ? 'high' : option.totalScore > 5 ? 'medium' : 'low'
        }
      })
    })

    setCriteria(decisionCriteria)
    setOptions(decisionOptions)
    setScenarios(scenarioAnalysis)
    setDecisionNodes(nodes)

    await new Promise(resolve => setTimeout(resolve, 2000))
    setIsAnalyzing(false)
    setAnalysisComplete(true)

    addNotification({
      type: 'success',
      title: 'Decision Analysis Complete',
      message: 'Multi-criteria analysis has been completed with scenario planning',
      duration: 4000,
    })
  }

  // ===== DECISION MAKING =====

  const makeDecision = (optionId: string) => {
    const option = options.find(o => o.id === optionId)
    if (!option) return

    const decisionPath: DecisionPath = {
      id: `decision-${Date.now()}`,
      selectedOption: option,
      reasoning: [
        `Selected based on highest weighted score: ${option.totalScore.toFixed(2)}`,
        `Key advantages: ${option.pros.slice(0, 2).join(', ')}`,
        `Implementation timeline: ${option.implementation.timeline}`,
        `Risk level: ${option.risks.length > 3 ? 'High' : option.risks.length > 1 ? 'Medium' : 'Low'}`
      ],
      confidence: option.totalScore / 10,
      alternatives: options.filter(o => o.id !== optionId),
      nextSteps: [
        'Finalize implementation plan',
        'Allocate required resources',
        'Set up project timeline',
        'Begin development phase'
      ]
    }

    setSelectedOption(optionId)
    onDecisionMade?.(decisionPath)

    addNotification({
      type: 'success',
      title: 'Decision Made',
      message: `Selected: ${option.title}`,
      duration: 3000,
    })
  }

  // ===== RENDER HELPERS =====

  const getScoreColor = (score: number) => {
    if (score >= 8) return 'text-green-600'
    if (score >= 6) return 'text-yellow-600'
    return 'text-red-600'
  }

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'low': return 'text-green-600 bg-green-100'
      case 'medium': return 'text-yellow-600 bg-yellow-100'
      case 'high': return 'text-red-600 bg-red-100'
      default: return 'text-gray-600 bg-gray-100'
    }
  }

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'positive': return 'text-green-600'
      case 'negative': return 'text-red-600'
      case 'neutral': return 'text-gray-600'
      default: return 'text-gray-600'
    }
  }

  return (
    <div className={`bg-white rounded-lg shadow-lg border border-gray-200 ${className}`}>
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 to-indigo-600 px-6 py-4 rounded-t-lg">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <GitBranch className="h-6 w-6 text-white" />
            <h2 className="text-xl font-semibold text-white">Decision Support System</h2>
            {isAnalyzing && (
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                className="flex items-center space-x-1 text-white"
              >
                <Brain className="h-4 w-4" />
                <span className="text-sm">Analyzing...</span>
              </motion.div>
            )}
          </div>
          
          {analysisComplete && (
            <div className="flex items-center space-x-2 text-white text-sm">
              <CheckCircle className="h-4 w-4" />
              <span>Analysis Complete</span>
            </div>
          )}
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-200">
        {[
          { id: 'tree', label: 'Decision Tree', icon: <GitBranch className="h-4 w-4" /> },
          { id: 'matrix', label: 'Decision Matrix', icon: <BarChart3 className="h-4 w-4" /> },
          { id: 'scenarios', label: 'Scenarios', icon: <Layers className="h-4 w-4" /> },
          { id: 'analysis', label: 'Analysis', icon: <Target className="h-4 w-4" /> },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={`flex items-center space-x-2 px-6 py-3 text-sm font-medium transition-colors ${
              activeTab === tab.id
                ? 'text-blue-600 border-b-2 border-blue-600'
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
        {isAnalyzing ? (
          <div className="flex items-center justify-center py-12">
            <div className="text-center">
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                className="mx-auto mb-4"
              >
                <Brain className="h-12 w-12 text-blue-600" />
              </motion.div>
              <p className="text-lg font-medium text-gray-900">Analyzing Decision Options</p>
              <p className="text-gray-600">Evaluating criteria, risks, and scenarios...</p>
            </div>
          </div>
        ) : (
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.2 }}
            >
              {/* Decision Tree Tab */}
              {activeTab === 'tree' && (
                <div className="space-y-6">
                  <h3 className="text-lg font-semibold text-gray-900">Decision Tree Visualization</h3>
                  
                  <div className="relative bg-gray-50 rounded-lg p-6 h-96 overflow-hidden">
                    {/* Decision Nodes */}
                    <AnimatePresence>
                      {decisionNodes.map((node) => (
                        <motion.div
                          key={node.id}
                          initial={{ opacity: 0, scale: 0 }}
                          animate={{ opacity: 1, scale: 1 }}
                          className="absolute"
                          style={{
                            left: `${(node.position.x / 800) * 100}%`,
                            top: `${(node.position.y / 500) * 100}%`,
                            transform: 'translate(-50%, -50%)'
                          }}
                        >
                          <div
                            className={`p-3 rounded-lg border-2 shadow-sm cursor-pointer transition-all hover:shadow-md ${
                              node.type === 'decision' ? 'bg-blue-50 border-blue-300' :
                              node.type === 'criteria' ? 'bg-yellow-50 border-yellow-300' :
                              node.type === 'option' ? 'bg-green-50 border-green-300' :
                              'bg-gray-50 border-gray-300'
                            } ${selectedOption === node.id ? 'ring-2 ring-indigo-500' : ''}`}
                            onClick={() => node.type === 'option' && makeDecision(node.id)}
                          >
                            <div className="text-xs font-medium text-gray-900 mb-1">
                              {node.title}
                            </div>
                            <div className="text-xs text-gray-600">
                              Weight: {Math.round(node.weight * 100)}%
                            </div>
                            {node.metadata.cost && (
                              <div className="text-xs text-gray-500">
                                ${node.metadata.cost.toLocaleString()}
                              </div>
                            )}
                          </div>
                        </motion.div>
                      ))}
                    </AnimatePresence>

                    {/* Connection Lines */}
                    <svg className="absolute inset-0 w-full h-full pointer-events-none">
                      {decisionNodes.map((node) =>
                        node.children.map((childId) => {
                          const childNode = decisionNodes.find(n => n.id === childId)
                          if (!childNode) return null
                          
                          return (
                            <motion.line
                              key={`${node.id}-${childId}`}
                              initial={{ pathLength: 0 }}
                              animate={{ pathLength: 1 }}
                              transition={{ duration: 1, delay: 0.5 }}
                              x1={`${(node.position.x / 800) * 100}%`}
                              y1={`${(node.position.y / 500) * 100}%`}
                              x2={`${(childNode.position.x / 800) * 100}%`}
                              y2={`${(childNode.position.y / 500) * 100}%`}
                              stroke="#6b7280"
                              strokeWidth="2"
                              markerEnd="url(#arrowhead)"
                            />
                          )
                        })
                      )}
                      
                      {/* Arrow marker definition */}
                      <defs>
                        <marker
                          id="arrowhead"
                          markerWidth="10"
                          markerHeight="7"
                          refX="9"
                          refY="3.5"
                          orient="auto"
                        >
                          <polygon
                            points="0 0, 10 3.5, 0 7"
                            fill="#6b7280"
                          />
                        </marker>
                      </defs>
                    </svg>
                  </div>
                </div>
              )}

              {/* Decision Matrix Tab */}
              {activeTab === 'matrix' && (
                <div className="space-y-6">
                  <h3 className="text-lg font-semibold text-gray-900">Multi-Criteria Decision Matrix</h3>
                  
                  <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-gray-200">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Options
                          </th>
                          {criteria.map((criterion) => (
                            <th key={criterion.id} className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                              {criterion.name}
                              <div className="text-xs text-gray-400 normal-case">
                                Weight: {Math.round(criterion.weight * 100)}%
                              </div>
                            </th>
                          ))}
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Total Score
                          </th>
                        </tr>
                      </thead>
                      <tbody className="bg-white divide-y divide-gray-200">
                        {options.map((option) => (
                          <tr
                            key={option.id}
                            className={`hover:bg-gray-50 cursor-pointer ${
                              selectedOption === option.id ? 'bg-blue-50' : ''
                            }`}
                            onClick={() => makeDecision(option.id)}
                          >
                            <td className="px-6 py-4 whitespace-nowrap">
                              <div className="font-medium text-gray-900">{option.title}</div>
                              <div className="text-sm text-gray-500">{option.description}</div>
                            </td>
                            {criteria.map((criterion) => (
                              <td key={criterion.id} className="px-6 py-4 whitespace-nowrap">
                                <div className={`text-sm font-medium ${getScoreColor(option.scores[criterion.id])}`}>
                                  {option.scores[criterion.id]}/10
                                </div>
                                <div className="text-xs text-gray-500">
                                  Weighted: {(option.scores[criterion.id] * criterion.weight).toFixed(1)}
                                </div>
                              </td>
                            ))}
                            <td className="px-6 py-4 whitespace-nowrap">
                              <div className={`text-lg font-bold ${getScoreColor(option.totalScore)}`}>
                                {option.totalScore.toFixed(1)}
                              </div>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* Scenarios Tab */}
              {activeTab === 'scenarios' && (
                <div className="space-y-6">
                  <h3 className="text-lg font-semibold text-gray-900">Scenario Analysis</h3>
                  
                  <div className="grid gap-4">
                    {scenarios.map((scenario) => (
                      <motion.div
                        key={scenario.id}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        className="border border-gray-200 rounded-lg p-4"
                      >
                        <div className="flex items-center justify-between mb-3">
                          <h4 className="font-semibold text-gray-900">{scenario.name}</h4>
                          <div className="flex items-center space-x-2">
                            <span className={`text-sm ${getImpactColor(scenario.impact)}`}>
                              {scenario.impact}
                            </span>
                            <span className="text-sm text-gray-500">
                              {Math.round(scenario.probability * 100)}% probability
                            </span>
                          </div>
                        </div>
                        
                        <p className="text-gray-600 mb-3">{scenario.description}</p>
                        
                        {scenario.mitigations && (
                          <div>
                            <span className="text-sm font-medium text-gray-700">Mitigations:</span>
                            <ul className="text-sm text-gray-600 mt-1">
                              {scenario.mitigations.map((mitigation, index) => (
                                <li key={index}>• {mitigation}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                        
                        {/* Probability bar */}
                        <div className="mt-3">
                          <div className="flex items-center justify-between text-xs text-gray-500 mb-1">
                            <span>Probability</span>
                            <span>{Math.round(scenario.probability * 100)}%</span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div
                              className={`h-2 rounded-full ${
                                scenario.impact === 'positive' ? 'bg-green-500' :
                                scenario.impact === 'negative' ? 'bg-red-500' : 'bg-gray-500'
                              }`}
                              style={{ width: `${scenario.probability * 100}%` }}
                            />
                          </div>
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </div>
              )}

              {/* Analysis Tab */}
              {activeTab === 'analysis' && (
                <div className="space-y-6">
                  <h3 className="text-lg font-semibold text-gray-900">Decision Analysis Summary</h3>
                  
                  {selectedOption ? (
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="bg-green-50 border border-green-200 rounded-lg p-6"
                    >
                      <div className="flex items-center space-x-2 mb-4">
                        <CheckCircle className="h-6 w-6 text-green-600" />
                        <h4 className="text-lg font-semibold text-green-900">Decision Made</h4>
                      </div>
                      
                      {(() => {
                        const option = options.find(o => o.id === selectedOption)
                        if (!option) return null
                        
                        return (
                          <div className="space-y-4">
                            <div>
                              <h5 className="font-medium text-green-900">{option.title}</h5>
                              <p className="text-green-700">{option.description}</p>
                            </div>
                            
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                              <div>
                                <span className="font-medium text-green-900">Key Advantages:</span>
                                <ul className="text-green-700 mt-1">
                                  {option.pros.slice(0, 3).map((pro, index) => (
                                    <li key={index}>• {pro}</li>
                                  ))}
                                </ul>
                              </div>
                              
                              <div>
                                <span className="font-medium text-green-900">Implementation:</span>
                                <div className="text-green-700 mt-1">
                                  <p>Timeline: {option.implementation.timeline}</p>
                                  <p>Resources: {option.implementation.resources.length} team members</p>
                                </div>
                              </div>
                            </div>
                            
                            <div className="flex items-center space-x-4 text-sm">
                              <div className="flex items-center space-x-1">
                                <Target className="h-4 w-4 text-green-600" />
                                <span className="text-green-700">Score: {option.totalScore.toFixed(1)}/10</span>
                              </div>
                              <div className="flex items-center space-x-1">
                                <Scale className="h-4 w-4 text-green-600" />
                                <span className="text-green-700">Confidence: {Math.round(option.totalScore * 10)}%</span>
                              </div>
                            </div>
                          </div>
                        )
                      })()}
                    </motion.div>
                  ) : (
                    <div className="text-center py-8">
                      <Target className="h-12 w-12 text-gray-300 mx-auto mb-4" />
                      <p className="text-gray-500">Select an option from the Decision Matrix to see detailed analysis</p>
                    </div>
                  )}
                  
                  {/* Recommendation */}
                  {!selectedOption && options.length > 0 && (
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="bg-blue-50 border border-blue-200 rounded-lg p-6"
                    >
                      <div className="flex items-center space-x-2 mb-4">
                        <Lightbulb className="h-6 w-6 text-blue-600" />
                        <h4 className="text-lg font-semibold text-blue-900">AI Recommendation</h4>
                      </div>
                      
                      {(() => {
                        const topOption = options.reduce((prev, current) => 
                          prev.totalScore > current.totalScore ? prev : current
                        )
                        
                        return (
                          <div className="space-y-3">
                            <p className="text-blue-700">
                              Based on the multi-criteria analysis, <strong>{topOption.title}</strong> is recommended 
                              with a score of <strong>{topOption.totalScore.toFixed(1)}/10</strong>.
                            </p>
                            
                            <div className="text-sm text-blue-600">
                              <p>Key factors supporting this recommendation:</p>
                              <ul className="mt-1 ml-4">
                                {topOption.pros.slice(0, 2).map((pro, index) => (
                                  <li key={index}>• {pro}</li>
                                ))}
                              </ul>
                            </div>
                            
                            <button
                              onClick={() => makeDecision(topOption.id)}
                              className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                            >
                              Accept Recommendation
                            </button>
                          </div>
                        )
                      })()}
                    </motion.div>
                  )}
                </div>
              )}
            </motion.div>
          </AnimatePresence>
        )}
      </div>
    </div>
  )
}

// ===== AI REASONING DISPLAY COMPONENT =====

interface AIReasoningProps {
  reasoning: ReasoningStep[]
  confidence: number
  alternatives: string[]
  onStepClick?: (step: ReasoningStep) => void
  className?: string
}

interface ReasoningStep {
  id: string
  title: string
  description: string
  type: 'analysis' | 'inference' | 'evaluation' | 'conclusion'
  confidence: number
  evidence: string[]
  assumptions: string[]
  alternatives: string[]
  duration: number // in milliseconds
}

export const AIReasoning: React.FC<AIReasoningProps> = ({
  reasoning,
  confidence,
  alternatives,
  onStepClick,
  className = ''
}) => {
  const [currentStep, setCurrentStep] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [playbackSpeed, setPlaybackSpeed] = useState(1)

  useEffect(() => {
    if (isPlaying && currentStep < reasoning.length - 1) {
      const timer = setTimeout(() => {
        setCurrentStep(prev => prev + 1)
      }, reasoning[currentStep]?.duration / playbackSpeed || 1000)

      return () => clearTimeout(timer)
    } else if (currentStep >= reasoning.length - 1) {
      setIsPlaying(false)
    }
  }, [isPlaying, currentStep, reasoning, playbackSpeed])

  const getStepIcon = (type: string) => {
    switch (type) {
      case 'analysis': return <BarChart3 className="h-4 w-4" />
      case 'inference': return <Brain className="h-4 w-4" />
      case 'evaluation': return <Scale className="h-4 w-4" />
      case 'conclusion': return <Target className="h-4 w-4" />
      default: return <Lightbulb className="h-4 w-4" />
    }
  }

  const getStepColor = (type: string) => {
    switch (type) {
      case 'analysis': return 'text-blue-600 bg-blue-100'
      case 'inference': return 'text-purple-600 bg-purple-100'
      case 'evaluation': return 'text-yellow-600 bg-yellow-100'
      case 'conclusion': return 'text-green-600 bg-green-100'
      default: return 'text-gray-600 bg-gray-100'
    }
  }

  return (
    <div className={`bg-white rounded-lg shadow-lg border border-gray-200 ${className}`}>
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-600 to-indigo-600 px-6 py-4 rounded-t-lg">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Brain className="h-6 w-6 text-white" />
            <h2 className="text-xl font-semibold text-white">AI Reasoning Process</h2>
          </div>

          <div className="flex items-center space-x-4 text-white">
            <div className="text-sm">
              Confidence: {Math.round(confidence * 100)}%
            </div>

            <div className="flex items-center space-x-2">
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className="p-1 hover:bg-white hover:bg-opacity-20 rounded"
              >
                {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
              </button>

              <select
                value={playbackSpeed}
                onChange={(e) => setPlaybackSpeed(Number(e.target.value))}
                className="px-2 py-1 bg-white bg-opacity-20 text-white rounded text-sm border border-white border-opacity-30"
              >
                <option value={0.5}>0.5x</option>
                <option value={1}>1x</option>
                <option value={2}>2x</option>
                <option value={4}>4x</option>
              </select>
            </div>
          </div>
        </div>
      </div>

      {/* Reasoning Steps */}
      <div className="p-6">
        <div className="space-y-4">
          {reasoning.map((step, index) => (
            <motion.div
              key={step.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{
                opacity: index <= currentStep ? 1 : 0.3,
                x: 0,
                scale: index === currentStep ? 1.02 : 1
              }}
              transition={{ duration: 0.3 }}
              className={`border rounded-lg p-4 cursor-pointer transition-all ${
                index === currentStep
                  ? 'border-indigo-500 bg-indigo-50 shadow-md'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
              onClick={() => {
                setCurrentStep(index)
                onStepClick?.(step)
              }}
            >
              <div className="flex items-start space-x-3">
                <div className={`flex-shrink-0 p-2 rounded-lg ${getStepColor(step.type)}`}>
                  {getStepIcon(step.type)}
                </div>

                <div className="flex-1">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-semibold text-gray-900">{step.title}</h4>
                    <div className="flex items-center space-x-2">
                      <span className="text-sm text-gray-500">
                        Step {index + 1}/{reasoning.length}
                      </span>
                      <span className={`text-sm ${getStepColor(step.confidence > 0.8 ? 'analysis' : 'warning')}`}>
                        {Math.round(step.confidence * 100)}%
                      </span>
                    </div>
                  </div>

                  <p className="text-gray-600 mb-3">{step.description}</p>

                  {/* Evidence */}
                  {step.evidence.length > 0 && (
                    <div className="mb-3">
                      <span className="text-sm font-medium text-gray-700">Evidence:</span>
                      <ul className="text-sm text-gray-600 mt-1">
                        {step.evidence.map((evidence, evidenceIndex) => (
                          <li key={evidenceIndex}>• {evidence}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Assumptions */}
                  {step.assumptions.length > 0 && (
                    <div className="mb-3">
                      <span className="text-sm font-medium text-gray-700">Assumptions:</span>
                      <ul className="text-sm text-gray-600 mt-1">
                        {step.assumptions.map((assumption, assumptionIndex) => (
                          <li key={assumptionIndex}>• {assumption}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Alternatives */}
                  {step.alternatives.length > 0 && (
                    <div>
                      <span className="text-sm font-medium text-gray-700">Alternative paths:</span>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {step.alternatives.map((alternative, altIndex) => (
                          <span
                            key={altIndex}
                            className="px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded"
                          >
                            {alternative}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Progress indicator */}
              {index === currentStep && isPlaying && (
                <div className="mt-3">
                  <div className="w-full bg-gray-200 rounded-full h-1">
                    <motion.div
                      className="bg-indigo-600 h-1 rounded-full"
                      initial={{ width: '0%' }}
                      animate={{ width: '100%' }}
                      transition={{ duration: (step.duration / playbackSpeed) / 1000 }}
                    />
                  </div>
                </div>
              )}
            </motion.div>
          ))}
        </div>

        {/* Overall Confidence */}
        <div className="mt-6 p-4 bg-gray-50 rounded-lg">
          <div className="flex items-center justify-between mb-2">
            <span className="font-medium text-gray-900">Overall Reasoning Confidence</span>
            <span className={`font-bold ${getStepColor(confidence > 0.8 ? 'analysis' : 'warning')}`}>
              {Math.round(confidence * 100)}%
            </span>
          </div>

          <div className="w-full bg-gray-200 rounded-full h-2">
            <motion.div
              className={`h-2 rounded-full ${
                confidence >= 0.8 ? 'bg-green-500' :
                confidence >= 0.6 ? 'bg-yellow-500' : 'bg-red-500'
              }`}
              initial={{ width: '0%' }}
              animate={{ width: `${confidence * 100}%` }}
              transition={{ duration: 1, delay: 0.5 }}
            />
          </div>

          {alternatives.length > 0 && (
            <div className="mt-3">
              <span className="text-sm font-medium text-gray-700">Alternative reasoning paths:</span>
              <div className="flex flex-wrap gap-2 mt-1">
                {alternatives.map((alternative, index) => (
                  <span
                    key={index}
                    className="px-2 py-1 bg-blue-100 text-blue-700 text-sm rounded"
                  >
                    {alternative}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default DecisionSupport
