"""
Core enumerations for Quantum Intelligence Engine
"""

from enum import Enum, IntEnum


class QuantumLearningMode(Enum):
    """Revolutionary learning modes with quantum intelligence"""
    ADAPTIVE_QUANTUM = "adaptive_quantum"                           # AI-driven adaptive learning
    SOCRATIC_DISCOVERY = "socratic_discovery"                       # Question-based discovery learning
    DEBUG_MASTERY = "debug_mastery"                                 # Knowledge gap identification & fixing
    CHALLENGE_EVOLUTION = "challenge_evolution"                     # Progressive difficulty evolution
    MENTOR_WISDOM = "mentor_wisdom"                                 # Professional mentorship mode
    CREATIVE_SYNTHESIS = "creative_synthesis"                       # Creative learning & analogies
    ANALYTICAL_PRECISION = "analytical_precision"                   # Structured analytical learning
    EMOTIONAL_RESONANCE = "emotional_resonance"                     # Mood-based learning adaptation
    METACOGNITIVE_AWARENESS = "metacognitive_awareness"             # Self-reflection learning
    COLLABORATIVE_INTELLIGENCE = "collaborative_intelligence"       # Group learning mode


class QuantumState(Enum):
    """Quantum learning states representing user's cognitive state"""
    DISCOVERY = "discovery"                       # Exploring new concepts
    CONSOLIDATION = "consolidation"               # Reinforcing understanding
    APPLICATION = "application"                   # Applying knowledge
    SYNTHESIS = "synthesis"                       # Connecting concepts
    MASTERY = "mastery"                          # Achieving expertise
    TRANSFER = "transfer"                        # Applying to new domains
    INNOVATION = "innovation"                    # Creating new knowledge


class IntelligenceLevel(IntEnum):
    """Levels of AI intelligence response"""
    BASIC = 1                                    # Simple explanations
    ENHANCED = 2                                 # Detailed explanations with examples
    ADVANCED = 3                                 # Complex reasoning with multiple perspectives
    EXPERT = 4                                   # Professional-level insights
    QUANTUM = 5                                  # Revolutionary insights with innovation


class LearningStyle(Enum):
    """Learning style preferences"""
    VISUAL = "visual"                            # Visual learner
    AUDITORY = "auditory"                        # Auditory learner
    KINESTHETIC = "kinesthetic"                  # Hands-on learner
    READING_WRITING = "reading_writing"          # Text-based learner
    MULTIMODAL = "multimodal"                    # Multiple modalities
    BALANCED = "balanced"                        # Balanced approach


class EmotionalState(Enum):
    """Emotional states for mood-based adaptation"""
    EXCITED = "excited"                          # High energy, positive
    FOCUSED = "focused"                          # Concentrated, engaged
    CURIOUS = "curious"                          # Inquisitive, exploring
    NEUTRAL = "neutral"                          # Baseline state
    TIRED = "tired"                              # Low energy
    FRUSTRATED = "frustrated"                    # Struggling, needs support
    CONFUSED = "confused"                        # Needs clarification
    CONFIDENT = "confident"                      # Ready for challenges
    ANXIOUS = "anxious"                          # Needs reassurance


class LearningPace(Enum):
    """Learning pace preferences"""
    SLOW = "slow"                                # Deliberate, thorough
    MODERATE = "moderate"                        # Balanced pace
    FAST = "fast"                                # Quick, efficient
    ADAPTIVE = "adaptive"                        # AI-determined optimal pace


class DifficultyLevel(Enum):
    """Content difficulty levels"""
    BEGINNER = "beginner"                        # Basic concepts
    ELEMENTARY = "elementary"                    # Foundational knowledge
    INTERMEDIATE = "intermediate"                # Building complexity
    ADVANCED = "advanced"                        # Complex concepts
    EXPERT = "expert"                            # Expert-level content
    ADAPTIVE = "adaptive"                        # AI-determined difficulty


class ContentType(Enum):
    """Types of learning content"""
    EXPLANATION = "explanation"                  # Conceptual explanations
    EXAMPLE = "example"                          # Practical examples
    EXERCISE = "exercise"                        # Practice exercises
    QUIZ = "quiz"                                # Assessment questions
    PROJECT = "project"                          # Hands-on projects
    DISCUSSION = "discussion"                    # Interactive discussions
    REFLECTION = "reflection"                    # Metacognitive reflection


class InteractionType(Enum):
    """Types of learning interactions"""
    QUESTION = "question"                        # User asking questions
    ANSWER = "answer"                            # Providing answers
    CLARIFICATION = "clarification"              # Seeking clarification
    FEEDBACK = "feedback"                        # Requesting feedback
    EXPLORATION = "exploration"                  # Exploring topics
    PRACTICE = "practice"                        # Practicing skills
    ASSESSMENT = "assessment"                    # Being assessed


class LearningObjective(Enum):
    """Learning objectives and goals"""
    UNDERSTAND = "understand"                    # Comprehension
    APPLY = "apply"                              # Application
    ANALYZE = "analyze"                          # Analysis
    SYNTHESIZE = "synthesize"                    # Synthesis
    EVALUATE = "evaluate"                        # Evaluation
    CREATE = "create"                            # Creation
    REMEMBER = "remember"                        # Recall


class FeedbackType(Enum):
    """Types of feedback"""
    POSITIVE = "positive"                        # Encouraging feedback
    CORRECTIVE = "corrective"                    # Error correction
    SUGGESTIVE = "suggestive"                    # Improvement suggestions
    NEUTRAL = "neutral"                          # Informational feedback
    MOTIVATIONAL = "motivational"                # Motivation boost


class EngagementLevel(Enum):
    """User engagement levels"""
    VERY_LOW = "very_low"                        # Disengaged
    LOW = "low"                                  # Minimal engagement
    MODERATE = "moderate"                        # Average engagement
    HIGH = "high"                                # Highly engaged
    VERY_HIGH = "very_high"                      # Extremely engaged


class CognitiveLoad(Enum):
    """Cognitive load levels"""
    MINIMAL = "minimal"                          # Very light load
    LOW = "low"                                  # Light load
    MODERATE = "moderate"                        # Manageable load
    HIGH = "high"                                # Heavy load
    EXCESSIVE = "excessive"                      # Overwhelming load


class LearningPhase(Enum):
    """Phases of learning process"""
    PREPARATION = "preparation"                  # Getting ready to learn
    ACQUISITION = "acquisition"                  # Initial learning
    ELABORATION = "elaboration"                  # Deepening understanding
    INTEGRATION = "integration"                  # Connecting knowledge
    PRACTICE = "practice"                        # Skill development
    TRANSFER = "transfer"                        # Applying to new contexts
    RETENTION = "retention"                      # Long-term memory


class AssessmentType(Enum):
    """Types of assessments"""
    FORMATIVE = "formative"                      # Ongoing assessment
    SUMMATIVE = "summative"                      # Final assessment
    DIAGNOSTIC = "diagnostic"                    # Identifying gaps
    ADAPTIVE = "adaptive"                        # AI-driven assessment
    PEER = "peer"                                # Peer assessment
    SELF = "self"                                # Self-assessment


class MotivationType(Enum):
    """Types of motivation"""
    INTRINSIC = "intrinsic"                      # Internal motivation
    EXTRINSIC = "extrinsic"                      # External motivation
    ACHIEVEMENT = "achievement"                  # Goal achievement
    SOCIAL = "social"                            # Social recognition
    CURIOSITY = "curiosity"                      # Knowledge seeking
    MASTERY = "mastery"                          # Skill mastery


class LearningEnvironment(Enum):
    """Learning environment types"""
    INDIVIDUAL = "individual"                    # Solo learning
    COLLABORATIVE = "collaborative"              # Group learning
    COMPETITIVE = "competitive"                  # Competition-based
    MENTORED = "mentored"                        # With mentor guidance
    PEER_TO_PEER = "peer_to_peer"               # Peer learning
    MIXED = "mixed"                              # Combination of types


class ResponseFormat(Enum):
    """Response format preferences"""
    TEXT = "text"                                # Text-based response
    STRUCTURED = "structured"                    # Structured format
    CONVERSATIONAL = "conversational"            # Natural conversation
    STEP_BY_STEP = "step_by_step"               # Sequential steps
    BULLET_POINTS = "bullet_points"             # Bulleted lists
    NARRATIVE = "narrative"                      # Story format


class PriorityLevel(Enum):
    """Priority levels for learning content"""
    CRITICAL = "critical"                        # Must learn
    HIGH = "high"                                # Important to learn
    MEDIUM = "medium"                            # Useful to learn
    LOW = "low"                                  # Nice to learn
    OPTIONAL = "optional"                        # Supplementary


class ProgressStatus(Enum):
    """Learning progress status"""
    NOT_STARTED = "not_started"                 # Haven't begun
    IN_PROGRESS = "in_progress"                 # Currently learning
    COMPLETED = "completed"                      # Finished learning
    MASTERED = "mastered"                        # Achieved mastery
    NEEDS_REVIEW = "needs_review"               # Requires revision
    STRUGGLING = "struggling"                    # Having difficulties
