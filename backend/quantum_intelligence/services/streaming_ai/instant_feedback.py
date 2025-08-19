"""
Instant Feedback Generation Engine

AI-powered system for generating personalized, contextual feedback in real-time
with sub-100ms latency, emotional intelligence, and adaptive personalization.
"""

import asyncio
import time
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import logging

# Try to import structlog, fall back to standard logging
try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    logger = logging.getLogger(__name__)

from ...core.exceptions import QuantumEngineError
from ...utils.caching import CacheService
from .data_structures import (
    InstantFeedback, FeedbackType, StreamingEvent, StreamingEventType
)


class FeedbackUrgency(Enum):
    """Feedback urgency levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EmotionalTone(Enum):
    """Emotional tones for feedback"""
    ENCOURAGING = "encouraging"
    SUPPORTIVE = "supportive"
    CHALLENGING = "challenging"
    NEUTRAL = "neutral"
    CELEBRATORY = "celebratory"


@dataclass
class FeedbackTemplate:
    """Template for generating feedback"""
    template_id: str
    feedback_type: FeedbackType
    emotional_tone: EmotionalTone
    content_template: str
    personalization_variables: List[str]
    effectiveness_score: float
    usage_count: int = 0
    last_used: Optional[datetime] = None


@dataclass
class PersonalizationProfile:
    """User personalization profile for feedback"""
    user_id: str
    preferred_feedback_types: Dict[FeedbackType, float]
    emotional_preferences: Dict[EmotionalTone, float]
    response_patterns: Dict[str, Any]
    learning_style: str
    motivation_factors: List[str]
    feedback_effectiveness_history: deque = field(default_factory=lambda: deque(maxlen=100))
    last_updated: datetime = field(default_factory=datetime.now)


class InstantFeedbackEngine:
    """
    💬 INSTANT FEEDBACK GENERATION ENGINE
    
    AI-powered system for generating personalized, contextual feedback
    in real-time with sub-100ms latency and emotional intelligence.
    """
    
    def __init__(self, cache_service: Optional[CacheService] = None):
        self.cache = cache_service
        
        # Feedback generation components
        self.feedback_templates: Dict[str, FeedbackTemplate] = {}
        self.personalization_profiles: Dict[str, PersonalizationProfile] = {}
        self.feedback_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self.effectiveness_tracking: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Real-time processing
        self.feedback_queue: asyncio.Queue = asyncio.Queue()
        self.processing_task = None
        
        # Performance metrics
        self.generation_metrics = {
            'total_generated': 0,
            'avg_latency_ms': 0,
            'effectiveness_scores': deque(maxlen=1000),
            'user_satisfaction_scores': deque(maxlen=1000)
        }
        
        # Initialize feedback templates
        self._initialize_feedback_templates()
        
        # Feedback type priorities
        self.feedback_type_priorities = {
            FeedbackType.CORRECTNESS: 0.9,
            FeedbackType.ENCOURAGEMENT: 0.8,
            FeedbackType.SUGGESTION: 0.7,
            FeedbackType.CLARIFICATION: 0.6,
            FeedbackType.CHALLENGE: 0.5,
            FeedbackType.REDIRECT: 0.4,
            FeedbackType.METACOGNITIVE: 0.3
        }
        
        logger.info("Instant Feedback Engine initialized")

    def _ensure_processing_task(self):
        """Ensure processing task is running"""
        if self.processing_task is None:
            try:
                self.processing_task = asyncio.create_task(self._process_feedback_requests())
            except RuntimeError:
                # No event loop running, task will be created when needed
                pass
    
    async def generate_real_time_feedback(self,
                                        event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate instant feedback for user action event
        
        Args:
            event: User action event data
            
        Returns:
            Dict: Generated feedback result
        """
        start_time = time.time()

        # Ensure processing task is running
        self._ensure_processing_task()

        try:
            user_id = event.get('user_id', '')
            session_id = event.get('session_id', '')
            action_data = event.get('event_data', {})
            
            # Analyze user action
            action_analysis = await self._analyze_user_action(action_data)
            
            # Get or create personalization profile
            profile = await self._get_personalization_profile(user_id)
            
            # Determine optimal feedback type
            feedback_type = await self._determine_feedback_type(
                action_analysis, profile, action_data
            )
            
            # Generate feedback content
            feedback_content = await self._generate_feedback_content(
                feedback_type, action_analysis, profile, action_data
            )
            
            # Determine emotional tone
            emotional_tone = await self._determine_emotional_tone(
                action_analysis, profile
            )
            
            # Generate suggested actions
            suggested_actions = await self._generate_suggested_actions(
                feedback_type, action_analysis, action_data
            )
            
            # Calculate quality metrics
            quality_metrics = await self._calculate_feedback_quality(
                feedback_content, feedback_type, action_analysis
            )
            
            # Create instant feedback
            instant_feedback = InstantFeedback(
                feedback_id=str(uuid.uuid4()),
                user_id=user_id,
                session_id=session_id,
                feedback_type=feedback_type,
                content=feedback_content,
                confidence_score=quality_metrics['confidence_score'],
                relevance_score=quality_metrics['relevance_score'],
                timing_appropriateness=quality_metrics['timing_appropriateness'],
                learning_impact_prediction=quality_metrics['learning_impact_prediction'],
                suggested_actions=suggested_actions,
                emotional_tone=emotional_tone.value,
                personalization_factors=await self._get_personalization_factors(profile),
                delivery_timestamp=datetime.now(),
                response_required=feedback_type in [FeedbackType.CLARIFICATION, FeedbackType.CHALLENGE]
            )
            
            # Record feedback
            self.feedback_history[user_id].append(instant_feedback)
            
            # Update metrics
            processing_time = (time.time() - start_time) * 1000  # ms
            self._update_generation_metrics(processing_time, quality_metrics)
            
            return {
                'status': 'success',
                'feedback': instant_feedback.__dict__,
                'processing_latency_ms': processing_time,
                'generation_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating instant feedback: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'processing_latency_ms': (time.time() - start_time) * 1000
            }
    
    async def generate_feedback(self,
                              user_action: Dict[str, Any],
                              context: Dict[str, Any],
                              feedback_type: str = 'suggestion') -> InstantFeedback:
        """
        Generate feedback with specified type (convenience method)
        
        Args:
            user_action: User action data
            context: Learning context
            feedback_type: Type of feedback to generate
            
        Returns:
            InstantFeedback: Generated feedback
        """
        try:
            # Convert string to enum
            fb_type = FeedbackType(feedback_type)
            
            # Create event structure
            event = {
                'user_id': context.get('user_id', ''),
                'session_id': context.get('session_id', ''),
                'event_data': {
                    'action_type': user_action.get('action_type', 'general'),
                    'success_level': user_action.get('success_level', 0.5),
                    'completion_status': user_action.get('completed', False),
                    'context': context
                }
            }
            
            # Generate feedback
            result = await self.generate_real_time_feedback(event)
            
            if result['status'] == 'success':
                return InstantFeedback(**result['feedback'])
            else:
                raise QuantumEngineError(f"Feedback generation failed: {result['error']}")
                
        except Exception as e:
            logger.error(f"Error in generate_feedback: {e}")
            raise QuantumEngineError(f"Failed to generate feedback: {e}")
    
    async def _process_feedback_requests(self):
        """Process feedback requests from queue"""
        while True:
            try:
                # Get request from queue with timeout
                try:
                    request = await asyncio.wait_for(
                        self.feedback_queue.get(), timeout=1.0
                    )
                    
                    # Process feedback request
                    result = await self.generate_real_time_feedback(request)
                    
                    # Handle result (could send via WebSocket, etc.)
                    if result['status'] == 'success':
                        logger.debug(f"Generated feedback for user {request.get('user_id')}")
                    else:
                        logger.error(f"Failed to generate feedback: {result.get('error')}")
                        
                except asyncio.TimeoutError:
                    # No requests to process
                    pass
                
                # Brief sleep to prevent CPU spinning
                await asyncio.sleep(0.001)  # 1ms
                
            except Exception as e:
                logger.error(f"Error in feedback processing: {e}")
                await asyncio.sleep(0.1)
    
    async def _analyze_user_action(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user action to understand feedback needs"""
        return {
            'action_type': action_data.get('action_type', 'unknown'),
            'success_level': action_data.get('success_level', 0.5),
            'completion_status': action_data.get('completed', False),
            'effort_indicators': action_data.get('effort_level', 0.5),
            'time_taken': action_data.get('time_taken_seconds', 60),
            'mistakes_made': action_data.get('mistakes', []),
            'help_requested': action_data.get('help_requested', False),
            'confidence_expressed': action_data.get('confidence_level', 0.5),
            'emotional_state': action_data.get('emotional_state', 'neutral'),
            'difficulty_level': action_data.get('difficulty_level', 0.5),
            'feedback_urgency': self._calculate_feedback_urgency(action_data)
        }
    
    async def _get_personalization_profile(self, user_id: str) -> PersonalizationProfile:
        """Get or create personalization profile for user"""
        if user_id not in self.personalization_profiles:
            # Create default profile
            self.personalization_profiles[user_id] = PersonalizationProfile(
                user_id=user_id,
                preferred_feedback_types={
                    FeedbackType.ENCOURAGEMENT: 0.8,
                    FeedbackType.SUGGESTION: 0.7,
                    FeedbackType.CORRECTNESS: 0.6,
                    FeedbackType.CLARIFICATION: 0.5,
                    FeedbackType.CHALLENGE: 0.4,
                    FeedbackType.REDIRECT: 0.3,
                    FeedbackType.METACOGNITIVE: 0.2
                },
                emotional_preferences={
                    EmotionalTone.ENCOURAGING: 0.8,
                    EmotionalTone.SUPPORTIVE: 0.7,
                    EmotionalTone.NEUTRAL: 0.6,
                    EmotionalTone.CHALLENGING: 0.4,
                    EmotionalTone.CELEBRATORY: 0.5
                },
                response_patterns={},
                learning_style='adaptive',
                motivation_factors=['achievement', 'progress', 'mastery']
            )
        
        return self.personalization_profiles[user_id]
    
    async def _determine_feedback_type(self,
                                     action_analysis: Dict[str, Any],
                                     profile: PersonalizationProfile,
                                     action_data: Dict[str, Any]) -> FeedbackType:
        """Determine optimal feedback type"""
        feedback_scores = {}
        
        # Correctness feedback
        if action_analysis['completion_status']:
            if action_analysis['success_level'] > 0.8:
                feedback_scores[FeedbackType.CORRECTNESS] = 0.9
            elif action_analysis['success_level'] < 0.5:
                feedback_scores[FeedbackType.CORRECTNESS] = 0.8
            else:
                feedback_scores[FeedbackType.CORRECTNESS] = 0.6
        
        # Encouragement feedback
        if (action_analysis['confidence_expressed'] < 0.5 or 
            action_analysis['emotional_state'] in ['frustrated', 'confused']):
            feedback_scores[FeedbackType.ENCOURAGEMENT] = 0.8
        elif action_analysis['success_level'] > 0.7:
            feedback_scores[FeedbackType.ENCOURAGEMENT] = 0.7
        
        # Suggestion feedback
        if action_analysis['mistakes_made'] or action_analysis['help_requested']:
            feedback_scores[FeedbackType.SUGGESTION] = 0.8
        elif action_analysis['success_level'] < 0.6:
            feedback_scores[FeedbackType.SUGGESTION] = 0.6
        
        # Challenge feedback
        if action_analysis['success_level'] > 0.9 and action_analysis['time_taken'] < 30:
            feedback_scores[FeedbackType.CHALLENGE] = 0.7
        
        # Clarification feedback
        if action_analysis['emotional_state'] == 'confused' or action_analysis['help_requested']:
            feedback_scores[FeedbackType.CLARIFICATION] = 0.7
        
        # Apply personalization weights
        for fb_type, score in feedback_scores.items():
            personal_weight = profile.preferred_feedback_types.get(fb_type, 0.5)
            feedback_scores[fb_type] = score * personal_weight
        
        # Return highest scoring feedback type
        if feedback_scores:
            return max(feedback_scores.items(), key=lambda x: x[1])[0]
        else:
            return FeedbackType.SUGGESTION  # Default
    
    async def _generate_feedback_content(self,
                                       feedback_type: FeedbackType,
                                       action_analysis: Dict[str, Any],
                                       profile: PersonalizationProfile,
                                       action_data: Dict[str, Any]) -> str:
        """Generate personalized feedback content"""
        # Get appropriate template
        template = self._get_feedback_template(feedback_type, profile)
        
        if template:
            # Personalize template
            content = await self._personalize_template(template, action_analysis, profile)
        else:
            # Generate basic feedback
            content = await self._generate_basic_feedback(feedback_type, action_analysis)
        
        return content
    
    def _get_feedback_template(self, feedback_type: FeedbackType, profile: PersonalizationProfile) -> Optional[FeedbackTemplate]:
        """Get appropriate feedback template"""
        # Find templates for this feedback type
        matching_templates = [
            template for template in self.feedback_templates.values()
            if template.feedback_type == feedback_type
        ]
        
        if matching_templates:
            # Return template with highest effectiveness score
            return max(matching_templates, key=lambda t: t.effectiveness_score)
        
        return None
    
    async def _personalize_template(self,
                                  template: FeedbackTemplate,
                                  action_analysis: Dict[str, Any],
                                  profile: PersonalizationProfile) -> str:
        """Personalize feedback template"""
        content = template.content_template
        
        # Replace personalization variables
        replacements = {
            '{user_name}': profile.user_id,  # In production, would use actual name
            '{success_level}': f"{action_analysis['success_level']:.0%}",
            '{action_type}': action_analysis['action_type'],
            '{encouragement}': self._get_encouragement_phrase(profile),
            '{suggestion}': self._get_suggestion_phrase(action_analysis),
            '{next_step}': self._get_next_step_phrase(action_analysis)
        }
        
        for variable, value in replacements.items():
            content = content.replace(variable, str(value))
        
        return content
    
    async def _generate_basic_feedback(self, feedback_type: FeedbackType, action_analysis: Dict[str, Any]) -> str:
        """Generate basic feedback when no template available"""
        if feedback_type == FeedbackType.ENCOURAGEMENT:
            if action_analysis['success_level'] > 0.7:
                return "Great work! You're making excellent progress."
            else:
                return "Keep going! You're learning and improving."
        
        elif feedback_type == FeedbackType.SUGGESTION:
            return "Here's a suggestion to help you improve your approach."
        
        elif feedback_type == FeedbackType.CORRECTNESS:
            if action_analysis['success_level'] > 0.8:
                return "Correct! Well done."
            else:
                return "Let's review this together."
        
        elif feedback_type == FeedbackType.CLARIFICATION:
            return "Let me clarify this concept for you."
        
        elif feedback_type == FeedbackType.CHALLENGE:
            return "Ready for a challenge? Let's try something more advanced."
        
        else:
            return "Here's some feedback on your progress."
    
    async def _determine_emotional_tone(self,
                                      action_analysis: Dict[str, Any],
                                      profile: PersonalizationProfile) -> EmotionalTone:
        """Determine appropriate emotional tone"""
        # Base tone on action analysis
        if action_analysis['success_level'] > 0.9:
            base_tone = EmotionalTone.CELEBRATORY
        elif action_analysis['success_level'] > 0.7:
            base_tone = EmotionalTone.ENCOURAGING
        elif action_analysis['emotional_state'] in ['frustrated', 'confused']:
            base_tone = EmotionalTone.SUPPORTIVE
        elif action_analysis['confidence_expressed'] > 0.8:
            base_tone = EmotionalTone.CHALLENGING
        else:
            base_tone = EmotionalTone.NEUTRAL
        
        # Apply personalization
        personal_preference = profile.emotional_preferences.get(base_tone, 0.5)
        if personal_preference < 0.3:
            # User doesn't prefer this tone, use their most preferred
            base_tone = max(profile.emotional_preferences.items(), key=lambda x: x[1])[0]
        
        return base_tone
    
    async def _generate_suggested_actions(self,
                                        feedback_type: FeedbackType,
                                        action_analysis: Dict[str, Any],
                                        action_data: Dict[str, Any]) -> List[str]:
        """Generate suggested actions for the user"""
        suggestions = []
        
        if feedback_type == FeedbackType.SUGGESTION:
            if action_analysis['mistakes_made']:
                suggestions.append("Review the areas where mistakes occurred")
                suggestions.append("Practice similar problems")
            if action_analysis['help_requested']:
                suggestions.append("Ask specific questions about confusing concepts")
        
        elif feedback_type == FeedbackType.CHALLENGE:
            suggestions.append("Try a more advanced version of this problem")
            suggestions.append("Explore related concepts")
        
        elif feedback_type == FeedbackType.ENCOURAGEMENT:
            suggestions.append("Continue with the current approach")
            suggestions.append("Build on this success")
        
        return suggestions
    
    async def _calculate_feedback_quality(self,
                                        content: str,
                                        feedback_type: FeedbackType,
                                        action_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate feedback quality metrics"""
        # Simplified quality calculation
        return {
            'confidence_score': 0.8,  # Based on template quality and personalization
            'relevance_score': 0.9,   # Based on action analysis match
            'timing_appropriateness': 0.95,  # Real-time is always appropriate
            'learning_impact_prediction': 0.7  # Based on feedback type effectiveness
        }
    
    async def _get_personalization_factors(self, profile: PersonalizationProfile) -> Dict[str, Any]:
        """Get personalization factors for feedback"""
        return {
            'learning_style': profile.learning_style,
            'motivation_factors': profile.motivation_factors,
            'preferred_feedback_types': {k.value: v for k, v in profile.preferred_feedback_types.items()},
            'emotional_preferences': {k.value: v for k, v in profile.emotional_preferences.items()}
        }
    
    def _calculate_feedback_urgency(self, action_data: Dict[str, Any]) -> FeedbackUrgency:
        """Calculate feedback urgency level"""
        if action_data.get('emotional_state') in ['frustrated', 'confused']:
            return FeedbackUrgency.HIGH
        elif action_data.get('help_requested'):
            return FeedbackUrgency.MEDIUM
        elif action_data.get('success_level', 0.5) < 0.3:
            return FeedbackUrgency.HIGH
        else:
            return FeedbackUrgency.LOW
    
    def _update_generation_metrics(self, processing_time: float, quality_metrics: Dict[str, float]):
        """Update feedback generation metrics"""
        self.generation_metrics['total_generated'] += 1
        
        # Update average latency
        current_avg = self.generation_metrics['avg_latency_ms']
        total_count = self.generation_metrics['total_generated']
        self.generation_metrics['avg_latency_ms'] = (current_avg * (total_count - 1) + processing_time) / total_count
        
        # Track effectiveness
        self.generation_metrics['effectiveness_scores'].append(quality_metrics['learning_impact_prediction'])
    
    def _initialize_feedback_templates(self):
        """Initialize feedback templates"""
        # Sample templates (in production, these would be more comprehensive)
        templates = [
            FeedbackTemplate(
                template_id="encourage_success",
                feedback_type=FeedbackType.ENCOURAGEMENT,
                emotional_tone=EmotionalTone.CELEBRATORY,
                content_template="Excellent work! You achieved {success_level} success on this {action_type}. {encouragement}",
                personalization_variables=['{success_level}', '{action_type}', '{encouragement}'],
                effectiveness_score=0.9
            ),
            FeedbackTemplate(
                template_id="suggest_improvement",
                feedback_type=FeedbackType.SUGGESTION,
                emotional_tone=EmotionalTone.SUPPORTIVE,
                content_template="I notice you're working on {action_type}. {suggestion} This should help improve your approach.",
                personalization_variables=['{action_type}', '{suggestion}'],
                effectiveness_score=0.8
            ),
            FeedbackTemplate(
                template_id="clarify_concept",
                feedback_type=FeedbackType.CLARIFICATION,
                emotional_tone=EmotionalTone.NEUTRAL,
                content_template="Let me clarify this {action_type} concept. {next_step}",
                personalization_variables=['{action_type}', '{next_step}'],
                effectiveness_score=0.85
            )
        ]
        
        for template in templates:
            self.feedback_templates[template.template_id] = template
    
    # Helper methods for template personalization
    def _get_encouragement_phrase(self, profile: PersonalizationProfile) -> str:
        """Get personalized encouragement phrase"""
        if 'achievement' in profile.motivation_factors:
            return "You're achieving great results!"
        elif 'progress' in profile.motivation_factors:
            return "You're making steady progress!"
        else:
            return "Keep up the good work!"
    
    def _get_suggestion_phrase(self, action_analysis: Dict[str, Any]) -> str:
        """Get contextual suggestion phrase"""
        if action_analysis['mistakes_made']:
            return "Try reviewing the key concepts before attempting again."
        elif action_analysis['time_taken'] > 120:
            return "Consider breaking this down into smaller steps."
        else:
            return "Here's an approach that might help."
    
    def _get_next_step_phrase(self, action_analysis: Dict[str, Any]) -> str:
        """Get next step guidance phrase"""
        if action_analysis['success_level'] > 0.8:
            return "You're ready to move on to the next challenge."
        else:
            return "Let's practice this a bit more before moving forward."
