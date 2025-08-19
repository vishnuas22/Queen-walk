"""
Base Learning Mode class for all quantum learning modes
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import logging

# Try to import structlog, fall back to standard logging
try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    logger = logging.getLogger(__name__)

from ..core.data_structures import QuantumLearningContext, QuantumResponse
from ..core.enums import QuantumLearningMode, QuantumState, IntelligenceLevel


class BaseLearningMode(ABC):
    """Abstract base class for all learning modes"""
    
    def __init__(self, mode: QuantumLearningMode):
        self.mode = mode
        # Handle both structlog and standard logging
        if hasattr(logger, 'bind'):
            self.logger = logger.bind(learning_mode=mode.value)
        else:
            self.logger = logger
    
    @abstractmethod
    async def generate_response(
        self, 
        user_message: str, 
        context: QuantumLearningContext
    ) -> QuantumResponse:
        """Generate a response using this learning mode"""
        pass
    
    @abstractmethod
    async def analyze_user_input(
        self, 
        user_message: str, 
        context: QuantumLearningContext
    ) -> Dict[str, Any]:
        """Analyze user input for mode-specific insights"""
        pass
    
    @abstractmethod
    async def determine_optimal_state(
        self, 
        user_message: str, 
        context: QuantumLearningContext
    ) -> QuantumState:
        """Determine optimal quantum state for this interaction"""
        pass
    
    @abstractmethod
    async def calculate_intelligence_level(
        self, 
        user_message: str, 
        context: QuantumLearningContext
    ) -> IntelligenceLevel:
        """Calculate appropriate intelligence level for response"""
        pass
    
    async def pre_process(
        self, 
        user_message: str, 
        context: QuantumLearningContext
    ) -> Dict[str, Any]:
        """Pre-process user input before generating response"""
        self.logger.info("Pre-processing user input", message_length=len(user_message))
        
        # Common pre-processing logic
        analysis = await self.analyze_user_input(user_message, context)
        optimal_state = await self.determine_optimal_state(user_message, context)
        intelligence_level = await self.calculate_intelligence_level(user_message, context)
        
        return {
            'analysis': analysis,
            'optimal_state': optimal_state,
            'intelligence_level': intelligence_level,
            'preprocessed_message': user_message.strip()
        }
    
    async def post_process(
        self, 
        response: QuantumResponse, 
        context: QuantumLearningContext
    ) -> QuantumResponse:
        """Post-process generated response"""
        self.logger.info("Post-processing response", response_length=len(response.content))
        
        # Common post-processing logic
        # Add mode-specific metadata
        if 'mode_specific' not in response.streaming_metadata:
            response.streaming_metadata['mode_specific'] = {}
        
        response.streaming_metadata['mode_specific']['learning_mode'] = self.mode.value
        response.streaming_metadata['mode_specific']['processing_timestamp'] = response.timestamp.isoformat()
        
        return response
    
    def _create_base_response(
        self,
        content: str,
        quantum_state: QuantumState,
        intelligence_level: IntelligenceLevel,
        context: QuantumLearningContext
    ) -> QuantumResponse:
        """Create a base quantum response with common fields"""
        return QuantumResponse(
            content=content,
            quantum_mode=self.mode,
            quantum_state=quantum_state,
            intelligence_level=intelligence_level,
            personalization_score=0.0,
            engagement_prediction=0.0,
            learning_velocity_boost=0.0,
            concept_connections=[],
            knowledge_gaps_identified=[],
            next_optimal_concepts=[],
            metacognitive_insights=[],
            emotional_resonance_score=0.0,
            adaptive_recommendations=[],
            streaming_metadata={},
            quantum_analytics={},
            suggested_actions=[],
            next_steps=""
        )
    
    def _extract_concepts(self, text: str) -> list:
        """Extract key concepts from text"""
        # Simple concept extraction - can be enhanced with NLP
        import re
        
        # Find capitalized words and technical terms
        concepts = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        # Filter out common words
        common_words = {'The', 'This', 'That', 'These', 'Those', 'When', 'Where', 'Why', 'How', 'What'}
        concepts = [c for c in concepts if c not in common_words]
        
        return list(set(concepts))[:10]  # Return up to 10 unique concepts
    
    def _analyze_complexity(self, text: str) -> float:
        """Analyze text complexity (0.0 to 1.0)"""
        # Simple complexity analysis based on sentence length and vocabulary
        sentences = text.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        
        # Normalize to 0-1 scale (assuming max reasonable sentence length of 30 words)
        complexity = min(avg_sentence_length / 30.0, 1.0)
        
        return complexity
    
    def _detect_question_type(self, message: str) -> str:
        """Detect the type of question being asked"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['why', 'because', 'reason']):
            return 'causal'
        elif any(word in message_lower for word in ['how', 'steps', 'process']):
            return 'procedural'
        elif any(word in message_lower for word in ['what', 'define', 'definition']):
            return 'definitional'
        elif any(word in message_lower for word in ['compare', 'difference', 'versus']):
            return 'comparative'
        elif any(word in message_lower for word in ['example', 'instance', 'demonstrate']):
            return 'exemplification'
        else:
            return 'general'
