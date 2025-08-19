"""
Advanced Learning Psychology Service
Implements metacognitive training, memory palace builder, elaborative interrogation, and transfer learning
"""

import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
import asyncio
from groq import AsyncGroq
import os

logger = logging.getLogger(__name__)

class MetacognitiveStrategy(Enum):
    """Metacognitive learning strategies"""
    SELF_QUESTIONING = "self_questioning"
    GOAL_SETTING = "goal_setting"
    PROGRESS_MONITORING = "progress_monitoring"
    STRATEGY_SELECTION = "strategy_selection"
    REFLECTION = "reflection"
    PLANNING = "planning"

class MemoryPalaceType(Enum):
    """Types of memory palaces"""
    HOME = "home"
    SCHOOL = "school"
    NATURE = "nature"
    CASTLE = "castle"
    LIBRARY = "library"
    LABORATORY = "laboratory"
    CUSTOM = "custom"

class QuestionType(Enum):
    """Types of elaborative interrogation questions"""
    WHY = "why"
    HOW = "how"
    WHAT_IF = "what_if"
    COMPARE = "compare"
    APPLY = "apply"
    SYNTHESIZE = "synthesize"

class TransferType(Enum):
    """Types of knowledge transfer"""
    ANALOGICAL = "analogical"
    PROCEDURAL = "procedural"
    CONCEPTUAL = "conceptual"
    STRATEGIC = "strategic"

@dataclass
class MetacognitiveSession:
    """Metacognitive training session data"""
    session_id: str
    user_id: str
    strategy: MetacognitiveStrategy
    topic: str
    level: str
    responses: List[Dict[str, Any]]
    insights: Dict[str, Any]
    recommendations: List[str]
    completion_score: float
    created_at: datetime
    updated_at: datetime

@dataclass
class MemoryPalace:
    """Memory palace structure"""
    palace_id: str
    user_id: str
    name: str
    palace_type: MemoryPalaceType
    description: str
    rooms: List[Dict[str, Any]]
    pathways: List[Dict[str, Any]]
    information_nodes: List[Dict[str, Any]]
    visualization_data: Dict[str, Any]
    effectiveness_score: float
    created_at: datetime
    updated_at: datetime

@dataclass
class ElaborativeQuestion:
    """Elaborative interrogation question"""
    question_id: str
    question_type: QuestionType
    content: str
    difficulty_level: str
    subject_area: str
    expected_answer_type: str
    evaluation_criteria: List[str]
    follow_up_questions: List[str]

@dataclass
class TransferScenario:
    """Knowledge transfer scenario"""
    scenario_id: str
    source_domain: str
    target_domain: str
    transfer_type: TransferType
    scenario_description: str
    key_concepts: List[str]
    analogy_mapping: Dict[str, str]
    exercises: List[Dict[str, Any]]
    difficulty_level: str

class LearningPsychologyService:
    """Advanced Learning Psychology Service"""
    
    def __init__(self):
        self.groq_client = AsyncGroq(api_key=os.getenv('GROQ_API_KEY'))
        self.model = "deepseek-r1-distill-llama-70b"
        
        # Storage for active sessions and palaces
        self.active_metacognitive_sessions: Dict[str, MetacognitiveSession] = {}
        self.memory_palaces: Dict[str, MemoryPalace] = {}
        self.user_progress: Dict[str, Dict[str, Any]] = {}
        
        # Pre-built memory palace templates
        self.palace_templates = {
            MemoryPalaceType.HOME: {
                "description": "A familiar home environment with multiple rooms",
                "default_rooms": ["living_room", "kitchen", "bedroom", "study", "bathroom"],
                "pathways": ["hallway", "stairs", "doorways"]
            },
            MemoryPalaceType.LIBRARY: {
                "description": "A grand library with sections and reading areas",
                "default_rooms": ["main_hall", "fiction_section", "reference_section", "study_area", "archives"],
                "pathways": ["main_aisle", "reading_corridors", "staircase"]
            },
            MemoryPalaceType.NATURE: {
                "description": "A natural environment with distinct landmarks",
                "default_rooms": ["forest_clearing", "river_bank", "mountain_peak", "cave", "meadow"],
                "pathways": ["forest_path", "river_flow", "mountain_trail"]
            }
        }
        
        logger.info("Learning Psychology Service initialized")

    # ================================
    # METACOGNITIVE TRAINING METHODS
    # ================================

    async def start_metacognitive_session(self, user_id: str, strategy: MetacognitiveStrategy, 
                                        topic: str, level: str = "intermediate") -> MetacognitiveSession:
        """Start a new metacognitive training session"""
        try:
            session_id = str(uuid.uuid4())
            
            session = MetacognitiveSession(
                session_id=session_id,
                user_id=user_id,
                strategy=strategy,
                topic=topic,
                level=level,
                responses=[],
                insights={},
                recommendations=[],
                completion_score=0.0,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            self.active_metacognitive_sessions[session_id] = session
            
            # Generate initial strategy-specific prompts
            initial_prompt = await self._generate_metacognitive_prompt(strategy, topic, level)
            
            session.responses.append({
                "type": "initial_prompt",
                "content": initial_prompt,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            logger.info(f"Started metacognitive session {session_id} for strategy {strategy.value}")
            return session
            
        except Exception as e:
            logger.error(f"Error starting metacognitive session: {str(e)}")
            raise

    async def process_metacognitive_response(self, session_id: str, user_response: str) -> Dict[str, Any]:
        """Process user response in metacognitive session"""
        try:
            session = self.active_metacognitive_sessions.get(session_id)
            if not session:
                raise ValueError("Session not found")
            
            # Add user response
            session.responses.append({
                "type": "user_response",
                "content": user_response,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Analyze response and generate feedback
            analysis = await self._analyze_metacognitive_response(session, user_response)
            
            # Generate next step or conclusion
            next_step = await self._generate_next_metacognitive_step(session, analysis)
            
            session.responses.append({
                "type": "ai_feedback",
                "content": next_step,
                "analysis": analysis,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Update session insights
            session.insights.update(analysis.get("insights", {}))
            session.completion_score = analysis.get("completion_score", session.completion_score)
            session.updated_at = datetime.utcnow()
            
            return {
                "feedback": next_step,
                "analysis": analysis,
                "session_progress": {
                    "completion_score": session.completion_score,
                    "insights": session.insights,
                    "responses_count": len([r for r in session.responses if r["type"] == "user_response"])
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing metacognitive response: {str(e)}")
            raise

    async def _generate_metacognitive_prompt(self, strategy: MetacognitiveStrategy, 
                                           topic: str, level: str) -> str:
        """Generate initial metacognitive prompt based on strategy"""
        try:
            strategy_prompts = {
                MetacognitiveStrategy.SELF_QUESTIONING: f"""
                Let's practice self-questioning techniques for learning {topic} at {level} level.
                
                Self-questioning is a powerful metacognitive strategy that helps you:
                - Monitor your understanding
                - Identify knowledge gaps
                - Connect new information to existing knowledge
                
                Start by answering these questions about {topic}:
                1. What do I already know about this topic?
                2. What specific aspects am I uncertain about?
                3. What questions should I be asking myself as I learn?
                4. How can I test my understanding?
                
                Choose one question to start with and provide a detailed response.
                """,
                
                MetacognitiveStrategy.GOAL_SETTING: f"""
                Let's set up effective learning goals for {topic} at {level} level.
                
                Effective goal setting involves:
                - Specific, measurable objectives
                - Timeline and milestones
                - Success criteria
                - Regular progress assessment
                
                Let's work together to create your learning goals:
                1. What specific aspects of {topic} do you want to master?
                2. By when do you want to achieve this learning goal?
                3. How will you measure your progress?
                4. What potential obstacles might you face?
                
                Start by describing your main learning objective for {topic}.
                """,
                
                MetacognitiveStrategy.PROGRESS_MONITORING: f"""
                Let's develop your progress monitoring skills for {topic} at {level} level.
                
                Progress monitoring helps you:
                - Track learning effectiveness
                - Adjust strategies when needed
                - Maintain motivation
                - Identify successful learning patterns
                
                Let's assess your current learning progress:
                1. What methods are you currently using to learn {topic}?
                2. How effective have these methods been so far?
                3. What evidence do you have of your progress?
                4. What adjustments might improve your learning?
                
                Begin by describing your current learning approach for {topic}.
                """
            }
            
            return strategy_prompts.get(strategy, f"Let's explore {strategy.value} techniques for learning {topic}.")
            
        except Exception as e:
            logger.error(f"Error generating metacognitive prompt: {str(e)}")
            return f"Let's work on developing your {strategy.value} skills for {topic}."

    async def _analyze_metacognitive_response(self, session: MetacognitiveSession, 
                                           user_response: str) -> Dict[str, Any]:
        """Analyze user's metacognitive response"""
        try:
            analysis_prompt = f"""
            Analyze this student's metacognitive response about {session.topic} using {session.strategy.value} strategy.
            
            Student Response: "{user_response}"
            Learning Level: {session.level}
            Strategy: {session.strategy.value}
            
            Analyze the response for:
            1. Depth of self-awareness
            2. Quality of reflection
            3. Strategic thinking evidence
            4. Areas for improvement
            5. Learning insights demonstrated
            
            Provide analysis as JSON:
            {{
                "self_awareness_score": 0.8,
                "reflection_quality": "high",
                "strategic_thinking": ["strategy1", "strategy2"],
                "improvement_areas": ["area1", "area2"],
                "insights": {{
                    "key_insight": "explanation"
                }},
                "completion_score": 0.7,
                "next_focus": "what to focus on next"
            }}
            """
            
            response = await self.groq_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.3,
                max_tokens=500
            )
            
            try:
                analysis = json.loads(response.choices[0].message.content)
                return analysis
            except json.JSONDecodeError:
                # Fallback analysis
                return {
                    "self_awareness_score": 0.6,
                    "reflection_quality": "moderate",
                    "strategic_thinking": ["basic reflection"],
                    "improvement_areas": ["deeper analysis"],
                    "insights": {"engagement": "User is actively engaging with the material"},
                    "completion_score": 0.5,
                    "next_focus": "developing deeper metacognitive awareness"
                }
                
        except Exception as e:
            logger.error(f"Error analyzing metacognitive response: {str(e)}")
            return {"error": str(e)}

    async def _generate_next_metacognitive_step(self, session: MetacognitiveSession, 
                                              analysis: Dict[str, Any]) -> str:
        """Generate next step in metacognitive training"""
        try:
            next_step_prompt = f"""
            Based on the analysis of a student's metacognitive response, generate the next training step.
            
            Session Context:
            - Topic: {session.topic}
            - Strategy: {session.strategy.value}
            - Level: {session.level}
            - Current Progress: {len(session.responses)} responses
            
            Analysis Results:
            - Self-awareness score: {analysis.get('self_awareness_score', 0.5)}
            - Reflection quality: {analysis.get('reflection_quality', 'moderate')}
            - Next focus: {analysis.get('next_focus', 'continued practice')}
            
            Generate an encouraging and constructive next step that:
            1. Acknowledges their current insights
            2. Guides them to deeper metacognitive awareness
            3. Provides specific, actionable guidance
            4. Maintains motivation and engagement
            
            Keep the response conversational and supportive.
            """
            
            response = await self.groq_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": next_step_prompt}],
                temperature=0.7,
                max_tokens=400
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating next metacognitive step: {str(e)}")
            return "Great work! Let's continue exploring your learning process. What patterns do you notice in how you approach new information?"

    # ================================
    # MEMORY PALACE BUILDER METHODS
    # ================================

    async def create_memory_palace(self, user_id: str, name: str, palace_type: MemoryPalaceType,
                                 topic: str, information_items: List[str]) -> MemoryPalace:
        """Create a new AI-assisted memory palace"""
        try:
            palace_id = str(uuid.uuid4())
            
            # Generate palace structure
            palace_structure = await self._generate_palace_structure(palace_type, topic, information_items)
            
            # Create memory palace
            palace = MemoryPalace(
                palace_id=palace_id,
                user_id=user_id,
                name=name,
                palace_type=palace_type,
                description=palace_structure["description"],
                rooms=palace_structure["rooms"],
                pathways=palace_structure["pathways"],
                information_nodes=palace_structure["information_nodes"],
                visualization_data=palace_structure["visualization_data"],
                effectiveness_score=0.0,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            self.memory_palaces[palace_id] = palace
            
            logger.info(f"Created memory palace {palace_id} for user {user_id}")
            return palace
            
        except Exception as e:
            logger.error(f"Error creating memory palace: {str(e)}")
            raise

    async def _generate_palace_structure(self, palace_type: MemoryPalaceType, 
                                       topic: str, information_items: List[str]) -> Dict[str, Any]:
        """Generate AI-powered memory palace structure"""
        try:
            template = self.palace_templates.get(palace_type, self.palace_templates[MemoryPalaceType.HOME])
            
            structure_prompt = f"""
            Create a detailed memory palace structure for learning {topic}.
            
            Palace Type: {palace_type.value}
            Information to Memorize: {information_items}
            Base Template: {template}
            
            Generate a memory palace with:
            1. Vivid, memorable room descriptions
            2. Clear pathways between locations
            3. Specific placement strategies for each information item
            4. Visual and sensory associations
            5. Logical flow and navigation
            
            Return as JSON:
            {{
                "description": "overall palace description",
                "rooms": [
                    {{
                        "name": "room_name",
                        "description": "vivid description",
                        "visual_elements": ["element1", "element2"],
                        "information_capacity": 3
                    }}
                ],
                "pathways": [
                    {{
                        "from": "room1",
                        "to": "room2", 
                        "description": "path description",
                        "navigation_cue": "memorable cue"
                    }}
                ],
                "information_nodes": [
                    {{
                        "information": "item to remember",
                        "location": "specific room",
                        "position": "exact position in room",
                        "visual_association": "memorable image",
                        "sensory_cues": ["sound", "texture", "smell"]
                    }}
                ],
                "visualization_data": {{
                    "layout": "palace layout description",
                    "key_landmarks": ["landmark1", "landmark2"],
                    "color_scheme": "dominant colors"
                }}
            }}
            """
            
            response = await self.groq_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": structure_prompt}],
                temperature=0.6,
                max_tokens=1000
            )
            
            try:
                structure = json.loads(response.choices[0].message.content)
                return structure
            except json.JSONDecodeError:
                # Fallback structure
                return self._create_fallback_palace_structure(palace_type, information_items)
                
        except Exception as e:
            logger.error(f"Error generating palace structure: {str(e)}")
            return self._create_fallback_palace_structure(palace_type, information_items)

    def _create_fallback_palace_structure(self, palace_type: MemoryPalaceType, 
                                        information_items: List[str]) -> Dict[str, Any]:
        """Create a fallback palace structure"""
        template = self.palace_templates.get(palace_type, self.palace_templates[MemoryPalaceType.HOME])
        
        rooms = []
        for i, room_name in enumerate(template["default_rooms"]):
            rooms.append({
                "name": room_name,
                "description": f"A memorable {room_name.replace('_', ' ')} with distinct features",
                "visual_elements": ["distinctive furniture", "unique decorations"],
                "information_capacity": 3
            })
        
        information_nodes = []
        for i, item in enumerate(information_items[:len(rooms)]):
            room = rooms[i % len(rooms)]
            information_nodes.append({
                "information": item,
                "location": room["name"],
                "position": "center of the room",
                "visual_association": f"vivid image representing {item}",
                "sensory_cues": ["visual", "spatial"]
            })
        
        return {
            "description": template["description"],
            "rooms": rooms,
            "pathways": [{"from": "entrance", "to": "main_area", "description": "clear pathway"}],
            "information_nodes": information_nodes,
            "visualization_data": {
                "layout": "linear progression through rooms",
                "key_landmarks": ["entrance", "main_area"],
                "color_scheme": "warm, memorable colors"
            }
        }

    async def practice_memory_palace(self, palace_id: str, practice_type: str = "recall") -> Dict[str, Any]:
        """Practice using the memory palace"""
        try:
            palace = self.memory_palaces.get(palace_id)
            if not palace:
                raise ValueError("Memory palace not found")
            
            if practice_type == "recall":
                return await self._generate_recall_practice(palace)
            elif practice_type == "navigation":
                return await self._generate_navigation_practice(palace)
            elif practice_type == "association":
                return await self._generate_association_practice(palace)
            else:
                raise ValueError("Invalid practice type")
                
        except Exception as e:
            logger.error(f"Error in memory palace practice: {str(e)}")
            raise

    async def _generate_recall_practice(self, palace: MemoryPalace) -> Dict[str, Any]:
        """Generate recall practice session"""
        try:
            recall_prompt = f"""
            Create a memory palace recall practice session.
            
            Palace: {palace.name}
            Type: {palace.palace_type.value}
            Information Nodes: {len(palace.information_nodes)}
            
            Generate practice questions that test:
            1. Location recall (where is information placed?)
            2. Information recall (what information is at each location?)
            3. Pathway navigation (how to get from A to B?)
            4. Association strength (visual/sensory connections)
            
            Return as JSON with practice questions and guidance.
            """
            
            response = await self.groq_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": recall_prompt}],
                temperature=0.5,
                max_tokens=600
            )
            
            # Parse response and create practice session
            practice_data = {
                "type": "recall",
                "instructions": "Navigate through your memory palace and recall the information at each location",
                "questions": self._extract_practice_questions(response.choices[0].message.content),
                "palace_map": {
                    "rooms": [{"name": room["name"], "description": room["description"]} for room in palace.rooms],
                    "pathways": palace.pathways
                }
            }
            
            return practice_data
            
        except Exception as e:
            logger.error(f"Error generating recall practice: {str(e)}")
            return {"type": "recall", "error": str(e)}

    def _extract_practice_questions(self, ai_response: str) -> List[Dict[str, str]]:
        """Extract practice questions from AI response"""
        # Simple extraction - in production, this would be more sophisticated
        questions = []
        lines = ai_response.split('\n')
        
        for line in lines:
            if '?' in line and len(line.strip()) > 10:
                questions.append({
                    "question": line.strip(),
                    "type": "recall",
                    "difficulty": "medium"
                })
        
        return questions[:5]  # Limit to 5 questions

    # ================================
    # ELABORATIVE INTERROGATION METHODS
    # ================================

    async def generate_elaborative_questions(self, topic: str, subject_area: str, 
                                           difficulty_level: str = "intermediate",
                                           question_count: int = 5) -> List[ElaborativeQuestion]:
        """Generate elaborative interrogation questions"""
        try:
            questions = []
            
            for question_type in QuestionType:
                if len(questions) >= question_count:
                    break
                
                question = await self._generate_single_elaborative_question(
                    question_type, topic, subject_area, difficulty_level
                )
                questions.append(question)
            
            return questions
            
        except Exception as e:
            logger.error(f"Error generating elaborative questions: {str(e)}")
            raise

    async def _generate_single_elaborative_question(self, question_type: QuestionType, 
                                                  topic: str, subject_area: str, 
                                                  difficulty_level: str) -> ElaborativeQuestion:
        """Generate a single elaborative question"""
        try:
            question_prompt = f"""
            Generate a {question_type.value} question for elaborative interrogation.
            
            Topic: {topic}
            Subject Area: {subject_area}
            Difficulty: {difficulty_level}
            Question Type: {question_type.value}
            
            Create a question that:
            1. Promotes deep thinking about the topic
            2. Encourages explanation and reasoning
            3. Connects to broader concepts
            4. Appropriate for {difficulty_level} level
            
            Return as JSON:
            {{
                "content": "the main question",
                "expected_answer_type": "explanation/comparison/analysis",
                "evaluation_criteria": ["criterion1", "criterion2"],
                "follow_up_questions": ["follow_up1", "follow_up2"]
            }}
            """
            
            response = await self.groq_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": question_prompt}],
                temperature=0.6,
                max_tokens=400
            )
            
            try:
                data = json.loads(response.choices[0].message.content)
                return ElaborativeQuestion(
                    question_id=str(uuid.uuid4()),
                    question_type=question_type,
                    content=data.get("content", f"Why is {topic} important in {subject_area}?"),
                    difficulty_level=difficulty_level,
                    subject_area=subject_area,
                    expected_answer_type=data.get("expected_answer_type", "explanation"),
                    evaluation_criteria=data.get("evaluation_criteria", ["accuracy", "depth"]),
                    follow_up_questions=data.get("follow_up_questions", [])
                )
            except json.JSONDecodeError:
                # Fallback question
                return self._create_fallback_question(question_type, topic, subject_area, difficulty_level)
                
        except Exception as e:
            logger.error(f"Error generating single elaborative question: {str(e)}")
            return self._create_fallback_question(question_type, topic, subject_area, difficulty_level)

    def _create_fallback_question(self, question_type: QuestionType, topic: str, 
                                subject_area: str, difficulty_level: str) -> ElaborativeQuestion:
        """Create a fallback elaborative question"""
        fallback_questions = {
            QuestionType.WHY: f"Why is {topic} significant in {subject_area}?",
            QuestionType.HOW: f"How does {topic} work in the context of {subject_area}?",
            QuestionType.WHAT_IF: f"What if {topic} didn't exist in {subject_area}?",
            QuestionType.COMPARE: f"How does {topic} compare to similar concepts in {subject_area}?",
            QuestionType.APPLY: f"How would you apply {topic} to solve problems in {subject_area}?",
            QuestionType.SYNTHESIZE: f"How does {topic} connect with other concepts in {subject_area}?"
        }
        
        return ElaborativeQuestion(
            question_id=str(uuid.uuid4()),
            question_type=question_type,
            content=fallback_questions.get(question_type, f"Explain {topic} in detail."),
            difficulty_level=difficulty_level,
            subject_area=subject_area,
            expected_answer_type="detailed explanation",
            evaluation_criteria=["accuracy", "completeness", "depth of understanding"],
            follow_up_questions=[f"Can you provide an example of {topic}?"]
        )

    async def evaluate_elaborative_response(self, question: ElaborativeQuestion, 
                                          user_response: str) -> Dict[str, Any]:
        """Evaluate user's response to elaborative question"""
        try:
            evaluation_prompt = f"""
            Evaluate this student's response to an elaborative interrogation question.
            
            Question: {question.content}
            Question Type: {question.question_type.value}
            Subject Area: {question.subject_area}
            Difficulty Level: {question.difficulty_level}
            Evaluation Criteria: {question.evaluation_criteria}
            
            Student Response: "{user_response}"
            
            Evaluate the response for:
            1. Accuracy of information
            2. Depth of explanation
            3. Use of examples
            4. Connection to broader concepts
            5. Critical thinking evidence
            
            Return evaluation as JSON:
            {{
                "accuracy_score": 0.8,
                "depth_score": 0.7,
                "examples_quality": "good",
                "concept_connections": ["connection1", "connection2"],
                "critical_thinking": 0.6,
                "overall_score": 0.7,
                "strengths": ["strength1", "strength2"],
                "improvement_areas": ["area1", "area2"],
                "follow_up_suggestion": "specific suggestion"
            }}
            """
            
            response = await self.groq_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": evaluation_prompt}],
                temperature=0.3,
                max_tokens=500
            )
            
            try:
                evaluation = json.loads(response.choices[0].message.content)
                return evaluation
            except json.JSONDecodeError:
                return {
                    "accuracy_score": 0.7,
                    "depth_score": 0.6,
                    "examples_quality": "adequate",
                    "concept_connections": ["basic understanding shown"],
                    "critical_thinking": 0.5,
                    "overall_score": 0.6,
                    "strengths": ["engaged with the question"],
                    "improvement_areas": ["provide more detail"],
                    "follow_up_suggestion": "Try to explain with specific examples"
                }
                
        except Exception as e:
            logger.error(f"Error evaluating elaborative response: {str(e)}")
            return {"error": str(e)}

    # ================================
    # TRANSFER LEARNING METHODS
    # ================================

    async def create_transfer_scenario(self, source_domain: str, target_domain: str, 
                                     key_concepts: List[str], 
                                     transfer_type: TransferType = TransferType.ANALOGICAL) -> TransferScenario:
        """Create a knowledge transfer learning scenario"""
        try:
            scenario_id = str(uuid.uuid4())
            
            scenario_data = await self._generate_transfer_scenario_content(
                source_domain, target_domain, key_concepts, transfer_type
            )
            
            scenario = TransferScenario(
                scenario_id=scenario_id,
                source_domain=source_domain,
                target_domain=target_domain,
                transfer_type=transfer_type,
                scenario_description=scenario_data["description"],
                key_concepts=key_concepts,
                analogy_mapping=scenario_data["analogy_mapping"],
                exercises=scenario_data["exercises"],
                difficulty_level=scenario_data["difficulty_level"]
            )
            
            logger.info(f"Created transfer scenario {scenario_id}")
            return scenario
            
        except Exception as e:
            logger.error(f"Error creating transfer scenario: {str(e)}")
            raise

    async def _generate_transfer_scenario_content(self, source_domain: str, target_domain: str,
                                                key_concepts: List[str], 
                                                transfer_type: TransferType) -> Dict[str, Any]:
        """Generate content for transfer learning scenario"""
        try:
            scenario_prompt = f"""
            Create a knowledge transfer learning scenario.
            
            Source Domain: {source_domain}
            Target Domain: {target_domain}
            Key Concepts: {key_concepts}
            Transfer Type: {transfer_type.value}
            
            Create a scenario that helps students transfer knowledge from {source_domain} to {target_domain}.
            
            Generate:
            1. Scenario description explaining the transfer opportunity
            2. Analogy mapping between domains
            3. Practice exercises to reinforce transfer
            4. Assessment of difficulty level
            
            Return as JSON:
            {{
                "description": "detailed scenario description",
                "analogy_mapping": {{
                    "concept1_source": "concept1_target",
                    "concept2_source": "concept2_target"
                }},
                "exercises": [
                    {{
                        "title": "exercise title",
                        "description": "exercise description",
                        "instruction": "what student should do",
                        "expected_outcome": "learning outcome"
                    }}
                ],
                "difficulty_level": "beginner/intermediate/advanced"
            }}
            """
            
            response = await self.groq_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": scenario_prompt}],
                temperature=0.6,
                max_tokens=800
            )
            
            try:
                scenario_data = json.loads(response.choices[0].message.content)
                return scenario_data
            except json.JSONDecodeError:
                # Fallback scenario
                return self._create_fallback_transfer_scenario(source_domain, target_domain, key_concepts)
                
        except Exception as e:
            logger.error(f"Error generating transfer scenario content: {str(e)}")
            return self._create_fallback_transfer_scenario(source_domain, target_domain, key_concepts)

    def _create_fallback_transfer_scenario(self, source_domain: str, target_domain: str, 
                                         key_concepts: List[str]) -> Dict[str, Any]:
        """Create fallback transfer scenario"""
        return {
            "description": f"Learn how concepts from {source_domain} can be applied in {target_domain}",
            "analogy_mapping": {concept: f"{concept} in {target_domain}" for concept in key_concepts},
            "exercises": [
                {
                    "title": "Basic Transfer Exercise",
                    "description": f"Apply {source_domain} concepts to {target_domain}",
                    "instruction": "Identify similarities and differences between the domains",
                    "expected_outcome": "Understanding of knowledge transfer principles"
                }
            ],
            "difficulty_level": "intermediate"
        }

    async def practice_knowledge_transfer(self, scenario_id: str, user_response: str) -> Dict[str, Any]:
        """Practice knowledge transfer with feedback"""
        # Implementation would involve evaluating user's transfer attempts
        # and providing guidance on improving transfer skills
        pass

    # ================================
    # UTILITY METHODS
    # ================================

    def get_user_progress_summary(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive progress summary for user"""
        try:
            progress = self.user_progress.get(user_id, {})
            
            summary = {
                "metacognitive_sessions": len([s for s in self.active_metacognitive_sessions.values() 
                                             if s.user_id == user_id]),
                "memory_palaces": len([p for p in self.memory_palaces.values() 
                                     if p.user_id == user_id]),
                "overall_progress": progress.get("overall_progress", 0.0),
                "strengths": progress.get("strengths", []),
                "improvement_areas": progress.get("improvement_areas", []),
                "recommendations": progress.get("recommendations", [])
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting progress summary: {str(e)}")
            return {}

    def update_user_progress(self, user_id: str, session_data: Dict[str, Any]):
        """Update user's learning psychology progress"""
        if user_id not in self.user_progress:
            self.user_progress[user_id] = {
                "overall_progress": 0.0,
                "strengths": [],
                "improvement_areas": [],
                "recommendations": [],
                "session_history": []
            }
        
        self.user_progress[user_id]["session_history"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "data": session_data
        })

# Create global instance
learning_psychology_service = LearningPsychologyService()