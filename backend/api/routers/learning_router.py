"""
Learning Router for MasterX Quantum Intelligence Platform

Comprehensive learning management API that integrates with the quantum intelligence
engine to provide personalized learning paths, progress tracking, and adaptive
learning experiences.

📚 LEARNING CAPABILITIES:
- Learning goal creation and management
- Progress tracking and analytics
- Adaptive learning path optimization
- Session management and recording
- Achievement and milestone tracking
- Personalized recommendations

Author: MasterX AI Team - API Integration Division
Version: 1.0 - Phase 12 Enhanced Backend APIs Integration
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks

from ..models import (
    LearningGoal, LearningSession, ProgressRequest, ProgressResponse,
    LearningGoalStatus, UserProfile, BaseResponse
)
from ..auth import get_current_user, require_permission
from ..utils import APIResponseHandler

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# ============================================================================
# LEARNING SERVICE
# ============================================================================

class LearningService:
    """
    📚 LEARNING SERVICE
    
    Comprehensive learning management service that integrates with quantum
    intelligence services to provide personalized learning experiences.
    """
    
    def __init__(self):
        """Initialize the learning service"""
        
        # Learning data stores (replace with database in production)
        self.learning_goals = {}
        self.learning_sessions = {}
        self.user_progress = {}
        
        # Response handler
        self.response_handler = APIResponseHandler()
        
        logger.info("📚 Learning Service initialized")
    
    async def create_learning_goal(self, user_id: str, goal_data: Dict[str, Any]) -> LearningGoal:
        """Create a new learning goal"""
        
        try:
            goal = LearningGoal(
                user_id=user_id,
                title=goal_data['title'],
                description=goal_data['description'],
                subject=goal_data['subject'],
                target_skills=goal_data['target_skills'],
                difficulty_level=goal_data.get('difficulty_level', 0.5),
                estimated_duration_hours=goal_data.get('estimated_duration_hours', 10),
                target_completion_date=goal_data.get('target_completion_date')
            )
            
            # Store goal
            self.learning_goals[goal.goal_id] = goal
            
            # Initialize user progress if not exists
            if user_id not in self.user_progress:
                self.user_progress[user_id] = {
                    'goals': [],
                    'total_study_time': 0,
                    'achievements': [],
                    'skill_levels': {}
                }
            
            self.user_progress[user_id]['goals'].append(goal.goal_id)
            
            logger.info(f"Created learning goal: {goal.goal_id} for user {user_id}")
            return goal
            
        except Exception as e:
            logger.error(f"Error creating learning goal: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create learning goal: {str(e)}")
    
    async def get_learning_goals(self, user_id: str) -> List[LearningGoal]:
        """Get all learning goals for a user"""
        
        try:
            user_goals = []
            
            for goal_id, goal in self.learning_goals.items():
                if goal.user_id == user_id:
                    user_goals.append(goal)
            
            return sorted(user_goals, key=lambda x: x.created_at, reverse=True)
            
        except Exception as e:
            logger.error(f"Error getting learning goals: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get learning goals: {str(e)}")
    
    async def update_learning_goal(self, goal_id: str, user_id: str, updates: Dict[str, Any]) -> LearningGoal:
        """Update a learning goal"""
        
        try:
            if goal_id not in self.learning_goals:
                raise HTTPException(status_code=404, detail="Learning goal not found")
            
            goal = self.learning_goals[goal_id]
            
            # Check ownership
            if goal.user_id != user_id:
                raise HTTPException(status_code=403, detail="Access denied to this learning goal")
            
            # Update fields
            for field, value in updates.items():
                if hasattr(goal, field):
                    setattr(goal, field, value)
            
            logger.info(f"Updated learning goal: {goal_id}")
            return goal
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error updating learning goal: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to update learning goal: {str(e)}")
    
    async def record_learning_session(self, user_id: str, session_data: Dict[str, Any]) -> LearningSession:
        """Record a learning session"""
        
        try:
            session = LearningSession(
                user_id=user_id,
                goal_id=session_data.get('goal_id'),
                subject=session_data['subject'],
                duration_minutes=session_data['duration_minutes'],
                activities=session_data.get('activities', []),
                performance_metrics=session_data.get('performance_metrics', {}),
                engagement_score=session_data.get('engagement_score', 0.5),
                completed_at=datetime.now()
            )
            
            # Store session
            self.learning_sessions[session.session_id] = session
            
            # Update user progress
            await self._update_user_progress(user_id, session)
            
            # Update goal progress if applicable
            if session.goal_id and session.goal_id in self.learning_goals:
                await self._update_goal_progress(session.goal_id, session)
            
            logger.info(f"Recorded learning session: {session.session_id} for user {user_id}")
            return session
            
        except Exception as e:
            logger.error(f"Error recording learning session: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to record learning session: {str(e)}")
    
    async def get_learning_progress(self, user_id: str, request: ProgressRequest) -> ProgressResponse:
        """Get comprehensive learning progress for a user"""
        
        try:
            # Get user progress data
            user_data = self.user_progress.get(user_id, {})
            
            # Get user goals
            user_goals = await self.get_learning_goals(user_id)
            
            # Get recent sessions
            recent_sessions = await self._get_recent_sessions(user_id, request.time_range)
            
            # Calculate overall progress
            overall_progress = await self._calculate_overall_progress(user_id, user_goals)
            
            # Get goals progress
            goals_progress = await self._calculate_goals_progress(user_goals)
            
            # Get subject progress
            subject_progress = await self._calculate_subject_progress(user_id, recent_sessions)
            
            # Get achievements
            achievements = user_data.get('achievements', [])
            
            # Generate recommendations
            recommendations = await self._generate_learning_recommendations(user_id, user_goals, recent_sessions)
            
            return ProgressResponse(
                user_id=user_id,
                overall_progress=overall_progress,
                goals_progress=goals_progress,
                subject_progress=subject_progress,
                recent_sessions=recent_sessions,
                achievements=achievements,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error getting learning progress: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get learning progress: {str(e)}")
    
    async def _update_user_progress(self, user_id: str, session: LearningSession):
        """Update user's overall progress based on session"""
        
        if user_id not in self.user_progress:
            self.user_progress[user_id] = {
                'goals': [],
                'total_study_time': 0,
                'achievements': [],
                'skill_levels': {}
            }
        
        progress = self.user_progress[user_id]
        
        # Update total study time
        progress['total_study_time'] += session.duration_minutes
        
        # Update skill levels based on performance
        for skill, performance in session.performance_metrics.items():
            if skill not in progress['skill_levels']:
                progress['skill_levels'][skill] = 0.0
            
            # Simple skill level update (can be enhanced with ML)
            current_level = progress['skill_levels'][skill]
            progress['skill_levels'][skill] = min(1.0, current_level + (performance * 0.1))
        
        # Check for achievements
        await self._check_achievements(user_id, session)
    
    async def _update_goal_progress(self, goal_id: str, session: LearningSession):
        """Update goal progress based on session"""
        
        if goal_id in self.learning_goals:
            goal = self.learning_goals[goal_id]
            
            # Simple progress calculation (can be enhanced)
            progress_increment = (session.duration_minutes / (goal.estimated_duration_hours * 60)) * 100
            goal.progress_percentage = min(100.0, goal.progress_percentage + progress_increment)
            
            # Update status based on progress
            if goal.progress_percentage >= 100.0:
                goal.status = LearningGoalStatus.COMPLETED
            elif goal.progress_percentage > 0:
                goal.status = LearningGoalStatus.IN_PROGRESS
    
    async def _get_recent_sessions(self, user_id: str, time_range: str) -> List[LearningSession]:
        """Get recent learning sessions for a user"""
        
        # Calculate time cutoff
        now = datetime.now()
        if time_range == "day":
            cutoff = now - timedelta(days=1)
        elif time_range == "week":
            cutoff = now - timedelta(weeks=1)
        elif time_range == "month":
            cutoff = now - timedelta(days=30)
        elif time_range == "quarter":
            cutoff = now - timedelta(days=90)
        else:  # year
            cutoff = now - timedelta(days=365)
        
        # Filter sessions
        recent_sessions = []
        for session in self.learning_sessions.values():
            if session.user_id == user_id and session.started_at >= cutoff:
                recent_sessions.append(session)
        
        return sorted(recent_sessions, key=lambda x: x.started_at, reverse=True)
    
    async def _calculate_overall_progress(self, user_id: str, goals: List[LearningGoal]) -> float:
        """Calculate overall learning progress"""
        
        if not goals:
            return 0.0
        
        total_progress = sum(goal.progress_percentage for goal in goals)
        return total_progress / len(goals)
    
    async def _calculate_goals_progress(self, goals: List[LearningGoal]) -> List[Dict[str, Any]]:
        """Calculate progress for each goal"""
        
        return [
            {
                'goal_id': goal.goal_id,
                'title': goal.title,
                'progress_percentage': goal.progress_percentage,
                'status': goal.status.value,
                'target_completion_date': goal.target_completion_date.isoformat() if goal.target_completion_date else None
            }
            for goal in goals
        ]
    
    async def _calculate_subject_progress(self, user_id: str, sessions: List[LearningSession]) -> Dict[str, float]:
        """Calculate progress by subject"""
        
        subject_time = {}
        subject_performance = {}
        
        for session in sessions:
            subject = session.subject
            
            if subject not in subject_time:
                subject_time[subject] = 0
                subject_performance[subject] = []
            
            subject_time[subject] += session.duration_minutes
            subject_performance[subject].append(session.engagement_score)
        
        # Calculate progress based on time and performance
        subject_progress = {}
        for subject in subject_time:
            time_factor = min(100.0, (subject_time[subject] / 60) * 10)  # 1 hour = 10% progress
            performance_factor = (sum(subject_performance[subject]) / len(subject_performance[subject])) * 100
            subject_progress[subject] = (time_factor + performance_factor) / 2
        
        return subject_progress
    
    async def _check_achievements(self, user_id: str, session: LearningSession):
        """Check and award achievements based on session"""
        
        progress = self.user_progress[user_id]
        achievements = progress['achievements']
        
        # Check for study time achievements
        total_time = progress['total_study_time']
        
        if total_time >= 60 and not any(a['type'] == 'study_time_1h' for a in achievements):
            achievements.append({
                'type': 'study_time_1h',
                'title': 'First Hour',
                'description': 'Completed your first hour of study',
                'earned_at': datetime.now().isoformat()
            })
        
        if total_time >= 600 and not any(a['type'] == 'study_time_10h' for a in achievements):
            achievements.append({
                'type': 'study_time_10h',
                'title': 'Dedicated Learner',
                'description': 'Completed 10 hours of study',
                'earned_at': datetime.now().isoformat()
            })
        
        # Check for engagement achievements
        if session.engagement_score >= 0.9 and not any(a['type'] == 'high_engagement' for a in achievements):
            achievements.append({
                'type': 'high_engagement',
                'title': 'Highly Engaged',
                'description': 'Achieved high engagement in a learning session',
                'earned_at': datetime.now().isoformat()
            })
    
    async def _generate_learning_recommendations(
        self,
        user_id: str,
        goals: List[LearningGoal],
        sessions: List[LearningSession]
    ) -> List[str]:
        """Generate personalized learning recommendations"""
        
        recommendations = []
        
        # Analyze recent activity
        if not sessions:
            recommendations.append("Start your learning journey by creating a learning goal")
            return recommendations
        
        # Check study consistency
        recent_days = set()
        for session in sessions[-7:]:  # Last 7 sessions
            recent_days.add(session.started_at.date())
        
        if len(recent_days) < 3:
            recommendations.append("Try to study more consistently - aim for at least 3 days per week")
        
        # Check session duration
        avg_duration = sum(s.duration_minutes for s in sessions[-5:]) / len(sessions[-5:])
        if avg_duration < 30:
            recommendations.append("Consider longer study sessions (30+ minutes) for better learning retention")
        
        # Check engagement levels
        avg_engagement = sum(s.engagement_score for s in sessions[-5:]) / len(sessions[-5:])
        if avg_engagement < 0.6:
            recommendations.append("Try different learning activities to increase engagement")
        
        # Subject-specific recommendations
        subject_counts = {}
        for session in sessions:
            subject_counts[session.subject] = subject_counts.get(session.subject, 0) + 1
        
        if len(subject_counts) == 1:
            recommendations.append("Consider exploring related subjects to broaden your knowledge")
        
        return recommendations[:5]  # Return max 5 recommendations

# ============================================================================
# LEARNING ENDPOINTS
# ============================================================================

# Initialize learning service
learning_service = LearningService()

@router.post("/goals", response_model=LearningGoal)
async def create_learning_goal(
    goal_data: Dict[str, Any],
    current_user: UserProfile = Depends(require_permission("learning:write"))
):
    """Create a new learning goal"""
    
    try:
        goal = await learning_service.create_learning_goal(current_user.user_id, goal_data)
        return goal
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Create learning goal endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Learning goal creation error")

@router.get("/goals", response_model=List[LearningGoal])
async def get_learning_goals(
    current_user: UserProfile = Depends(require_permission("learning:read"))
):
    """Get all learning goals for the current user"""
    
    try:
        goals = await learning_service.get_learning_goals(current_user.user_id)
        return goals
        
    except Exception as e:
        logger.error(f"Get learning goals endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Learning goals retrieval error")

@router.put("/goals/{goal_id}", response_model=LearningGoal)
async def update_learning_goal(
    goal_id: str,
    updates: Dict[str, Any],
    current_user: UserProfile = Depends(require_permission("learning:write"))
):
    """Update a learning goal"""
    
    try:
        goal = await learning_service.update_learning_goal(goal_id, current_user.user_id, updates)
        return goal
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update learning goal endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Learning goal update error")

@router.post("/sessions", response_model=LearningSession)
async def record_learning_session(
    session_data: Dict[str, Any],
    current_user: UserProfile = Depends(require_permission("learning:write"))
):
    """Record a learning session"""
    
    try:
        session = await learning_service.record_learning_session(current_user.user_id, session_data)
        return session
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Record learning session endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Learning session recording error")

@router.post("/progress", response_model=ProgressResponse)
async def get_learning_progress(
    progress_request: ProgressRequest,
    current_user: UserProfile = Depends(require_permission("learning:read"))
):
    """Get comprehensive learning progress"""
    
    try:
        # Override user_id with current user
        progress_request.user_id = current_user.user_id
        
        progress = await learning_service.get_learning_progress(current_user.user_id, progress_request)
        return progress
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get learning progress endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Learning progress retrieval error")
