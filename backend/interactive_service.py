"""
🚀 REVOLUTIONARY INTERACTIVE CONTENT SERVICE
Advanced service layer for premium interactive experiences

This service handles:
- Interactive content generation and management
- Real-time collaboration
- Performance optimization
- Advanced analytics

Author: MasterX Quantum Intelligence Team
Version: 3.0 - Production Ready
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import weakref
from dataclasses import dataclass
import uuid

# Third-party imports
try:
    import numpy as np
    import pandas as pd
    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False
    logging.warning("NumPy/Pandas not available. Advanced analytics disabled.")

# Local imports
from interactive_models import (
    EnhancedMessage, MessageType, CodeBlockContent, ChartContent,
    DiagramContent, CalculatorContent, WhiteboardContent, 
    VisualizationContent, QuizContent, MathEquationContent,
    CollaborationSession, WhiteboardOperation, ChartDataUpdate,
    InteractiveContentAnalytics, ContentValidator
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring"""
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    memory_used_mb: Optional[float] = None
    cpu_usage: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None


class PerformanceMonitor:
    """Monitor performance of interactive content operations"""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.active_operations: Dict[str, PerformanceMetrics] = {}
    
    def start_operation(self, operation_name: str) -> str:
        """Start monitoring an operation"""
        operation_id = str(uuid.uuid4())
        metric = PerformanceMetrics(
            operation_name=operation_name,
            start_time=datetime.utcnow()
        )
        self.active_operations[operation_id] = metric
        return operation_id
    
    def end_operation(self, operation_id: str, success: bool = True, error: str = None):
        """End monitoring an operation"""
        if operation_id in self.active_operations:
            metric = self.active_operations[operation_id]
            metric.end_time = datetime.utcnow()
            metric.duration_ms = (metric.end_time - metric.start_time).total_seconds() * 1000
            metric.success = success
            metric.error_message = error
            
            self.metrics.append(metric)
            del self.active_operations[operation_id]
            
            # Log performance data
            if success:
                logger.info(f"Operation {metric.operation_name} completed in {metric.duration_ms:.2f}ms")
            else:
                logger.error(f"Operation {metric.operation_name} failed after {metric.duration_ms:.2f}ms: {error}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
        if not self.metrics:
            return {"message": "No metrics available"}
        
        successful_ops = [m for m in self.metrics if m.success]
        failed_ops = [m for m in self.metrics if not m.success]
        
        return {
            "total_operations": len(self.metrics),
            "successful_operations": len(successful_ops),
            "failed_operations": len(failed_ops),
            "success_rate": len(successful_ops) / len(self.metrics) * 100,
            "average_duration_ms": sum(m.duration_ms for m in successful_ops) / len(successful_ops) if successful_ops else 0,
            "operations_by_type": self._group_by_operation_type(),
            "recent_errors": [m.error_message for m in failed_ops[-5:]]
        }
    
    def _group_by_operation_type(self) -> Dict[str, int]:
        """Group metrics by operation type"""
        groups = {}
        for metric in self.metrics:
            groups[metric.operation_name] = groups.get(metric.operation_name, 0) + 1
        return groups


# ============================================================================
# CONTENT GENERATORS
# ============================================================================

class CodeContentGenerator:
    """Generate advanced code content with syntax highlighting"""
    
    LANGUAGE_TEMPLATES = {
        "python": {
            "hello_world": 'print("Hello, World!")',
            "fibonacci": '''def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Generate first 10 Fibonacci numbers
for i in range(10):
    print(f"F({i}) = {fibonacci(i)}")''',
            "data_analysis": '''import pandas as pd
import matplotlib.pyplot as plt

# Load and analyze data
data = pd.read_csv('data.csv')
print(data.describe())

# Create visualization
plt.figure(figsize=(10, 6))
plt.plot(data['x'], data['y'])
plt.title('Data Analysis Results')
plt.show()'''
        },
        "javascript": {
            "hello_world": 'console.log("Hello, World!");',
            "async_fetch": '''async function fetchData(url) {
    try {
        const response = await fetch(url);
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error fetching data:', error);
        throw error;
    }
}

// Usage
fetchData('https://api.example.com/data')
    .then(data => console.log(data))
    .catch(error => console.error(error));''',
            "react_component": '''import React, { useState, useEffect } from 'react';

const DataComponent = ({ apiUrl }) => {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await fetch(apiUrl);
                const result = await response.json();
                setData(result);
            } catch (err) {
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, [apiUrl]);

    if (loading) return <div>Loading...</div>;
    if (error) return <div>Error: {error}</div>;

    return (
        <div>
            <h2>Data Results</h2>
            <pre>{JSON.stringify(data, null, 2)}</pre>
        </div>
    );
};

export default DataComponent;'''
        }
    }
    
    @staticmethod
    def generate_code_example(language: str, example_type: str = "hello_world") -> CodeBlockContent:
        """Generate code example with syntax highlighting"""
        templates = CodeContentGenerator.LANGUAGE_TEMPLATES.get(language, {})
        code = templates.get(example_type, f'// {language} example\nconsole.log("Example code");')
        
        return CodeBlockContent(
            language=language,
            code=code,
            title=f"{language.title()} - {example_type.replace('_', ' ').title()}",
            is_executable=language in ["python", "javascript"],
            theme="vs-dark",
            line_numbers=True,
            allow_editing=True,
            show_output=True
        )
    
    @staticmethod
    def create_interactive_tutorial(language: str, topic: str) -> List[CodeBlockContent]:
        """Create multi-step interactive coding tutorial"""
        tutorials = []
        
        if language == "python" and topic == "data_science":
            steps = [
                ("Import Libraries", "import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt"),
                ("Load Data", "data = pd.read_csv('dataset.csv')\nprint(data.head())"),
                ("Data Analysis", "print(data.describe())\nprint(data.info())"),
                ("Visualization", "plt.figure(figsize=(10, 6))\nplt.plot(data['x'], data['y'])\nplt.title('Data Visualization')\nplt.show()"),
                ("Machine Learning", "from sklearn.linear_model import LinearRegression\nmodel = LinearRegression()\nmodel.fit(X_train, y_train)")
            ]
            
            for i, (title, code) in enumerate(steps):
                tutorials.append(CodeBlockContent(
                    language="python",
                    code=code,
                    title=f"Step {i+1}: {title}",
                    is_executable=True,
                    allow_editing=True,
                    metadata={"step": i+1, "total_steps": len(steps)}
                ))
        
        return tutorials


class ChartContentGenerator:
    """Generate interactive chart content"""
    
    @staticmethod
    def create_sample_data(chart_type: str, data_points: int = 10) -> Dict[str, Any]:
        """Create sample data for different chart types"""
        if not ANALYTICS_AVAILABLE:
            # Fallback data without NumPy
            return {
                "labels": [f"Item {i+1}" for i in range(data_points)],
                "datasets": [{
                    "label": "Sample Data",
                    "data": [i * 10 + (i % 3) * 5 for i in range(data_points)],
                    "backgroundColor": ["#8B5CF6", "#06B6D4", "#10B981", "#F59E0B", "#EF4444"]
                }]
            }
        
        # Generate realistic sample data with NumPy
        np.random.seed(42)  # For reproducible results
        
        if chart_type == "line":
            x = np.linspace(0, 10, data_points)
            y = np.sin(x) + np.random.normal(0, 0.1, data_points)
            return {
                "labels": [f"{x:.1f}" for x in x],
                "datasets": [{
                    "label": "Sine Wave with Noise",
                    "data": y.tolist(),
                    "borderColor": "#8B5CF6",
                    "backgroundColor": "rgba(139, 92, 246, 0.1)"
                }]
            }
        
        elif chart_type == "bar":
            categories = ["Q1", "Q2", "Q3", "Q4"]
            values = np.random.randint(50, 200, len(categories))
            return {
                "labels": categories,
                "datasets": [{
                    "label": "Quarterly Revenue ($K)",
                    "data": values.tolist(),
                    "backgroundColor": ["#8B5CF6", "#06B6D4", "#10B981", "#F59E0B"]
                }]
            }
        
        elif chart_type == "pie":
            categories = ["Desktop", "Mobile", "Tablet", "Other"]
            values = np.random.dirichlet(np.ones(len(categories))) * 100
            return {
                "labels": categories,
                "datasets": [{
                    "data": values.tolist(),
                    "backgroundColor": ["#8B5CF6", "#06B6D4", "#10B981", "#F59E0B"]
                }]
            }
        
        elif chart_type == "scatter":
            x = np.random.normal(50, 15, data_points)
            y = 2 * x + np.random.normal(0, 10, data_points)
            return {
                "datasets": [{
                    "label": "Correlation Data",
                    "data": [{"x": x[i], "y": y[i]} for i in range(data_points)],
                    "backgroundColor": "#8B5CF6"
                }]
            }
        
        # Default fallback
        return {
            "labels": [f"Category {i+1}" for i in range(data_points)],
            "datasets": [{
                "label": "Sample Dataset",
                "data": np.random.randint(10, 100, data_points).tolist(),
                "backgroundColor": "#8B5CF6"
            }]
        }
    
    @staticmethod
    def create_real_time_chart(chart_type: str, update_interval: int = 1000) -> ChartContent:
        """Create real-time updating chart"""
        data = ChartContentGenerator.create_sample_data(chart_type)
        
        return ChartContent(
            chart_type=chart_type,
            data=data,
            title=f"Real-time {chart_type.title()} Chart",
            auto_refresh=True,
            refresh_interval=update_interval,
            enable_zoom=True,
            enable_pan=True,
            animation_duration=750,
            metadata={
                "real_time": True,
                "update_frequency": f"{update_interval}ms"
            }
        )


class DiagramContentGenerator:
    """Generate interactive diagrams and flowcharts"""
    
    @staticmethod
    def create_flowchart(title: str, steps: List[str]) -> DiagramContent:
        """Create interactive flowchart"""
        nodes = []
        edges = []
        
        for i, step in enumerate(steps):
            node_id = f"step_{i}"
            nodes.append({
                "id": node_id,
                "label": step,
                "type": "rectangle",
                "x": 100,
                "y": i * 100 + 50
            })
            
            if i > 0:
                edges.append({
                    "from": f"step_{i-1}",
                    "to": node_id,
                    "arrow": True
                })
        
        return DiagramContent(
            diagram_type="flowchart",
            title=title,
            nodes=nodes,
            edges=edges,
            enable_drag=True,
            enable_zoom=True,
            auto_layout=True
        )
    
    @staticmethod
    def create_mind_map(central_topic: str, branches: Dict[str, List[str]]) -> DiagramContent:
        """Create interactive mind map"""
        nodes = [{"id": "center", "label": central_topic, "type": "circle", "x": 400, "y": 300}]
        edges = []
        
        angle_step = 360 / len(branches)
        
        for i, (branch_name, sub_items) in enumerate(branches.items()):
            branch_id = f"branch_{i}"
            angle = i * angle_step
            x = 400 + 200 * np.cos(np.radians(angle)) if ANALYTICS_AVAILABLE else 400 + 200 * (0.5 - i/len(branches))
            y = 300 + 200 * np.sin(np.radians(angle)) if ANALYTICS_AVAILABLE else 300 + 100 * i
            
            nodes.append({
                "id": branch_id,
                "label": branch_name,
                "type": "rectangle",
                "x": x,
                "y": y
            })
            
            edges.append({
                "from": "center",
                "to": branch_id,
                "arrow": False
            })
            
            # Add sub-items
            for j, sub_item in enumerate(sub_items):
                sub_id = f"sub_{i}_{j}"
                sub_x = x + 150 * (j % 2 * 2 - 1)
                sub_y = y + 80 * (j // 2 + 1)
                
                nodes.append({
                    "id": sub_id,
                    "label": sub_item,
                    "type": "ellipse",
                    "x": sub_x,
                    "y": sub_y
                })
                
                edges.append({
                    "from": branch_id,
                    "to": sub_id,
                    "arrow": False
                })
        
        return DiagramContent(
            diagram_type="mind_map",
            title=f"Mind Map: {central_topic}",
            nodes=nodes,
            edges=edges,
            enable_drag=True,
            enable_zoom=True
        )


# ============================================================================
# MAIN INTERACTIVE SERVICE
# ============================================================================

class InteractiveContentService:
    """Main service for managing interactive content"""
    
    def __init__(self, db_service=None):
        self.db = db_service
        self.performance_monitor = PerformanceMonitor()
        self.collaboration_sessions: Dict[str, CollaborationSession] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.content_cache = weakref.WeakValueDictionary()
        
        # Content generators
        self.code_generator = CodeContentGenerator()
        self.chart_generator = ChartContentGenerator()
        self.diagram_generator = DiagramContentGenerator()
        
        logger.info("Interactive Content Service initialized")
    
    async def create_interactive_content(
        self,
        message_id: str,
        content_type: MessageType,
        content_data: Dict[str, Any],
        user_id: str = "anonymous"
    ) -> Union[CodeBlockContent, ChartContent, DiagramContent, CalculatorContent, WhiteboardContent]:
        """Create interactive content based on type and data"""
        
        operation_id = self.performance_monitor.start_operation(f"create_{content_type.value}")
        
        try:
            # Validate content data
            if not ContentValidator.validate_content_by_type(content_type, content_data):
                raise ValueError(f"Invalid content data for type {content_type}")
            
            # Create content based on type
            if content_type == MessageType.CODE:
                content = await self._create_code_content(content_data)
            elif content_type == MessageType.CHART:
                content = await self._create_chart_content(content_data)
            elif content_type == MessageType.DIAGRAM:
                content = await self._create_diagram_content(content_data)
            elif content_type == MessageType.CALCULATOR:
                content = await self._create_calculator_content(content_data)
            elif content_type == MessageType.WHITEBOARD:
                content = await self._create_whiteboard_content(content_data)
            elif content_type == MessageType.VISUALIZATION:
                content = await self._create_visualization_content(content_data)
            elif content_type == MessageType.QUIZ:
                content = await self._create_quiz_content(content_data)
            elif content_type == MessageType.MATH_EQUATION:
                content = await self._create_math_content(content_data)
            else:
                raise ValueError(f"Unsupported content type: {content_type}")
            
            # Cache content for performance
            self.content_cache[content.content_id] = content
            
            # Store in database if available
            if self.db:
                await self._store_content_in_db(message_id, content, user_id)
            
            self.performance_monitor.end_operation(operation_id, success=True)
            logger.info(f"Created interactive content: {content_type.value} - {content.content_id}")
            
            return content
            
        except Exception as e:
            self.performance_monitor.end_operation(operation_id, success=False, error=str(e))
            logger.error(f"Failed to create interactive content: {str(e)}")
            raise
    
    async def _create_code_content(self, data: Dict[str, Any]) -> CodeBlockContent:
        """Create code block content"""
        return CodeBlockContent(
            language=data.get("language", "python"),
            code=data.get("code", ""),
            title=data.get("title", "Code Example"),
            is_executable=data.get("is_executable", False),
            theme=data.get("theme", "vs-dark"),
            line_numbers=data.get("line_numbers", True),
            allow_editing=data.get("allow_editing", False)
        )
    
    async def _create_chart_content(self, data: Dict[str, Any]) -> ChartContent:
        """Create chart content"""
        chart_type = data.get("chart_type", "line")
        chart_data = data.get("data")
        
        if not chart_data:
            chart_data = self.chart_generator.create_sample_data(chart_type)
        
        return ChartContent(
            chart_type=chart_type,
            data=chart_data,
            title=data.get("title", f"Interactive {chart_type.title()} Chart"),
            enable_zoom=data.get("enable_zoom", True),
            enable_pan=data.get("enable_pan", True),
            animation_duration=data.get("animation_duration", 750)
        )
    
    async def _create_diagram_content(self, data: Dict[str, Any]) -> DiagramContent:
        """Create diagram content"""
        return DiagramContent(
            diagram_type=data.get("diagram_type", "flowchart"),
            nodes=data.get("nodes", []),
            edges=data.get("edges", []),
            title=data.get("title", "Interactive Diagram"),
            enable_drag=data.get("enable_drag", True),
            enable_zoom=data.get("enable_zoom", True)
        )
    
    async def _create_calculator_content(self, data: Dict[str, Any]) -> CalculatorContent:
        """Create calculator content"""
        return CalculatorContent(
            calculator_type=data.get("calculator_type", "basic"),
            title=data.get("title", "Interactive Calculator"),
            initial_values=data.get("initial_values", {}),
            formulas=data.get("formulas", {}),
            show_history=data.get("show_history", True)
        )
    
    async def _create_whiteboard_content(self, data: Dict[str, Any]) -> WhiteboardContent:
        """Create whiteboard content"""
        return WhiteboardContent(
            title=data.get("title", "Collaborative Whiteboard"),
            width=data.get("width", 1200),
            height=data.get("height", 800),
            max_participants=data.get("max_participants", 10),
            real_time_sync=data.get("real_time_sync", True)
        )
    
    async def _create_visualization_content(self, data: Dict[str, Any]) -> VisualizationContent:
        """Create visualization content"""
        return VisualizationContent(
            visualization_type=data.get("visualization_type", "3d_scatter"),
            data=data.get("data", {}),
            title=data.get("title", "Advanced Visualization"),
            config=data.get("config", {})
        )
    
    async def _create_quiz_content(self, data: Dict[str, Any]) -> QuizContent:
        """Create quiz content"""
        return QuizContent(
            questions=data.get("questions", []),
            quiz_type=data.get("quiz_type", "multiple_choice"),
            title=data.get("title", "Interactive Quiz"),
            time_limit=data.get("time_limit"),
            show_correct_answers=data.get("show_correct_answers", True)
        )
    
    async def _create_math_content(self, data: Dict[str, Any]) -> MathEquationContent:
        """Create math equation content"""
        return MathEquationContent(
            latex=data.get("latex", ""),
            title=data.get("title", "Mathematical Equation"),
            variables=data.get("variables", {}),
            interactive_variables=data.get("interactive_variables", []),
            enable_graphing=data.get("enable_graphing", False)
        )
    
    async def _store_content_in_db(self, message_id: str, content: Any, user_id: str):
        """Store interactive content in database"""
        if self.db:
            try:
                await self.db.interactive_content.insert_one({
                    "message_id": message_id,
                    "content_id": content.content_id,
                    "content_type": content.content_type.value,
                    "content_data": content.dict(),
                    "created_by": user_id,
                    "created_at": datetime.utcnow()
                })
            except Exception as e:
                logger.error(f"Failed to store content in database: {str(e)}")
    
    async def start_collaboration_session(
        self,
        content_id: str,
        participant_ids: List[str],
        max_participants: int = 10
    ) -> CollaborationSession:
        """Start real-time collaboration session"""
        
        session = CollaborationSession(
            content_id=content_id,
            content_type=MessageType.WHITEBOARD,  # Default, should be dynamic
            max_participants=max_participants
        )
        
        # Add participants
        for participant_id in participant_ids:
            session.participants.append({
                "user_id": participant_id,
                "joined_at": datetime.utcnow(),
                "permissions": {"read": True, "write": True}
            })
        
        self.collaboration_sessions[session.session_id] = session
        
        logger.info(f"Started collaboration session: {session.session_id}")
        return session
    
    async def process_whiteboard_operation(
        self,
        session_id: str,
        operation: WhiteboardOperation
    ) -> bool:
        """Process whiteboard operation and broadcast to participants"""
        
        if session_id not in self.collaboration_sessions:
            logger.error(f"Collaboration session not found: {session_id}")
            return False
        
        session = self.collaboration_sessions[session_id]
        
        # Validate operation
        if not ContentValidator.validate_whiteboard_operation(operation.dict()):
            logger.error("Invalid whiteboard operation")
            return False
        
        # Add operation to session history
        session.operations.append(operation.dict())
        session.last_activity = datetime.utcnow()
        
        # Here you would broadcast to all participants via WebSocket
        # This is a placeholder for the actual WebSocket broadcasting logic
        
        logger.info(f"Processed whiteboard operation: {operation.operation_type}")
        return True
    
    async def get_content_analytics(self, content_id: str) -> InteractiveContentAnalytics:
        """Get analytics for interactive content"""
        
        # This would typically fetch from database and compute metrics
        # For now, return sample analytics
        
        return InteractiveContentAnalytics(
            content_id=content_id,
            content_type=MessageType.CHART,  # Would be dynamic
            total_views=150,
            unique_viewers=45,
            total_interactions=230,
            average_session_duration=125.5,
            completion_rate=0.78,
            insights=[
                "High engagement rate indicates effective content",
                "Most interactions occur in first 30 seconds",
                "Mobile users have 20% lower completion rate"
            ],
            recommendations=[
                "Optimize for mobile viewing",
                "Add interactive elements early in content",
                "Consider adding progress indicators"
            ]
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get service performance metrics"""
        return self.performance_monitor.get_metrics_summary()
    
    async def cleanup_expired_sessions(self):
        """Clean up expired collaboration sessions"""
        current_time = datetime.utcnow()
        expired_sessions = []
        
        for session_id, session in self.collaboration_sessions.items():
            if session.expires_at and current_time > session.expires_at:
                expired_sessions.append(session_id)
            elif not session.is_active and (current_time - session.last_activity).hours > 24:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.collaboration_sessions[session_id]
            logger.info(f"Cleaned up expired session: {session_id}")
    
    async def close(self):
        """Close service and cleanup resources"""
        self.executor.shutdown(wait=True)
        await self.cleanup_expired_sessions()
        logger.info("Interactive Content Service closed")


# ============================================================================
# CONTENT VALIDATION EXTENSIONS
# ============================================================================

class ContentValidator:
    """Extended content validator with comprehensive checks"""
    
    @staticmethod
    def validate_content_by_type(content_type: MessageType, data: Dict[str, Any]) -> bool:
        """Validate content data by type"""
        
        if content_type == MessageType.CODE:
            return ContentValidator.validate_code_content(data)
        elif content_type == MessageType.CHART:
            return ContentValidator.validate_chart_content(data)
        elif content_type == MessageType.DIAGRAM:
            return ContentValidator.validate_diagram_content(data)
        elif content_type == MessageType.WHITEBOARD:
            return ContentValidator.validate_whiteboard_content(data)
        elif content_type == MessageType.QUIZ:
            return ContentValidator.validate_quiz_content(data)
        
        return True  # Default validation for other types
    
    @staticmethod
    def validate_code_content(data: Dict[str, Any]) -> bool:
        """Validate code content"""
        required_fields = ['language', 'code']
        return all(field in data for field in required_fields) and bool(data.get('code', '').strip())
    
    @staticmethod
    def validate_chart_content(data: Dict[str, Any]) -> bool:
        """Validate chart content"""
        chart_data = data.get('data', {})
        return 'datasets' in chart_data or 'labels' in chart_data
    
    @staticmethod
    def validate_diagram_content(data: Dict[str, Any]) -> bool:
        """Validate diagram content"""
        nodes = data.get('nodes', [])
        return isinstance(nodes, list) and all('id' in node for node in nodes)
    
    @staticmethod
    def validate_whiteboard_content(data: Dict[str, Any]) -> bool:
        """Validate whiteboard content"""
        return 'width' in data and 'height' in data
    
    @staticmethod
    def validate_quiz_content(data: Dict[str, Any]) -> bool:
        """Validate quiz content"""
        questions = data.get('questions', [])
        return isinstance(questions, list) and len(questions) > 0
    
    @staticmethod
    def validate_whiteboard_operation(data: Dict[str, Any]) -> bool:
        """Validate whiteboard operation"""
        required_fields = ['operation_type', 'data', 'user_id']
        return all(field in data for field in required_fields)


# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    'InteractiveContentService',
    'CodeContentGenerator',
    'ChartContentGenerator', 
    'DiagramContentGenerator',
    'PerformanceMonitor',
    'ContentValidator'
]