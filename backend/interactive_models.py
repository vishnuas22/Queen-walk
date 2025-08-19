"""
🚀 REVOLUTIONARY INTERACTIVE MESSAGE MODELS
Advanced data structures for premium interactive experiences

Features:
- Code blocks with syntax highlighting
- Interactive diagrams and charts
- Embedded mini-apps and calculators
- Real-time collaborative whiteboards
- Advanced visualizations

Author: MasterX Quantum Intelligence Team
Version: 3.0 - Production Ready
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union, Literal, ClassVar
from datetime import datetime
from enum import Enum
import uuid


class MessageType(str, Enum):
    """Enhanced message types for premium experiences"""
    TEXT = "text"
    CODE = "code"
    CHART = "chart"
    DIAGRAM = "diagram"
    CALCULATOR = "calculator"
    WHITEBOARD = "whiteboard"
    QUIZ = "quiz"
    VISUALIZATION = "visualization"
    INTERACTIVE_DEMO = "interactive_demo"
    MATH_EQUATION = "math_equation"
    MINDMAP = "mindmap"
    TIMELINE = "timeline"


class CodeBlockType(str, Enum):
    """Supported programming languages with syntax highlighting"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    HTML = "html"
    CSS = "css"
    SQL = "sql"
    BASH = "bash"
    JSON = "json"
    YAML = "yaml"
    MARKDOWN = "markdown"
    REACT = "react"
    VUE = "vue"
    GO = "go"
    RUST = "rust"
    JAVA = "java"
    CSHARP = "csharp"
    CPP = "cpp"
    PHP = "php"
    RUBY = "ruby"
    SWIFT = "swift"
    KOTLIN = "kotlin"


class ChartType(str, Enum):
    """Interactive chart types"""
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    SCATTER = "scatter"
    AREA = "area"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    BUBBLE = "bubble"
    RADAR = "radar"
    TREEMAP = "treemap"
    CANDLESTICK = "candlestick"
    FUNNEL = "funnel"


class DiagramType(str, Enum):
    """Interactive diagram types"""
    FLOWCHART = "flowchart"
    MIND_MAP = "mind_map"
    NETWORK = "network"
    TREE = "tree"
    TIMELINE = "timeline"
    SEQUENCE = "sequence"
    CLASS_DIAGRAM = "class_diagram"
    ER_DIAGRAM = "er_diagram"
    ARCHITECTURE = "architecture"
    PROCESS_FLOW = "process_flow"


class CalculatorType(str, Enum):
    """Mini calculator types"""
    BASIC = "basic"
    SCIENTIFIC = "scientific"
    FINANCIAL = "financial"
    UNIT_CONVERTER = "unit_converter"
    MORTGAGE = "mortgage"
    STATISTICS = "statistics"
    PROGRAMMING = "programming"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"


# ============================================================================
# CORE MESSAGE MODELS
# ============================================================================

class InteractiveContent(BaseModel):
    """Base model for all interactive content"""
    content_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content_type: MessageType
    title: Optional[str] = None
    description: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_collaborative: bool = False
    permissions: Dict[str, Any] = Field(default_factory=dict)


class CodeBlockContent(InteractiveContent):
    """Enhanced code block with syntax highlighting and execution"""
    content_type: Literal[MessageType.CODE] = MessageType.CODE
    language: CodeBlockType
    code: str
    is_executable: bool = False
    execution_environment: Optional[str] = None
    expected_output: Optional[str] = None
    test_cases: List[Dict[str, Any]] = Field(default_factory=list)
    imports: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    
    # Syntax highlighting configuration
    theme: str = "vs-dark"
    line_numbers: bool = True
    word_wrap: bool = True
    auto_format: bool = True
    
    # Interactive features
    allow_editing: bool = False
    show_output: bool = True
    enable_collaboration: bool = False
    
    @validator('code')
    def validate_code(cls, v):
        """Validate code content"""
        if not v or not v.strip():
            raise ValueError("Code content cannot be empty")
        return v.strip()


class ChartContent(InteractiveContent):
    """Interactive chart with real-time data"""
    content_type: Literal[MessageType.CHART] = MessageType.CHART
    chart_type: ChartType
    data: Dict[str, Any]
    config: Dict[str, Any] = Field(default_factory=dict)
    
    # Chart styling
    theme: str = "quantum"
    colors: List[str] = Field(default_factory=lambda: [
        "#8B5CF6", "#06B6D4", "#10B981", "#F59E0B", "#EF4444", "#EC4899"
    ])
    
    # Interactive features
    enable_zoom: bool = True
    enable_pan: bool = True
    enable_selection: bool = True
    enable_crossfilter: bool = False
    animation_duration: int = 750
    
    # Real-time updates
    auto_refresh: bool = False
    refresh_interval: int = 5000  # milliseconds
    data_source: Optional[str] = None
    
    @validator('data')
    def validate_chart_data(cls, v):
        """Validate chart data structure"""
        if not v:
            raise ValueError("Chart data cannot be empty")
        if 'labels' not in v and 'datasets' not in v:
            raise ValueError("Chart data must contain labels and datasets")
        return v


class DiagramContent(InteractiveContent):
    """Interactive diagrams and flowcharts"""
    content_type: Literal[MessageType.DIAGRAM] = MessageType.DIAGRAM
    diagram_type: DiagramType
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    edges: List[Dict[str, Any]] = Field(default_factory=list)
    layout: Dict[str, Any] = Field(default_factory=dict)
    
    # Styling configuration
    node_style: Dict[str, Any] = Field(default_factory=lambda: {
        "shape": "rectangle",
        "color": "#8B5CF6",
        "border_color": "#7C3AED",
        "text_color": "#FFFFFF"
    })
    edge_style: Dict[str, Any] = Field(default_factory=lambda: {
        "color": "#64748B",
        "width": 2,
        "arrow": True
    })
    
    # Interactive features
    enable_drag: bool = True
    enable_zoom: bool = True
    enable_selection: bool = True
    auto_layout: bool = True
    
    @validator('nodes')
    def validate_nodes(cls, v):
        """Validate diagram nodes"""
        for node in v:
            if 'id' not in node:
                raise ValueError("Each node must have an 'id' field")
        return v


class CalculatorContent(InteractiveContent):
    """Interactive mini calculators"""
    content_type: Literal[MessageType.CALCULATOR] = MessageType.CALCULATOR
    calculator_type: CalculatorType
    initial_values: Dict[str, Any] = Field(default_factory=dict)
    formulas: Dict[str, str] = Field(default_factory=dict)
    variables: Dict[str, Any] = Field(default_factory=dict)
    
    # UI Configuration
    layout: str = "vertical"  # vertical, horizontal, grid
    theme: str = "quantum"
    show_history: bool = True
    show_steps: bool = True
    precision: int = 6
    
    # Feature flags
    allow_custom_formulas: bool = False
    enable_graphing: bool = False
    enable_export: bool = True


class WhiteboardContent(InteractiveContent):
    """Real-time collaborative whiteboard"""
    content_type: Literal[MessageType.WHITEBOARD] = MessageType.WHITEBOARD
    canvas_data: Dict[str, Any] = Field(default_factory=dict)
    tools: List[str] = Field(default_factory=lambda: [
        "pen", "eraser", "text", "shapes", "sticky_notes"
    ])
    
    # Canvas configuration
    width: int = 1200
    height: int = 800
    background_color: str = "#1E293B"
    grid_enabled: bool = True
    grid_size: int = 20
    
    # Collaboration settings
    max_participants: int = 10
    participant_cursors: bool = True
    real_time_sync: bool = True
    version_history: bool = True
    
    # Drawing tools configuration
    pen_colors: List[str] = Field(default_factory=lambda: [
        "#FFFFFF", "#EF4444", "#10B981", "#3B82F6", "#F59E0B", "#8B5CF6"
    ])
    pen_sizes: List[int] = Field(default_factory=lambda: [1, 2, 4, 8, 16])
    
    is_collaborative: bool = True


class VisualizationContent(InteractiveContent):
    """Advanced data visualizations"""
    content_type: Literal[MessageType.VISUALIZATION] = MessageType.VISUALIZATION
    visualization_type: str
    data: Dict[str, Any]
    config: Dict[str, Any] = Field(default_factory=dict)
    
    # Supported visualization types
    SUPPORTED_TYPES: ClassVar[List[str]] = [
        "3d_scatter", "network_graph", "treemap", "sunburst", 
        "parallel_coordinates", "sankey", "chord_diagram", "force_directed"
    ]
    
    # Interactive features
    enable_brushing: bool = True
    enable_linking: bool = True
    enable_filtering: bool = True
    enable_annotations: bool = True


class QuizContent(InteractiveContent):
    """Interactive quizzes and assessments"""
    content_type: Literal[MessageType.QUIZ] = MessageType.QUIZ
    questions: List[Dict[str, Any]]
    quiz_type: str = "multiple_choice"  # multiple_choice, true_false, short_answer, code
    time_limit: Optional[int] = None  # seconds
    randomize_questions: bool = False
    randomize_options: bool = False
    
    # Scoring configuration
    scoring_method: str = "weighted"  # simple, weighted, partial_credit
    pass_threshold: float = 0.7
    allow_retries: bool = True
    max_retries: int = 3
    
    # Feedback settings
    show_correct_answers: bool = True
    show_explanations: bool = True
    immediate_feedback: bool = False


class MathEquationContent(InteractiveContent):
    """Interactive mathematical equations with LaTeX support"""
    content_type: Literal[MessageType.MATH_EQUATION] = MessageType.MATH_EQUATION
    latex: str
    variables: Dict[str, Any] = Field(default_factory=dict)
    interactive_variables: List[str] = Field(default_factory=list)
    
    # Rendering configuration
    renderer: str = "katex"  # katex, mathjax
    display_mode: bool = False
    font_size: str = "16px"
    
    # Interactive features
    enable_graphing: bool = False
    enable_manipulation: bool = False
    show_steps: bool = False


# ============================================================================
# ENHANCED MESSAGE MODEL
# ============================================================================

class EnhancedMessage(BaseModel):
    """Revolutionary message model with interactive content support"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    message: str
    sender: str  # "user" or "ai"
    message_type: MessageType = MessageType.TEXT
    
    # Interactive content
    interactive_content: Optional[Union[
        CodeBlockContent,
        ChartContent, 
        DiagramContent,
        CalculatorContent,
        WhiteboardContent,
        VisualizationContent,
        QuizContent,
        MathEquationContent
    ]] = None
    
    # Enhanced metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    learning_insights: Dict[str, Any] = Field(default_factory=dict)
    personalization_data: Dict[str, Any] = Field(default_factory=dict)
    
    # Timestamps and versioning
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    version: int = 1
    
    # Collaboration and sharing
    is_shared: bool = False
    shared_with: List[str] = Field(default_factory=list)
    collaboration_id: Optional[str] = None
    
    # Analytics and tracking
    view_count: int = 0
    interaction_count: int = 0
    feedback_score: Optional[float] = None
    tags: List[str] = Field(default_factory=list)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class CreateInteractiveContentRequest(BaseModel):
    """Request model for creating interactive content"""
    message_id: str
    content_type: MessageType
    content_data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class UpdateInteractiveContentRequest(BaseModel):
    """Request model for updating interactive content"""
    content_id: str
    updates: Dict[str, Any]
    collaboration_token: Optional[str] = None


class CollaborationSessionRequest(BaseModel):
    """Request model for starting collaboration session"""
    content_id: str
    participant_ids: List[str]
    permissions: Dict[str, Any] = Field(default_factory=dict)


class WhiteboardOperation(BaseModel):
    """Individual whiteboard operation for real-time sync"""
    operation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    operation_type: str  # draw, erase, text, shape, clear
    data: Dict[str, Any]
    user_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChartDataUpdate(BaseModel):
    """Real-time chart data update"""
    chart_id: str
    data_points: List[Dict[str, Any]]
    update_type: str = "append"  # append, replace, update
    animation: bool = True


# ============================================================================
# COLLABORATION MODELS
# ============================================================================

class CollaborationSession(BaseModel):
    """Real-time collaboration session"""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content_id: str
    content_type: MessageType
    participants: List[Dict[str, Any]] = Field(default_factory=list)
    active_participants: List[str] = Field(default_factory=list)
    operations: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Session configuration
    max_participants: int = 10
    allow_anonymous: bool = False
    require_approval: bool = False
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    
    # Status
    is_active: bool = True
    is_locked: bool = False


class ParticipantCursor(BaseModel):
    """Real-time cursor position for collaboration"""
    user_id: str
    user_name: str
    x: float
    y: float
    color: str = "#8B5CF6"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# ANALYTICS AND INSIGHTS MODELS
# ============================================================================

class InteractiveContentAnalytics(BaseModel):
    """Analytics for interactive content performance"""
    content_id: str
    content_type: MessageType
    
    # Engagement metrics
    total_views: int = 0
    unique_viewers: int = 0
    total_interactions: int = 0
    average_session_duration: float = 0.0
    
    # Performance metrics
    load_time_ms: float = 0.0
    error_rate: float = 0.0
    completion_rate: float = 0.0
    
    # User behavior
    interaction_heatmap: Dict[str, Any] = Field(default_factory=dict)
    user_feedback: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Time-based analytics
    hourly_stats: Dict[str, int] = Field(default_factory=dict)
    daily_stats: Dict[str, int] = Field(default_factory=dict)
    
    # Generated insights
    insights: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_default_code_content(language: CodeBlockType, code: str) -> CodeBlockContent:
    """Create default code block content"""
    return CodeBlockContent(
        language=language,
        code=code,
        title=f"{language.value.title()} Code Example",
        is_executable=language in [CodeBlockType.PYTHON, CodeBlockType.JAVASCRIPT],
        theme="vs-dark",
        line_numbers=True,
        allow_editing=False
    )


def create_default_chart_content(chart_type: ChartType, data: Dict[str, Any]) -> ChartContent:
    """Create default chart content"""
    return ChartContent(
        chart_type=chart_type,
        data=data,
        title=f"Interactive {chart_type.value.title()} Chart",
        theme="quantum",
        enable_zoom=True,
        enable_pan=True,
        animation_duration=750
    )


def create_whiteboard_content(title: str = "Collaborative Whiteboard") -> WhiteboardContent:
    """Create default whiteboard content"""
    return WhiteboardContent(
        title=title,
        width=1200,
        height=800,
        background_color="#1E293B",
        grid_enabled=True,
        max_participants=10,
        real_time_sync=True,
        is_collaborative=True
    )


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

class ContentValidator:
    """Utility class for validating interactive content"""
    
    @staticmethod
    def validate_code_content(content: Dict[str, Any]) -> bool:
        """Validate code block content"""
        required_fields = ['language', 'code']
        return all(field in content for field in required_fields)
    
    @staticmethod
    def validate_chart_data(data: Dict[str, Any]) -> bool:
        """Validate chart data structure"""
        if 'datasets' not in data:
            return False
        
        for dataset in data['datasets']:
            if 'data' not in dataset:
                return False
        
        return True
    
    @staticmethod
    def validate_whiteboard_operation(operation: Dict[str, Any]) -> bool:
        """Validate whiteboard operation"""
        required_fields = ['operation_type', 'data', 'user_id']
        return all(field in operation for field in required_fields)


# Export all models
__all__ = [
    # Enums
    'MessageType', 'CodeBlockType', 'ChartType', 'DiagramType', 'CalculatorType',
    
    # Content Models
    'InteractiveContent', 'CodeBlockContent', 'ChartContent', 'DiagramContent',
    'CalculatorContent', 'WhiteboardContent', 'VisualizationContent', 
    'QuizContent', 'MathEquationContent',
    
    # Core Models
    'EnhancedMessage',
    
    # Request/Response Models
    'CreateInteractiveContentRequest', 'UpdateInteractiveContentRequest',
    'CollaborationSessionRequest', 'WhiteboardOperation', 'ChartDataUpdate',
    
    # Collaboration Models
    'CollaborationSession', 'ParticipantCursor',
    
    # Analytics Models
    'InteractiveContentAnalytics',
    
    # Utility Functions
    'create_default_code_content', 'create_default_chart_content', 
    'create_whiteboard_content', 'ContentValidator'
]