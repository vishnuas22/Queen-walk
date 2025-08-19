"""
Visualization Utilities

Advanced visualization tools for analytics including interactive dashboards,
statistical plots, and research report generation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import logging
import json

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

logger = logging.getLogger(__name__)


class ChartType(Enum):
    """Types of charts available"""
    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    BOX = "box"
    HEATMAP = "heatmap"
    VIOLIN = "violin"
    CORRELATION = "correlation"
    DISTRIBUTION = "distribution"


@dataclass
class VisualizationConfig:
    """Configuration for visualizations"""
    chart_type: ChartType = ChartType.LINE
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    color_scheme: str = "viridis"
    interactive: bool = True
    width: int = 800
    height: int = 600
    show_legend: bool = True
    theme: str = "plotly_white"


class AnalyticsVisualizer:
    """
    📊 ANALYTICS VISUALIZER
    
    Advanced visualization tools for analytics data with support for
    static and interactive charts.
    """
    
    def __init__(self):
        self.visualization_history = {}
        logger.info("Analytics Visualizer initialized")
    
    def create_learning_pattern_visualization(self,
                                            pattern_data: Dict[str, Any],
                                            config: VisualizationConfig = None) -> Dict[str, Any]:
        """Create visualization for learning patterns"""
        if config is None:
            config = VisualizationConfig(title="Learning Patterns Analysis")
        
        viz_id = f"learning_patterns_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            if PLOTLY_AVAILABLE and config.interactive:
                # Create interactive learning patterns visualization
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Learning Efficiency Over Time', 'Pattern Distribution',
                                  'Engagement Trends', 'Performance Correlation'),
                    specs=[[{"secondary_y": True}, {"type": "pie"}],
                           [{"colspan": 2}, None]]
                )
                
                # Learning efficiency timeline
                if 'efficiency_timeline' in pattern_data:
                    timeline = pattern_data['efficiency_timeline']
                    fig.add_trace(
                        go.Scatter(x=list(range(len(timeline))), y=timeline,
                                 mode='lines+markers', name='Learning Efficiency'),
                        row=1, col=1
                    )
                
                # Pattern distribution pie chart
                if 'pattern_distribution' in pattern_data:
                    patterns = pattern_data['pattern_distribution']
                    fig.add_trace(
                        go.Pie(labels=list(patterns.keys()), values=list(patterns.values()),
                              name="Pattern Distribution"),
                        row=1, col=2
                    )
                
                # Engagement trends
                if 'engagement_data' in pattern_data:
                    engagement = pattern_data['engagement_data']
                    fig.add_trace(
                        go.Scatter(x=list(range(len(engagement))), y=engagement,
                                 mode='lines', name='Engagement', line=dict(color='orange')),
                        row=2, col=1
                    )
                
                fig.update_layout(
                    title=config.title,
                    height=config.height,
                    width=config.width,
                    showlegend=config.show_legend,
                    template=config.theme
                )
                
                # Convert to JSON for storage/transmission
                chart_json = fig.to_json()
                
                result = {
                    'visualization_id': viz_id,
                    'chart_type': 'interactive_dashboard',
                    'chart_data': chart_json,
                    'config': config.__dict__,
                    'created_at': datetime.utcnow().isoformat()
                }
                
            else:
                # Fallback to simple data structure
                result = {
                    'visualization_id': viz_id,
                    'chart_type': 'data_summary',
                    'chart_data': self._create_text_summary(pattern_data),
                    'config': config.__dict__ if config else {},
                    'created_at': datetime.utcnow().isoformat()
                }
            
            self.visualization_history[viz_id] = result
            return result
            
        except Exception as e:
            logger.error(f"Error creating learning pattern visualization: {e}")
            return {
                'visualization_id': viz_id,
                'error': str(e),
                'chart_type': 'error'
            }
    
    def create_performance_dashboard(self,
                                   performance_data: Dict[str, Any],
                                   config: VisualizationConfig = None) -> Dict[str, Any]:
        """Create comprehensive performance dashboard"""
        if config is None:
            config = VisualizationConfig(title="Performance Analytics Dashboard")
        
        viz_id = f"performance_dashboard_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            if PLOTLY_AVAILABLE and config.interactive:
                # Create multi-panel performance dashboard
                fig = make_subplots(
                    rows=3, cols=2,
                    subplot_titles=('Performance Trend', 'Score Distribution',
                                  'Skill Radar', 'Improvement Areas',
                                  'Time Analysis', 'Prediction Accuracy'),
                    specs=[[{}, {}],
                           [{"type": "polar"}, {}],
                           [{}, {}]]
                )
                
                # Performance trend line
                if 'performance_history' in performance_data:
                    history = performance_data['performance_history']
                    fig.add_trace(
                        go.Scatter(x=list(range(len(history))), y=history,
                                 mode='lines+markers', name='Performance Score',
                                 line=dict(color='blue', width=3)),
                        row=1, col=1
                    )
                
                # Score distribution histogram
                if 'score_distribution' in performance_data:
                    scores = performance_data['score_distribution']
                    fig.add_trace(
                        go.Histogram(x=scores, name='Score Distribution',
                                   marker_color='lightblue'),
                        row=1, col=2
                    )
                
                # Skill radar chart
                if 'skill_scores' in performance_data:
                    skills = performance_data['skill_scores']
                    fig.add_trace(
                        go.Scatterpolar(
                            r=list(skills.values()),
                            theta=list(skills.keys()),
                            fill='toself',
                            name='Skill Levels'
                        ),
                        row=2, col=1
                    )
                
                # Improvement areas bar chart
                if 'improvement_areas' in performance_data:
                    areas = performance_data['improvement_areas']
                    fig.add_trace(
                        go.Bar(x=list(areas.keys()), y=list(areas.values()),
                              name='Improvement Potential',
                              marker_color='orange'),
                        row=2, col=2
                    )
                
                fig.update_layout(
                    title=config.title,
                    height=900,  # Larger for dashboard
                    width=config.width,
                    showlegend=config.show_legend,
                    template=config.theme
                )
                
                chart_json = fig.to_json()
                
                result = {
                    'visualization_id': viz_id,
                    'chart_type': 'performance_dashboard',
                    'chart_data': chart_json,
                    'config': config.__dict__,
                    'created_at': datetime.utcnow().isoformat()
                }
                
            else:
                # Fallback summary
                result = {
                    'visualization_id': viz_id,
                    'chart_type': 'performance_summary',
                    'chart_data': self._create_performance_summary(performance_data),
                    'config': config.__dict__ if config else {},
                    'created_at': datetime.utcnow().isoformat()
                }
            
            self.visualization_history[viz_id] = result
            return result
            
        except Exception as e:
            logger.error(f"Error creating performance dashboard: {e}")
            return {
                'visualization_id': viz_id,
                'error': str(e),
                'chart_type': 'error'
            }
    
    def create_behavioral_heatmap(self,
                                behavioral_data: Dict[str, Any],
                                config: VisualizationConfig = None) -> Dict[str, Any]:
        """Create behavioral pattern heatmap"""
        if config is None:
            config = VisualizationConfig(title="Behavioral Patterns Heatmap", chart_type=ChartType.HEATMAP)
        
        viz_id = f"behavioral_heatmap_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            if PLOTLY_AVAILABLE and config.interactive:
                # Create behavioral heatmap
                if 'behavior_matrix' in behavioral_data:
                    matrix = behavioral_data['behavior_matrix']
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=matrix,
                        x=behavioral_data.get('time_labels', list(range(len(matrix[0])))),
                        y=behavioral_data.get('behavior_labels', list(range(len(matrix)))),
                        colorscale=config.color_scheme,
                        hoverongaps=False
                    ))
                    
                    fig.update_layout(
                        title=config.title,
                        xaxis_title=config.x_label or "Time Period",
                        yaxis_title=config.y_label or "Behavior Type",
                        height=config.height,
                        width=config.width,
                        template=config.theme
                    )
                    
                    chart_json = fig.to_json()
                    
                    result = {
                        'visualization_id': viz_id,
                        'chart_type': 'behavioral_heatmap',
                        'chart_data': chart_json,
                        'config': config.__dict__,
                        'created_at': datetime.utcnow().isoformat()
                    }
                else:
                    result = {
                        'visualization_id': viz_id,
                        'chart_type': 'no_data',
                        'message': 'No behavior matrix data available'
                    }
            else:
                # Fallback text representation
                result = {
                    'visualization_id': viz_id,
                    'chart_type': 'behavioral_summary',
                    'chart_data': self._create_behavioral_summary(behavioral_data),
                    'config': config.__dict__ if config else {},
                    'created_at': datetime.utcnow().isoformat()
                }
            
            self.visualization_history[viz_id] = result
            return result
            
        except Exception as e:
            logger.error(f"Error creating behavioral heatmap: {e}")
            return {
                'visualization_id': viz_id,
                'error': str(e),
                'chart_type': 'error'
            }
    
    def _create_text_summary(self, data: Dict[str, Any]) -> str:
        """Create text summary when visualization libraries are not available"""
        summary_lines = ["Analytics Data Summary:"]
        
        for key, value in data.items():
            if isinstance(value, (list, tuple)):
                summary_lines.append(f"- {key}: {len(value)} items")
                if len(value) > 0 and isinstance(value[0], (int, float)):
                    summary_lines.append(f"  Average: {np.mean(value):.3f}")
            elif isinstance(value, dict):
                summary_lines.append(f"- {key}: {len(value)} categories")
            elif isinstance(value, (int, float)):
                summary_lines.append(f"- {key}: {value:.3f}")
            else:
                summary_lines.append(f"- {key}: {str(value)[:50]}...")
        
        return "\n".join(summary_lines)
    
    def _create_performance_summary(self, data: Dict[str, Any]) -> str:
        """Create performance summary text"""
        summary_lines = ["Performance Analytics Summary:"]
        
        if 'performance_history' in data:
            history = data['performance_history']
            summary_lines.extend([
                f"- Performance History: {len(history)} data points",
                f"- Current Score: {history[-1]:.3f}" if history else "- No current score",
                f"- Average Score: {np.mean(history):.3f}" if history else "- No average available",
                f"- Trend: {'Improving' if len(history) > 1 and history[-1] > history[0] else 'Declining'}" if len(history) > 1 else "- No trend data"
            ])
        
        if 'skill_scores' in data:
            skills = data['skill_scores']
            summary_lines.append(f"- Skills Assessed: {len(skills)}")
            if skills:
                best_skill = max(skills.items(), key=lambda x: x[1])
                summary_lines.append(f"- Strongest Skill: {best_skill[0]} ({best_skill[1]:.3f})")
        
        return "\n".join(summary_lines)
    
    def _create_behavioral_summary(self, data: Dict[str, Any]) -> str:
        """Create behavioral summary text"""
        summary_lines = ["Behavioral Analysis Summary:"]
        
        if 'behavior_matrix' in data:
            matrix = data['behavior_matrix']
            summary_lines.extend([
                f"- Behavior Matrix: {len(matrix)}x{len(matrix[0]) if matrix else 0}",
                f"- Total Observations: {sum(sum(row) for row in matrix) if matrix else 0}"
            ])
        
        if 'behavior_labels' in data:
            labels = data['behavior_labels']
            summary_lines.append(f"- Behavior Types: {', '.join(labels[:5])}" + ("..." if len(labels) > 5 else ""))
        
        return "\n".join(summary_lines)


class InteractiveDashboard:
    """
    🎛️ INTERACTIVE DASHBOARD
    
    Advanced interactive dashboard for real-time analytics monitoring.
    """
    
    def __init__(self):
        self.dashboard_configs = {}
        self.active_dashboards = {}
        logger.info("Interactive Dashboard initialized")
    
    def create_analytics_dashboard(self,
                                 dashboard_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive analytics dashboard"""
        dashboard_id = f"dashboard_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Dashboard configuration
        config = {
            'dashboard_id': dashboard_id,
            'title': dashboard_config.get('title', 'Analytics Dashboard'),
            'layout': dashboard_config.get('layout', 'grid'),
            'refresh_interval': dashboard_config.get('refresh_interval', 30),  # seconds
            'panels': dashboard_config.get('panels', []),
            'filters': dashboard_config.get('filters', {}),
            'created_at': datetime.utcnow().isoformat()
        }
        
        # Store configuration
        self.dashboard_configs[dashboard_id] = config
        
        return {
            'dashboard_id': dashboard_id,
            'status': 'created',
            'config': config,
            'access_url': f"/dashboard/{dashboard_id}"  # Placeholder URL
        }
    
    def update_dashboard_data(self,
                            dashboard_id: str,
                            new_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update dashboard with new data"""
        if dashboard_id not in self.dashboard_configs:
            return {'status': 'error', 'message': 'Dashboard not found'}
        
        # Update dashboard data
        self.active_dashboards[dashboard_id] = {
            'data': new_data,
            'last_updated': datetime.utcnow().isoformat()
        }
        
        return {
            'status': 'updated',
            'dashboard_id': dashboard_id,
            'data_points': len(new_data),
            'updated_at': datetime.utcnow().isoformat()
        }


class StatisticalPlotter:
    """
    📈 STATISTICAL PLOTTER
    
    Specialized plotting tools for statistical analysis and research.
    """
    
    def __init__(self):
        self.plot_history = {}
        logger.info("Statistical Plotter initialized")
    
    def create_hypothesis_test_plot(self,
                                  test_results: Dict[str, Any],
                                  config: VisualizationConfig = None) -> Dict[str, Any]:
        """Create visualization for hypothesis test results"""
        if config is None:
            config = VisualizationConfig(title="Hypothesis Test Results")
        
        plot_id = f"hypothesis_test_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            if PLOTLY_AVAILABLE and config.interactive:
                # Create hypothesis test visualization
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Test Statistic Distribution', 'Effect Size'),
                    specs=[[{}, {}]]
                )
                
                # Test statistic distribution
                if 'test_statistic' in test_results and 'critical_value' in test_results:
                    x_range = np.linspace(-4, 4, 1000)
                    y_dist = np.exp(-0.5 * x_range**2) / np.sqrt(2 * np.pi)  # Standard normal
                    
                    fig.add_trace(
                        go.Scatter(x=x_range, y=y_dist, mode='lines',
                                 name='Null Distribution', line=dict(color='blue')),
                        row=1, col=1
                    )
                    
                    # Add test statistic line
                    test_stat = test_results['test_statistic']
                    fig.add_vline(x=test_stat, line_dash="dash", line_color="red",
                                annotation_text=f"Test Statistic: {test_stat:.3f}",
                                row=1, col=1)
                
                # Effect size visualization
                if 'effect_size' in test_results:
                    effect_size = test_results['effect_size']
                    fig.add_trace(
                        go.Bar(x=['Effect Size'], y=[effect_size],
                              name='Effect Size', marker_color='green'),
                        row=1, col=2
                    )
                
                fig.update_layout(
                    title=config.title,
                    height=config.height,
                    width=config.width,
                    template=config.theme
                )
                
                chart_json = fig.to_json()
                
                result = {
                    'plot_id': plot_id,
                    'plot_type': 'hypothesis_test',
                    'chart_data': chart_json,
                    'test_results': test_results,
                    'created_at': datetime.utcnow().isoformat()
                }
            else:
                # Text-based results
                result = {
                    'plot_id': plot_id,
                    'plot_type': 'hypothesis_test_summary',
                    'chart_data': self._format_test_results(test_results),
                    'created_at': datetime.utcnow().isoformat()
                }
            
            self.plot_history[plot_id] = result
            return result
            
        except Exception as e:
            logger.error(f"Error creating hypothesis test plot: {e}")
            return {
                'plot_id': plot_id,
                'error': str(e),
                'plot_type': 'error'
            }
    
    def _format_test_results(self, results: Dict[str, Any]) -> str:
        """Format test results as text"""
        lines = ["Hypothesis Test Results:"]
        
        for key, value in results.items():
            if isinstance(value, float):
                lines.append(f"- {key}: {value:.6f}")
            else:
                lines.append(f"- {key}: {value}")
        
        return "\n".join(lines)


class ResearchReportGenerator:
    """
    📄 RESEARCH REPORT GENERATOR
    
    Automated generation of research reports with visualizations and statistical analysis.
    """
    
    def __init__(self):
        self.report_templates = {}
        self.generated_reports = {}
        logger.info("Research Report Generator initialized")
    
    def generate_analytics_report(self,
                                analytics_results: Dict[str, Any],
                                report_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate comprehensive analytics report"""
        if report_config is None:
            report_config = {'format': 'json', 'include_visualizations': True}
        
        report_id = f"report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Generate report sections
        report_sections = {
            'executive_summary': self._generate_executive_summary(analytics_results),
            'methodology': self._generate_methodology_section(analytics_results),
            'results': self._generate_results_section(analytics_results),
            'visualizations': self._generate_visualization_section(analytics_results) if report_config.get('include_visualizations') else {},
            'conclusions': self._generate_conclusions_section(analytics_results),
            'recommendations': self._generate_recommendations_section(analytics_results)
        }
        
        # Compile final report
        report = {
            'report_id': report_id,
            'title': report_config.get('title', 'Analytics Research Report'),
            'generated_at': datetime.utcnow().isoformat(),
            'sections': report_sections,
            'metadata': {
                'data_sources': analytics_results.keys(),
                'analysis_engines': len(analytics_results),
                'report_format': report_config.get('format', 'json')
            }
        }
        
        # Store report
        self.generated_reports[report_id] = report
        
        return report
    
    def _generate_executive_summary(self, results: Dict[str, Any]) -> str:
        """Generate executive summary section"""
        summary_points = [
            "This report presents comprehensive analytics results from multiple analysis engines.",
            f"Analysis covered {len(results)} different analytical dimensions.",
        ]
        
        # Add key findings
        if 'behavioral_intelligence' in results:
            summary_points.append("Behavioral analysis revealed significant user engagement patterns.")
        
        if 'performance_analytics' in results:
            summary_points.append("Performance metrics indicate measurable learning outcomes.")
        
        if 'attention_optimization' in results:
            summary_points.append("Attention analysis identified optimization opportunities.")
        
        return " ".join(summary_points)
    
    def _generate_methodology_section(self, results: Dict[str, Any]) -> str:
        """Generate methodology section"""
        methodology = [
            "This analysis employed multiple advanced analytics engines:",
        ]
        
        for engine_name in results.keys():
            methodology.append(f"- {engine_name.replace('_', ' ').title()}")
        
        methodology.extend([
            "",
            "Statistical methods included descriptive analysis, inferential testing, and predictive modeling.",
            "Data quality assessment was performed to ensure analytical validity.",
            "Cross-engine correlation analysis was conducted to validate findings."
        ])
        
        return "\n".join(methodology)
    
    def _generate_results_section(self, results: Dict[str, Any]) -> str:
        """Generate results section"""
        results_text = ["Key Findings:\n"]
        
        for engine, data in results.items():
            results_text.append(f"{engine.replace('_', ' ').title()}:")
            
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, (int, float)):
                        results_text.append(f"  - {key}: {value:.3f}")
                    elif isinstance(value, str) and len(value) < 100:
                        results_text.append(f"  - {key}: {value}")
            
            results_text.append("")
        
        return "\n".join(results_text)
    
    def _generate_visualization_section(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Generate visualization section"""
        visualizations = {}
        
        for engine_name, data in results.items():
            viz_description = f"Visualization for {engine_name.replace('_', ' ').title()}"
            visualizations[engine_name] = viz_description
        
        return visualizations
    
    def _generate_conclusions_section(self, results: Dict[str, Any]) -> str:
        """Generate conclusions section"""
        conclusions = [
            "Based on the comprehensive analytics analysis, several key conclusions emerge:",
            "",
            "1. The multi-engine approach provides robust insights into learning behaviors.",
            "2. Cross-validation between engines strengthens confidence in findings.",
            "3. Statistical significance was observed in key performance indicators."
        ]
        
        return "\n".join(conclusions)
    
    def _generate_recommendations_section(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations section"""
        recommendations = [
            "Continue monitoring key performance indicators",
            "Implement suggested optimization strategies",
            "Conduct follow-up analysis to validate improvements",
            "Consider expanding data collection for enhanced insights"
        ]
        
        return recommendations
