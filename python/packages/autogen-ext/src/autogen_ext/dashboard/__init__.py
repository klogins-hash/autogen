"""
Real-time progress visualization dashboard for AutoGen enhanced system.

This module provides:
- Web-based dashboard for monitoring agent activities
- Real-time progress tracking and visualization
- Performance metrics and analytics
- Interactive controls for agent management
- WebSocket-based live updates
"""

from ._dashboard_server import DashboardServer, DashboardConfig
from ._metrics_collector import MetricsCollector, MetricType, MetricData
from ._progress_tracker import ProgressTracker, TaskProgress, ProgressStatus
from ._visualization_components import ChartComponent, MetricWidget, LogViewer
from ._websocket_manager import WebSocketManager, ClientConnection

__all__ = [
    "DashboardServer",
    "DashboardConfig", 
    "MetricsCollector",
    "MetricType",
    "MetricData",
    "ProgressTracker",
    "TaskProgress",
    "ProgressStatus",
    "ChartComponent",
    "MetricWidget",
    "LogViewer",
    "WebSocketManager",
    "ClientConnection"
]
