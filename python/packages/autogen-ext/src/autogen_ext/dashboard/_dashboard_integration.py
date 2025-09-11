"""
Integration utilities for connecting the dashboard with AutoGen agents.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass

from autogen_core import AgentId, MessageContext
from autogen_agentchat import ChatCompletionClient

from ._dashboard_server import DashboardServer, DashboardConfig
from ._metrics_collector import MetricsCollector, MetricData, MetricType
from ._progress_tracker import ProgressTracker, TaskProgress, ProgressStatus
from ._visualization_components import ChartComponent, MetricWidget, LogViewer, DashboardLayout


@dataclass
class AgentDashboardConfig:
    """Configuration for agent dashboard integration."""
    enable_metrics: bool = True
    enable_progress_tracking: bool = True
    enable_logging: bool = True
    auto_start_dashboard: bool = True
    dashboard_config: Optional[DashboardConfig] = None
    update_interval: float = 1.0


class DashboardIntegration:
    """Integration layer between AutoGen agents and the dashboard."""
    
    def __init__(self, config: Optional[AgentDashboardConfig] = None):
        self.config = config or AgentDashboardConfig()
        
        # Dashboard components
        self.dashboard_server: Optional[DashboardServer] = None
        self.metrics_collector: Optional[MetricsCollector] = None
        self.progress_tracker: Optional[ProgressTracker] = None
        
        # Agent tracking
        self.registered_agents: Dict[str, Dict[str, Any]] = {}
        self.agent_tasks: Dict[str, List[str]] = {}
        
        # Dashboard layout
        self.dashboard_layout: Optional[DashboardLayout] = None
        
        # Background tasks
        self.update_task: Optional[asyncio.Task] = None
        self.is_running = False
    
    async def initialize(self) -> None:
        """Initialize the dashboard integration."""
        
        # Create dashboard server
        dashboard_config = self.config.dashboard_config or DashboardConfig()
        self.dashboard_server = DashboardServer(dashboard_config)
        
        # Get components from server
        self.metrics_collector = self.dashboard_server.metrics_collector
        self.progress_tracker = self.dashboard_server.progress_tracker
        
        # Create dashboard layout
        self.dashboard_layout = DashboardLayout("main_dashboard", "AutoGen Enhanced Dashboard")
        
        # Setup default dashboard components
        await self._setup_default_dashboard()
        
        # Register callbacks
        self._register_callbacks()
        
        # Start dashboard server if configured
        if self.config.auto_start_dashboard:
            await self.start_dashboard()
    
    async def _setup_default_dashboard(self) -> None:
        """Setup default dashboard components."""
        
        # Agent status widget
        agent_widget = MetricWidget(
            widget_id="agent_count",
            title="Connected Agents",
            format_string="{value} agents"
        )
        self.dashboard_layout.add_widget(agent_widget, {"x": 0, "y": 0, "w": 3, "h": 2})
        
        # Task progress widget
        task_widget = MetricWidget(
            widget_id="active_tasks",
            title="Active Tasks",
            format_string="{value} tasks"
        )
        self.dashboard_layout.add_widget(task_widget, {"x": 3, "y": 0, "w": 3, "h": 2})
        
        # Success rate widget
        success_widget = MetricWidget(
            widget_id="success_rate",
            title="Success Rate",
            format_string="{value}%"
        )
        self.dashboard_layout.add_widget(success_widget, {"x": 6, "y": 0, "w": 3, "h": 2})
        
        # Message rate widget
        message_widget = MetricWidget(
            widget_id="message_rate",
            title="Messages/min",
            format_string="{value}/min"
        )
        self.dashboard_layout.add_widget(message_widget, {"x": 9, "y": 0, "w": 3, "h": 2})
        
        # Performance chart
        performance_chart = ChartComponent(
            chart_id="performance_chart",
            title="Performance Metrics",
            width=800,
            height=400
        )
        self.dashboard_layout.add_chart(performance_chart, {"x": 0, "y": 2, "w": 12, "h": 6})
        
        # Activity log
        activity_log = LogViewer(
            viewer_id="activity_log",
            title="Activity Log",
            max_entries=1000
        )
        self.dashboard_layout.add_log_viewer(activity_log, {"x": 0, "y": 8, "w": 12, "h": 6})
    
    def _register_callbacks(self) -> None:
        """Register callbacks for dashboard events."""
        
        if self.dashboard_server:
            # Agent connection callbacks
            self.dashboard_server.register_agent_callbacks(
                on_connected=self._on_agent_connected,
                on_disconnected=self._on_agent_disconnected
            )
            
            # Task callbacks
            self.dashboard_server.register_task_callbacks(
                on_started=self._on_task_started,
                on_completed=self._on_task_completed
            )
        
        if self.progress_tracker:
            # Progress tracking callbacks
            self.progress_tracker.register_callback("task_created", self._on_task_created)
            self.progress_tracker.register_callback("task_updated", self._on_task_updated)
            self.progress_tracker.register_callback("task_completed", self._on_task_completed_callback)
    
    async def register_agent(
        self,
        agent_id: str,
        agent_name: str,
        agent_type: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register an agent with the dashboard."""
        
        agent_info = {
            "id": agent_id,
            "name": agent_name,
            "type": agent_type,
            "registered_at": time.time(),
            "metadata": metadata or {},
            "status": "registered"
        }
        
        self.registered_agents[agent_id] = agent_info
        
        # Log agent registration
        if self.dashboard_layout and "activity_log" in self.dashboard_layout.log_viewers:
            log_viewer = self.dashboard_layout.log_viewers["activity_log"]
            log_viewer.add_entry(
                message=f"Agent {agent_name} ({agent_id}) registered",
                level="INFO",
                source="dashboard_integration"
            )
        
        # Update agent count metric
        if self.metrics_collector:
            self.metrics_collector.set_gauge(
                "registered_agents",
                len(self.registered_agents),
                agent_id=agent_id
            )
    
    async def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the dashboard."""
        
        if agent_id in self.registered_agents:
            agent_info = self.registered_agents[agent_id]
            del self.registered_agents[agent_id]
            
            # Log agent unregistration
            if self.dashboard_layout and "activity_log" in self.dashboard_layout.log_viewers:
                log_viewer = self.dashboard_layout.log_viewers["activity_log"]
                log_viewer.add_entry(
                    message=f"Agent {agent_info['name']} ({agent_id}) unregistered",
                    level="INFO",
                    source="dashboard_integration"
                )
            
            # Update agent count metric
            if self.metrics_collector:
                self.metrics_collector.set_gauge(
                    "registered_agents",
                    len(self.registered_agents),
                    agent_id=agent_id
                )
    
    async def track_agent_message(
        self,
        agent_id: str,
        message_type: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Track an agent message."""
        
        if not self.config.enable_logging:
            return
        
        # Add to metrics
        if self.metrics_collector:
            self.metrics_collector.record_rate("messages", agent_id=agent_id)
            self.metrics_collector.add_counter(f"messages_{message_type}", agent_id=agent_id)
        
        # Add to activity log
        if self.dashboard_layout and "activity_log" in self.dashboard_layout.log_viewers:
            log_viewer = self.dashboard_layout.log_viewers["activity_log"]
            
            agent_name = self.registered_agents.get(agent_id, {}).get("name", agent_id)
            
            log_viewer.add_entry(
                message=f"[{agent_name}] {message_type}: {content[:100]}{'...' if len(content) > 100 else ''}",
                level="INFO",
                source=agent_id,
                metadata=metadata
            )
    
    async def track_task_progress(
        self,
        task_id: str,
        task_name: str,
        progress: float,
        agent_id: Optional[str] = None,
        status: Optional[ProgressStatus] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Track task progress."""
        
        if not self.config.enable_progress_tracking:
            return
        
        if not self.progress_tracker:
            return
        
        # Create or update task progress
        task_progress = TaskProgress(
            task_id=task_id,
            name=task_name,
            progress=progress,
            status=status or ProgressStatus.RUNNING,
            agent_id=agent_id,
            metadata=metadata or {}
        )
        
        await self.progress_tracker.update_task_progress(task_progress)
        
        # Track agent tasks
        if agent_id:
            if agent_id not in self.agent_tasks:
                self.agent_tasks[agent_id] = []
            
            if task_id not in self.agent_tasks[agent_id]:
                self.agent_tasks[agent_id].append(task_id)
    
    async def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        agent_id: Optional[str] = None,
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a custom metric."""
        
        if not self.config.enable_metrics:
            return
        
        if not self.metrics_collector:
            return
        
        metric_data = MetricData(
            name=name,
            value=value,
            timestamp=time.time(),
            metric_type=metric_type,
            agent_id=agent_id,
            task_id=task_id,
            metadata=metadata or {}
        )
        
        await self.metrics_collector.add_metric(metric_data)
    
    async def start_dashboard(self) -> None:
        """Start the dashboard server."""
        
        if not self.dashboard_server:
            await self.initialize()
        
        self.is_running = True
        
        # Start background update task
        self.update_task = asyncio.create_task(self._update_loop())
        
        # Start dashboard server (this will block)
        await self.dashboard_server.start()
    
    async def stop_dashboard(self) -> None:
        """Stop the dashboard server."""
        
        self.is_running = False
        
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        
        if self.dashboard_server:
            await self.dashboard_server.stop()
    
    async def _update_loop(self) -> None:
        """Background update loop for dashboard components."""
        
        while self.is_running:
            try:
                await self._update_dashboard_components()
                await asyncio.sleep(self.config.update_interval)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in dashboard update loop: {e}")
                await asyncio.sleep(1)
    
    async def _update_dashboard_components(self) -> None:
        """Update dashboard components with current data."""
        
        if not self.dashboard_layout:
            return
        
        # Update agent count widget
        if "agent_count" in self.dashboard_layout.widgets:
            widget = self.dashboard_layout.widgets["agent_count"]
            widget.update_value(len(self.registered_agents))
        
        # Update active tasks widget
        if "active_tasks" in self.dashboard_layout.widgets and self.progress_tracker:
            widget = self.dashboard_layout.widgets["active_tasks"]
            active_tasks = self.progress_tracker.get_active_tasks()
            widget.update_value(len(active_tasks))
        
        # Update success rate widget
        if "success_rate" in self.dashboard_layout.widgets and self.metrics_collector:
            widget = self.dashboard_layout.widgets["success_rate"]
            success_rate = self.metrics_collector.get_success_rate()
            widget.update_value(success_rate)
        
        # Update message rate widget
        if "message_rate" in self.dashboard_layout.widgets and self.metrics_collector:
            widget = self.dashboard_layout.widgets["message_rate"]
            message_rate = self.metrics_collector.get_message_rate()
            widget.update_value(message_rate)
        
        # Update performance chart
        if "performance_chart" in self.dashboard_layout.charts and self.metrics_collector:
            chart = self.dashboard_layout.charts["performance_chart"]
            
            # Get performance data
            performance_data = self.metrics_collector.get_performance_data(minutes=30)
            
            # Update chart series (simplified for now)
            # In a real implementation, you'd convert the time series data to chart format
    
    async def _on_agent_connected(self, agent_id: str) -> None:
        """Handle agent connection event."""
        
        if agent_id in self.registered_agents:
            self.registered_agents[agent_id]["status"] = "connected"
            self.registered_agents[agent_id]["connected_at"] = time.time()
    
    async def _on_agent_disconnected(self, agent_id: str) -> None:
        """Handle agent disconnection event."""
        
        if agent_id in self.registered_agents:
            self.registered_agents[agent_id]["status"] = "disconnected"
            self.registered_agents[agent_id]["disconnected_at"] = time.time()
    
    async def _on_task_started(self, task_id: str) -> None:
        """Handle task started event."""
        
        if self.dashboard_layout and "activity_log" in self.dashboard_layout.log_viewers:
            log_viewer = self.dashboard_layout.log_viewers["activity_log"]
            log_viewer.add_entry(
                message=f"Task {task_id} started",
                level="INFO",
                source="task_manager"
            )
    
    async def _on_task_completed(self, task_id: str) -> None:
        """Handle task completed event."""
        
        if self.dashboard_layout and "activity_log" in self.dashboard_layout.log_viewers:
            log_viewer = self.dashboard_layout.log_viewers["activity_log"]
            log_viewer.add_entry(
                message=f"Task {task_id} completed",
                level="INFO",
                source="task_manager"
            )
    
    async def _on_task_created(self, task: TaskProgress) -> None:
        """Handle task created event."""
        
        if self.dashboard_layout and "activity_log" in self.dashboard_layout.log_viewers:
            log_viewer = self.dashboard_layout.log_viewers["activity_log"]
            log_viewer.add_entry(
                message=f"Task '{task.name}' created (ID: {task.task_id})",
                level="INFO",
                source="progress_tracker"
            )
    
    async def _on_task_updated(self, task: TaskProgress) -> None:
        """Handle task updated event."""
        
        # Record progress metric
        if self.metrics_collector:
            await self.record_metric(
                f"task_progress_{task.task_id}",
                task.progress,
                MetricType.GAUGE,
                agent_id=task.agent_id,
                task_id=task.task_id
            )
    
    async def _on_task_completed_callback(self, task: TaskProgress) -> None:
        """Handle task completed callback."""
        
        # Record completion metric
        if self.metrics_collector:
            if task.status == ProgressStatus.COMPLETED:
                self.metrics_collector.add_counter("task_success", agent_id=task.agent_id)
            else:
                self.metrics_collector.add_counter("task_failure", agent_id=task.agent_id)
    
    def get_dashboard_url(self) -> Optional[str]:
        """Get the dashboard URL."""
        
        if self.dashboard_server:
            return self.dashboard_server.get_dashboard_url()
        
        return None
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data."""
        
        data = {
            "registered_agents": self.registered_agents,
            "agent_tasks": self.agent_tasks,
            "is_running": self.is_running
        }
        
        if self.dashboard_layout:
            data["layout"] = self.dashboard_layout.to_dict()
        
        if self.metrics_collector:
            data["metrics"] = self.metrics_collector.get_metrics_summary()
        
        if self.progress_tracker:
            data["tasks"] = {
                "active": self.progress_tracker.get_active_tasks(),
                "completed": self.progress_tracker.get_completed_tasks(limit=10),
                "metrics": self.progress_tracker.get_task_metrics()
            }
        
        return data


# Decorator for easy agent integration
def dashboard_enabled(config: Optional[AgentDashboardConfig] = None):
    """Decorator to enable dashboard integration for an agent class."""
    
    def decorator(agent_class):
        original_init = agent_class.__init__
        
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            
            # Add dashboard integration
            self._dashboard_integration = DashboardIntegration(config)
            
            # Register the agent
            agent_id = getattr(self, 'id', str(id(self)))
            agent_name = getattr(self, 'name', agent_class.__name__)
            
            asyncio.create_task(
                self._dashboard_integration.register_agent(
                    agent_id=agent_id,
                    agent_name=agent_name,
                    agent_type=agent_class.__name__
                )
            )
        
        agent_class.__init__ = new_init
        
        # Add dashboard methods to the agent class
        def track_message(self, message_type: str, content: str, metadata: Optional[Dict[str, Any]] = None):
            agent_id = getattr(self, 'id', str(id(self)))
            asyncio.create_task(
                self._dashboard_integration.track_agent_message(
                    agent_id, message_type, content, metadata
                )
            )
        
        def track_task(self, task_id: str, task_name: str, progress: float, 
                      status: Optional[ProgressStatus] = None, metadata: Optional[Dict[str, Any]] = None):
            agent_id = getattr(self, 'id', str(id(self)))
            asyncio.create_task(
                self._dashboard_integration.track_task_progress(
                    task_id, task_name, progress, agent_id, status, metadata
                )
            )
        
        def record_metric(self, name: str, value: float, metric_type: MetricType = MetricType.GAUGE,
                         metadata: Optional[Dict[str, Any]] = None):
            agent_id = getattr(self, 'id', str(id(self)))
            asyncio.create_task(
                self._dashboard_integration.record_metric(
                    name, value, metric_type, agent_id, None, metadata
                )
            )
        
        agent_class.track_message = track_message
        agent_class.track_task = track_task
        agent_class.record_metric = record_metric
        
        return agent_class
    
    return decorator
