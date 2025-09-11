"""
Dashboard server for real-time monitoring of AutoGen enhanced system.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Callable
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from ._metrics_collector import MetricsCollector, MetricData
from ._progress_tracker import ProgressTracker, TaskProgress
from ._websocket_manager import WebSocketManager


@dataclass
class DashboardConfig:
    """Configuration for the dashboard server."""
    host: str = "localhost"
    port: int = 8080
    debug: bool = False
    auto_reload: bool = False
    static_dir: str = "static"
    templates_dir: str = "templates"
    update_interval: float = 1.0  # seconds
    max_history_entries: int = 1000
    enable_authentication: bool = False
    api_key: Optional[str] = None


class DashboardServer:
    """Real-time dashboard server for AutoGen enhanced system."""
    
    def __init__(self, config: Optional[DashboardConfig] = None):
        self.config = config or DashboardConfig()
        
        # FastAPI app
        self.app = FastAPI(
            title="AutoGen Enhanced Dashboard",
            description="Real-time monitoring and control dashboard",
            version="1.0.0",
            debug=self.config.debug
        )
        
        # Components
        self.metrics_collector = MetricsCollector()
        self.progress_tracker = ProgressTracker()
        self.websocket_manager = WebSocketManager()
        
        # State
        self.is_running = False
        self.update_task: Optional[asyncio.Task] = None
        self.connected_agents: Dict[str, Dict[str, Any]] = {}
        
        # Setup routes and static files
        self._setup_routes()
        self._setup_static_files()
        
        # Event callbacks
        self.on_agent_connected: Optional[Callable] = None
        self.on_agent_disconnected: Optional[Callable] = None
        self.on_task_started: Optional[Callable] = None
        self.on_task_completed: Optional[Callable] = None
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home(request: Request):
            """Main dashboard page."""
            return self._render_template("dashboard.html", {
                "title": "AutoGen Enhanced Dashboard",
                "config": self.config
            })
        
        @self.app.get("/api/status")
        async def get_status():
            """Get system status."""
            return {
                "status": "running" if self.is_running else "stopped",
                "connected_agents": len(self.connected_agents),
                "active_tasks": len(self.progress_tracker.get_active_tasks()),
                "uptime": time.time() - getattr(self, 'start_time', time.time()),
                "metrics_count": len(self.metrics_collector.get_recent_metrics())
            }
        
        @self.app.get("/api/agents")
        async def get_agents():
            """Get connected agents information."""
            return {
                "agents": self.connected_agents,
                "count": len(self.connected_agents)
            }
        
        @self.app.get("/api/metrics")
        async def get_metrics():
            """Get system metrics."""
            return {
                "metrics": self.metrics_collector.get_recent_metrics(),
                "summary": self.metrics_collector.get_metrics_summary()
            }
        
        @self.app.get("/api/tasks")
        async def get_tasks():
            """Get task progress information."""
            return {
                "active_tasks": self.progress_tracker.get_active_tasks(),
                "completed_tasks": self.progress_tracker.get_completed_tasks(),
                "task_history": self.progress_tracker.get_task_history()
            }
        
        @self.app.post("/api/agents/{agent_id}/command")
        async def send_agent_command(agent_id: str, command: Dict[str, Any]):
            """Send command to a specific agent."""
            if agent_id in self.connected_agents:
                await self.websocket_manager.send_to_agent(agent_id, command)
                return {"status": "sent", "agent_id": agent_id, "command": command}
            else:
                return {"status": "error", "message": f"Agent {agent_id} not connected"}
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates."""
            await self.websocket_manager.connect(websocket)
            
            try:
                while True:
                    # Wait for messages from client
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    # Handle different message types
                    await self._handle_websocket_message(websocket, message)
                    
            except WebSocketDisconnect:
                await self.websocket_manager.disconnect(websocket)
        
        @self.app.websocket("/ws/agent/{agent_id}")
        async def agent_websocket(websocket: WebSocket, agent_id: str):
            """WebSocket endpoint for agent connections."""
            await self.websocket_manager.connect_agent(websocket, agent_id)
            
            # Register agent
            self.connected_agents[agent_id] = {
                "id": agent_id,
                "connected_at": time.time(),
                "status": "connected",
                "last_activity": time.time()
            }
            
            if self.on_agent_connected:
                await self.on_agent_connected(agent_id)
            
            try:
                while True:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    # Update agent activity
                    self.connected_agents[agent_id]["last_activity"] = time.time()
                    
                    # Handle agent messages
                    await self._handle_agent_message(agent_id, message)
                    
            except WebSocketDisconnect:
                await self.websocket_manager.disconnect_agent(agent_id)
                
                # Unregister agent
                if agent_id in self.connected_agents:
                    del self.connected_agents[agent_id]
                
                if self.on_agent_disconnected:
                    await self.on_agent_disconnected(agent_id)
    
    def _setup_static_files(self):
        """Setup static file serving."""
        
        # Create static directory if it doesn't exist
        static_path = Path(__file__).parent / self.config.static_dir
        static_path.mkdir(exist_ok=True)
        
        # Mount static files
        self.app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
        
        # Create templates directory
        templates_path = Path(__file__).parent / self.config.templates_dir
        templates_path.mkdir(exist_ok=True)
        
        # Setup Jinja2 templates
        self.templates = Jinja2Templates(directory=str(templates_path))
        
        # Create default template if it doesn't exist
        self._create_default_template()
    
    def _create_default_template(self):
        """Create default dashboard template."""
        
        template_path = Path(__file__).parent / self.config.templates_dir / "dashboard.html"
        
        if not template_path.exists():
            template_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .metric-card { transition: all 0.3s ease; }
        .metric-card:hover { transform: translateY(-2px); }
        .status-indicator { animation: pulse 2s infinite; }
    </style>
</head>
<body class="bg-gray-100 min-h-screen">
    <div id="app" class="container mx-auto px-4 py-8">
        <!-- Header -->
        <header class="mb-8">
            <h1 class="text-4xl font-bold text-gray-800 mb-2">{{ title }}</h1>
            <div class="flex items-center space-x-4">
                <div class="flex items-center">
                    <div id="status-indicator" class="w-3 h-3 bg-green-500 rounded-full status-indicator mr-2"></div>
                    <span id="status-text" class="text-gray-600">Connected</span>
                </div>
                <div class="text-gray-500">
                    <span id="uptime">Uptime: --</span>
                </div>
            </div>
        </header>

        <!-- Metrics Overview -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <div class="metric-card bg-white rounded-lg shadow-md p-6">
                <h3 class="text-lg font-semibold text-gray-700 mb-2">Connected Agents</h3>
                <div class="text-3xl font-bold text-blue-600" id="agent-count">0</div>
            </div>
            <div class="metric-card bg-white rounded-lg shadow-md p-6">
                <h3 class="text-lg font-semibold text-gray-700 mb-2">Active Tasks</h3>
                <div class="text-3xl font-bold text-green-600" id="task-count">0</div>
            </div>
            <div class="metric-card bg-white rounded-lg shadow-md p-6">
                <h3 class="text-lg font-semibold text-gray-700 mb-2">Messages/min</h3>
                <div class="text-3xl font-bold text-purple-600" id="message-rate">0</div>
            </div>
            <div class="metric-card bg-white rounded-lg shadow-md p-6">
                <h3 class="text-lg font-semibold text-gray-700 mb-2">Success Rate</h3>
                <div class="text-3xl font-bold text-orange-600" id="success-rate">0%</div>
            </div>
        </div>

        <!-- Main Content -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <!-- Agent Status -->
            <div class="bg-white rounded-lg shadow-md p-6">
                <h2 class="text-xl font-semibold text-gray-800 mb-4">Agent Status</h2>
                <div id="agent-list" class="space-y-3">
                    <!-- Agent items will be populated here -->
                </div>
            </div>

            <!-- Task Progress -->
            <div class="bg-white rounded-lg shadow-md p-6">
                <h2 class="text-xl font-semibold text-gray-800 mb-4">Task Progress</h2>
                <div id="task-list" class="space-y-3">
                    <!-- Task items will be populated here -->
                </div>
            </div>

            <!-- Performance Chart -->
            <div class="bg-white rounded-lg shadow-md p-6 lg:col-span-2">
                <h2 class="text-xl font-semibold text-gray-800 mb-4">Performance Metrics</h2>
                <canvas id="performance-chart" width="400" height="200"></canvas>
            </div>

            <!-- Activity Log -->
            <div class="bg-white rounded-lg shadow-md p-6 lg:col-span-2">
                <h2 class="text-xl font-semibold text-gray-800 mb-4">Activity Log</h2>
                <div id="activity-log" class="h-64 overflow-y-auto bg-gray-50 rounded p-4 font-mono text-sm">
                    <!-- Log entries will be populated here -->
                </div>
            </div>
        </div>
    </div>

    <script>
        // Dashboard JavaScript
        class Dashboard {
            constructor() {
                this.ws = null;
                this.chart = null;
                this.connectWebSocket();
                this.initChart();
                this.startUpdates();
            }

            connectWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws`;
                
                this.ws = new WebSocket(wsUrl);
                
                this.ws.onopen = () => {
                    console.log('WebSocket connected');
                    this.updateStatus('Connected', 'green');
                };
                
                this.ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    this.handleMessage(data);
                };
                
                this.ws.onclose = () => {
                    console.log('WebSocket disconnected');
                    this.updateStatus('Disconnected', 'red');
                    // Reconnect after 5 seconds
                    setTimeout(() => this.connectWebSocket(), 5000);
                };
            }

            handleMessage(data) {
                switch(data.type) {
                    case 'status_update':
                        this.updateMetrics(data.data);
                        break;
                    case 'agent_update':
                        this.updateAgents(data.data);
                        break;
                    case 'task_update':
                        this.updateTasks(data.data);
                        break;
                    case 'log_entry':
                        this.addLogEntry(data.data);
                        break;
                }
            }

            updateStatus(status, color) {
                const indicator = document.getElementById('status-indicator');
                const text = document.getElementById('status-text');
                
                indicator.className = `w-3 h-3 bg-${color}-500 rounded-full status-indicator mr-2`;
                text.textContent = status;
            }

            updateMetrics(data) {
                document.getElementById('agent-count').textContent = data.connected_agents || 0;
                document.getElementById('task-count').textContent = data.active_tasks || 0;
                document.getElementById('message-rate').textContent = data.message_rate || 0;
                document.getElementById('success-rate').textContent = (data.success_rate || 0) + '%';
                
                if (data.uptime) {
                    const hours = Math.floor(data.uptime / 3600);
                    const minutes = Math.floor((data.uptime % 3600) / 60);
                    document.getElementById('uptime').textContent = `Uptime: ${hours}h ${minutes}m`;
                }
            }

            updateAgents(agents) {
                const agentList = document.getElementById('agent-list');
                agentList.innerHTML = '';
                
                Object.values(agents).forEach(agent => {
                    const agentDiv = document.createElement('div');
                    agentDiv.className = 'flex items-center justify-between p-3 bg-gray-50 rounded';
                    agentDiv.innerHTML = `
                        <div>
                            <div class="font-medium">${agent.id}</div>
                            <div class="text-sm text-gray-500">Status: ${agent.status}</div>
                        </div>
                        <div class="w-3 h-3 bg-green-500 rounded-full"></div>
                    `;
                    agentList.appendChild(agentDiv);
                });
            }

            updateTasks(tasks) {
                const taskList = document.getElementById('task-list');
                taskList.innerHTML = '';
                
                tasks.forEach(task => {
                    const taskDiv = document.createElement('div');
                    taskDiv.className = 'p-3 bg-gray-50 rounded';
                    taskDiv.innerHTML = `
                        <div class="font-medium">${task.name}</div>
                        <div class="w-full bg-gray-200 rounded-full h-2 mt-2">
                            <div class="bg-blue-600 h-2 rounded-full" style="width: ${task.progress}%"></div>
                        </div>
                        <div class="text-sm text-gray-500 mt-1">${task.progress}% complete</div>
                    `;
                    taskList.appendChild(taskDiv);
                });
            }

            addLogEntry(entry) {
                const log = document.getElementById('activity-log');
                const logEntry = document.createElement('div');
                logEntry.className = 'mb-1';
                logEntry.innerHTML = `<span class="text-gray-500">[${new Date(entry.timestamp).toLocaleTimeString()}]</span> ${entry.message}`;
                log.appendChild(logEntry);
                log.scrollTop = log.scrollHeight;
            }

            initChart() {
                const ctx = document.getElementById('performance-chart').getContext('2d');
                this.chart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Messages/min',
                            data: [],
                            borderColor: 'rgb(59, 130, 246)',
                            backgroundColor: 'rgba(59, 130, 246, 0.1)',
                            tension: 0.1
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            y: {
                                beginAtZero: true
                            }
                        }
                    }
                });
            }

            async startUpdates() {
                setInterval(async () => {
                    try {
                        const response = await fetch('/api/status');
                        const data = await response.json();
                        this.updateMetrics(data);
                    } catch (error) {
                        console.error('Failed to fetch status:', error);
                    }
                }, 5000);
            }
        }

        // Initialize dashboard when page loads
        document.addEventListener('DOMContentLoaded', () => {
            new Dashboard();
        });
    </script>
</body>
</html>'''
            
            with open(template_path, 'w') as f:
                f.write(template_content)
    
    def _render_template(self, template_name: str, context: Dict[str, Any]) -> str:
        """Render a template with context."""
        return self.templates.TemplateResponse(template_name, {"request": {}, **context})
    
    async def _handle_websocket_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """Handle WebSocket message from client."""
        
        message_type = message.get("type")
        
        if message_type == "subscribe":
            # Client wants to subscribe to updates
            await self.websocket_manager.add_subscription(websocket, message.get("topics", []))
        
        elif message_type == "unsubscribe":
            # Client wants to unsubscribe from updates
            await self.websocket_manager.remove_subscription(websocket, message.get("topics", []))
        
        elif message_type == "command":
            # Client wants to send a command
            command = message.get("command")
            if command:
                await self._execute_command(command)
    
    async def _handle_agent_message(self, agent_id: str, message: Dict[str, Any]):
        """Handle message from an agent."""
        
        message_type = message.get("type")
        
        if message_type == "metric":
            # Agent is reporting a metric
            metric_data = MetricData(
                name=message.get("name"),
                value=message.get("value"),
                timestamp=time.time(),
                agent_id=agent_id,
                metadata=message.get("metadata", {})
            )
            await self.metrics_collector.add_metric(metric_data)
        
        elif message_type == "task_progress":
            # Agent is reporting task progress
            task_progress = TaskProgress(
                task_id=message.get("task_id"),
                name=message.get("name"),
                progress=message.get("progress", 0),
                status=message.get("status"),
                agent_id=agent_id,
                metadata=message.get("metadata", {})
            )
            await self.progress_tracker.update_task_progress(task_progress)
        
        elif message_type == "log":
            # Agent is sending a log entry
            log_entry = {
                "timestamp": time.time(),
                "agent_id": agent_id,
                "level": message.get("level", "INFO"),
                "message": message.get("message"),
                "metadata": message.get("metadata", {})
            }
            await self._broadcast_log_entry(log_entry)
    
    async def _execute_command(self, command: Dict[str, Any]):
        """Execute a command from the dashboard."""
        
        command_type = command.get("type")
        
        if command_type == "pause_agent":
            agent_id = command.get("agent_id")
            if agent_id in self.connected_agents:
                await self.websocket_manager.send_to_agent(agent_id, {"type": "pause"})
        
        elif command_type == "resume_agent":
            agent_id = command.get("agent_id")
            if agent_id in self.connected_agents:
                await self.websocket_manager.send_to_agent(agent_id, {"type": "resume"})
        
        elif command_type == "restart_agent":
            agent_id = command.get("agent_id")
            if agent_id in self.connected_agents:
                await self.websocket_manager.send_to_agent(agent_id, {"type": "restart"})
    
    async def _broadcast_log_entry(self, log_entry: Dict[str, Any]):
        """Broadcast log entry to all connected clients."""
        
        message = {
            "type": "log_entry",
            "data": log_entry
        }
        
        await self.websocket_manager.broadcast(message)
    
    async def start_background_updates(self):
        """Start background update task."""
        
        async def update_loop():
            while self.is_running:
                try:
                    # Collect current status
                    status_data = {
                        "connected_agents": len(self.connected_agents),
                        "active_tasks": len(self.progress_tracker.get_active_tasks()),
                        "uptime": time.time() - getattr(self, 'start_time', time.time()),
                        "message_rate": self.metrics_collector.get_message_rate(),
                        "success_rate": self.metrics_collector.get_success_rate()
                    }
                    
                    # Broadcast status update
                    await self.websocket_manager.broadcast({
                        "type": "status_update",
                        "data": status_data
                    })
                    
                    # Broadcast agent updates
                    await self.websocket_manager.broadcast({
                        "type": "agent_update",
                        "data": self.connected_agents
                    })
                    
                    # Broadcast task updates
                    active_tasks = self.progress_tracker.get_active_tasks()
                    await self.websocket_manager.broadcast({
                        "type": "task_update",
                        "data": active_tasks
                    })
                    
                    await asyncio.sleep(self.config.update_interval)
                    
                except Exception as e:
                    print(f"Error in update loop: {e}")
                    await asyncio.sleep(1)
        
        self.update_task = asyncio.create_task(update_loop())
    
    async def start(self):
        """Start the dashboard server."""
        
        self.is_running = True
        self.start_time = time.time()
        
        # Start background updates
        await self.start_background_updates()
        
        # Start the server
        config = uvicorn.Config(
            self.app,
            host=self.config.host,
            port=self.config.port,
            reload=self.config.auto_reload,
            log_level="info" if self.config.debug else "warning"
        )
        
        server = uvicorn.Server(config)
        await server.serve()
    
    async def stop(self):
        """Stop the dashboard server."""
        
        self.is_running = False
        
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        
        # Disconnect all WebSocket connections
        await self.websocket_manager.disconnect_all()
    
    def register_agent_callbacks(
        self,
        on_connected: Optional[Callable] = None,
        on_disconnected: Optional[Callable] = None
    ):
        """Register callbacks for agent events."""
        
        self.on_agent_connected = on_connected
        self.on_agent_disconnected = on_disconnected
    
    def register_task_callbacks(
        self,
        on_started: Optional[Callable] = None,
        on_completed: Optional[Callable] = None
    ):
        """Register callbacks for task events."""
        
        self.on_task_started = on_started
        self.on_task_completed = on_completed
    
    def get_dashboard_url(self) -> str:
        """Get the dashboard URL."""
        return f"http://{self.config.host}:{self.config.port}"
