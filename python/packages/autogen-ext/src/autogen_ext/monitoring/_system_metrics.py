"""
System metrics collection and monitoring for AutoGen agents.
"""

import asyncio
import psutil
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import deque


@dataclass
class SystemMetrics:
    """System performance metrics snapshot."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    active_connections: int
    load_average: List[float] = field(default_factory=list)


@dataclass
class AgentMetrics:
    """Agent-specific performance metrics."""
    agent_name: str
    timestamp: float
    response_time_ms: float
    token_usage: int
    memory_usage_mb: float
    error_count: int
    success_count: int
    active_tasks: int


class MetricsCollector:
    """Collects and manages system and agent metrics."""
    
    def __init__(self, max_history: int = 1000, collection_interval: float = 30.0):
        self.max_history = max_history
        self.collection_interval = collection_interval
        self.system_metrics_history: deque = deque(maxlen=max_history)
        self.agent_metrics_history: Dict[str, deque] = {}
        self.collection_task: Optional[asyncio.Task] = None
        self.is_collecting = False
        
    async def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_used_mb = memory.used / 1024 / 1024
        memory_available_mb = memory.available / 1024 / 1024
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage_percent = (disk.used / disk.total) * 100
        disk_free_gb = disk.free / 1024 / 1024 / 1024
        
        # Network stats
        network = psutil.net_io_counters()
        
        # Active connections
        connections = len(psutil.net_connections())
        
        # Load average (Unix-like systems)
        try:
            load_avg = list(psutil.getloadavg())
        except AttributeError:
            load_avg = []
        
        metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory_used_mb,
            memory_available_mb=memory_available_mb,
            disk_usage_percent=disk_usage_percent,
            disk_free_gb=disk_free_gb,
            network_bytes_sent=network.bytes_sent,
            network_bytes_recv=network.bytes_recv,
            active_connections=connections,
            load_average=load_avg
        )
        
        self.system_metrics_history.append(metrics)
        return metrics
        
    def record_agent_metrics(self, metrics: AgentMetrics) -> None:
        """Record metrics for a specific agent."""
        if metrics.agent_name not in self.agent_metrics_history:
            self.agent_metrics_history[metrics.agent_name] = deque(maxlen=self.max_history)
            
        self.agent_metrics_history[metrics.agent_name].append(metrics)
        
    async def start_collection(self) -> None:
        """Start automatic metrics collection."""
        if self.is_collecting:
            return
            
        self.is_collecting = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        
    async def stop_collection(self) -> None:
        """Stop automatic metrics collection."""
        self.is_collecting = False
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
            self.collection_task = None
            
    async def _collection_loop(self) -> None:
        """Main metrics collection loop."""
        while self.is_collecting:
            try:
                await self.collect_system_metrics()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception:
                # Continue collecting even if individual collection fails
                await asyncio.sleep(self.collection_interval)
                
    def get_system_stats(self, minutes: int = 60) -> Dict[str, Any]:
        """Get system statistics for the last N minutes."""
        cutoff_time = time.time() - (minutes * 60)
        recent_metrics = [
            m for m in self.system_metrics_history 
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return {}
            
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        max_memory = max(m.memory_percent for m in recent_metrics)
        min_disk_free = min(m.disk_free_gb for m in recent_metrics)
        
        return {
            "period_minutes": minutes,
            "sample_count": len(recent_metrics),
            "avg_cpu_percent": avg_cpu,
            "avg_memory_percent": avg_memory,
            "max_memory_percent": max_memory,
            "min_disk_free_gb": min_disk_free,
            "current_connections": recent_metrics[-1].active_connections if recent_metrics else 0
        }
        
    def get_agent_stats(self, agent_name: str, minutes: int = 60) -> Dict[str, Any]:
        """Get agent statistics for the last N minutes."""
        if agent_name not in self.agent_metrics_history:
            return {}
            
        cutoff_time = time.time() - (minutes * 60)
        recent_metrics = [
            m for m in self.agent_metrics_history[agent_name]
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return {}
            
        total_requests = len(recent_metrics)
        avg_response_time = sum(m.response_time_ms for m in recent_metrics) / total_requests
        total_tokens = sum(m.token_usage for m in recent_metrics)
        total_errors = sum(m.error_count for m in recent_metrics)
        total_successes = sum(m.success_count for m in recent_metrics)
        avg_memory = sum(m.memory_usage_mb for m in recent_metrics) / total_requests
        
        success_rate = total_successes / (total_successes + total_errors) if (total_successes + total_errors) > 0 else 0
        
        return {
            "agent_name": agent_name,
            "period_minutes": minutes,
            "total_requests": total_requests,
            "avg_response_time_ms": avg_response_time,
            "total_token_usage": total_tokens,
            "total_errors": total_errors,
            "success_rate": success_rate,
            "avg_memory_usage_mb": avg_memory
        }
        
    def get_performance_alerts(self) -> List[Dict[str, Any]]:
        """Get performance alerts based on thresholds."""
        alerts = []
        
        if self.system_metrics_history:
            latest = self.system_metrics_history[-1]
            
            # CPU alert
            if latest.cpu_percent > 90:
                alerts.append({
                    "type": "high_cpu",
                    "severity": "critical",
                    "message": f"CPU usage at {latest.cpu_percent:.1f}%",
                    "timestamp": latest.timestamp
                })
                
            # Memory alert
            if latest.memory_percent > 85:
                alerts.append({
                    "type": "high_memory",
                    "severity": "warning" if latest.memory_percent < 95 else "critical",
                    "message": f"Memory usage at {latest.memory_percent:.1f}%",
                    "timestamp": latest.timestamp
                })
                
            # Disk space alert
            if latest.disk_free_gb < 1:
                alerts.append({
                    "type": "low_disk_space",
                    "severity": "critical",
                    "message": f"Only {latest.disk_free_gb:.1f}GB free disk space",
                    "timestamp": latest.timestamp
                })
                
        return alerts
