"""
Metrics collection system for the dashboard.
"""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import statistics


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


@dataclass
class MetricData:
    """Represents a single metric data point."""
    name: str
    value: Union[int, float]
    timestamp: float
    metric_type: MetricType = MetricType.GAUGE
    agent_id: Optional[str] = None
    task_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSummary:
    """Summary statistics for a metric."""
    name: str
    count: int
    min_value: float
    max_value: float
    avg_value: float
    median_value: float
    last_value: float
    last_timestamp: float
    rate_per_minute: float = 0.0


class MetricsCollector:
    """Collects and manages metrics for the dashboard."""
    
    def __init__(self, max_history: int = 10000, retention_hours: int = 24):
        self.max_history = max_history
        self.retention_seconds = retention_hours * 3600
        
        # Storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.metric_types: Dict[str, MetricType] = {}
        
        # Aggregated data
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.timers: Dict[str, List[float]] = defaultdict(list)
        
        # Rate tracking
        self.rate_windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=60))  # 1 minute windows
        
        # Background cleanup task
        self.cleanup_task: Optional[asyncio.Task] = None
        self.start_cleanup_task()
    
    def start_cleanup_task(self):
        """Start background cleanup task."""
        
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(300)  # Clean up every 5 minutes
                    await self.cleanup_old_metrics()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    print(f"Error in metrics cleanup: {e}")
        
        self.cleanup_task = asyncio.create_task(cleanup_loop())
    
    async def add_metric(self, metric: MetricData) -> None:
        """Add a metric data point."""
        
        # Store the metric
        self.metrics[metric.name].append(metric)
        self.metric_types[metric.name] = metric.metric_type
        
        # Update aggregated data based on metric type
        if metric.metric_type == MetricType.COUNTER:
            self.counters[metric.name] += metric.value
        
        elif metric.metric_type == MetricType.GAUGE:
            self.gauges[metric.name] = metric.value
        
        elif metric.metric_type == MetricType.TIMER:
            self.timers[metric.name].append(metric.value)
            # Keep only recent timer values
            if len(self.timers[metric.name]) > 1000:
                self.timers[metric.name] = self.timers[metric.name][-1000:]
        
        elif metric.metric_type == MetricType.RATE:
            # Track rate in time windows
            current_minute = int(time.time() // 60)
            rate_window = self.rate_windows[metric.name]
            
            if not rate_window or rate_window[-1][0] != current_minute:
                rate_window.append((current_minute, metric.value))
            else:
                # Update current minute
                rate_window[-1] = (current_minute, rate_window[-1][1] + metric.value)
    
    def add_counter(self, name: str, value: float = 1, **kwargs) -> None:
        """Add a counter metric."""
        
        metric = MetricData(
            name=name,
            value=value,
            timestamp=time.time(),
            metric_type=MetricType.COUNTER,
            **kwargs
        )
        
        asyncio.create_task(self.add_metric(metric))
    
    def set_gauge(self, name: str, value: float, **kwargs) -> None:
        """Set a gauge metric."""
        
        metric = MetricData(
            name=name,
            value=value,
            timestamp=time.time(),
            metric_type=MetricType.GAUGE,
            **kwargs
        )
        
        asyncio.create_task(self.add_metric(metric))
    
    def record_timer(self, name: str, duration: float, **kwargs) -> None:
        """Record a timer metric."""
        
        metric = MetricData(
            name=name,
            value=duration,
            timestamp=time.time(),
            metric_type=MetricType.TIMER,
            **kwargs
        )
        
        asyncio.create_task(self.add_metric(metric))
    
    def record_rate(self, name: str, count: float = 1, **kwargs) -> None:
        """Record a rate metric."""
        
        metric = MetricData(
            name=name,
            value=count,
            timestamp=time.time(),
            metric_type=MetricType.RATE,
            **kwargs
        )
        
        asyncio.create_task(self.add_metric(metric))
    
    def get_metric_history(self, name: str, limit: Optional[int] = None) -> List[MetricData]:
        """Get metric history for a specific metric."""
        
        if name not in self.metrics:
            return []
        
        history = list(self.metrics[name])
        
        if limit:
            history = history[-limit:]
        
        return history
    
    def get_recent_metrics(self, minutes: int = 5) -> Dict[str, List[MetricData]]:
        """Get recent metrics from the last N minutes."""
        
        cutoff_time = time.time() - (minutes * 60)
        recent_metrics = {}
        
        for name, metric_history in self.metrics.items():
            recent = [m for m in metric_history if m.timestamp >= cutoff_time]
            if recent:
                recent_metrics[name] = recent
        
        return recent_metrics
    
    def get_metric_summary(self, name: str) -> Optional[MetricSummary]:
        """Get summary statistics for a metric."""
        
        if name not in self.metrics or not self.metrics[name]:
            return None
        
        history = list(self.metrics[name])
        values = [m.value for m in history]
        
        if not values:
            return None
        
        # Calculate rate per minute for rate metrics
        rate_per_minute = 0.0
        if self.metric_types.get(name) == MetricType.RATE:
            rate_per_minute = self.calculate_rate_per_minute(name)
        
        return MetricSummary(
            name=name,
            count=len(values),
            min_value=min(values),
            max_value=max(values),
            avg_value=statistics.mean(values),
            median_value=statistics.median(values),
            last_value=values[-1],
            last_timestamp=history[-1].timestamp,
            rate_per_minute=rate_per_minute
        )
    
    def get_metrics_summary(self) -> Dict[str, MetricSummary]:
        """Get summary for all metrics."""
        
        summaries = {}
        
        for name in self.metrics.keys():
            summary = self.get_metric_summary(name)
            if summary:
                summaries[name] = summary
        
        return summaries
    
    def calculate_rate_per_minute(self, name: str) -> float:
        """Calculate rate per minute for a rate metric."""
        
        if name not in self.rate_windows:
            return 0.0
        
        rate_window = self.rate_windows[name]
        
        if len(rate_window) < 2:
            return 0.0
        
        # Calculate average rate over the available window
        total_count = sum(count for _, count in rate_window)
        time_span_minutes = len(rate_window)
        
        return total_count / time_span_minutes if time_span_minutes > 0 else 0.0
    
    def get_message_rate(self) -> float:
        """Get current message rate per minute."""
        return self.calculate_rate_per_minute("messages")
    
    def get_success_rate(self) -> float:
        """Get current success rate percentage."""
        
        success_count = self.counters.get("task_success", 0)
        failure_count = self.counters.get("task_failure", 0)
        total = success_count + failure_count
        
        if total == 0:
            return 0.0
        
        return (success_count / total) * 100
    
    def get_agent_metrics(self, agent_id: str) -> Dict[str, List[MetricData]]:
        """Get metrics for a specific agent."""
        
        agent_metrics = {}
        
        for name, metric_history in self.metrics.items():
            agent_data = [m for m in metric_history if m.agent_id == agent_id]
            if agent_data:
                agent_metrics[name] = agent_data
        
        return agent_metrics
    
    def get_task_metrics(self, task_id: str) -> Dict[str, List[MetricData]]:
        """Get metrics for a specific task."""
        
        task_metrics = {}
        
        for name, metric_history in self.metrics.items():
            task_data = [m for m in metric_history if m.task_id == task_id]
            if task_data:
                task_metrics[name] = task_data
        
        return task_metrics
    
    def get_top_metrics(self, metric_type: MetricType, limit: int = 10) -> List[MetricSummary]:
        """Get top metrics by value for a specific type."""
        
        summaries = []
        
        for name, mtype in self.metric_types.items():
            if mtype == metric_type:
                summary = self.get_metric_summary(name)
                if summary:
                    summaries.append(summary)
        
        # Sort by last value (descending)
        summaries.sort(key=lambda s: s.last_value, reverse=True)
        
        return summaries[:limit]
    
    def get_performance_data(self, minutes: int = 60) -> Dict[str, Any]:
        """Get performance data for dashboard charts."""
        
        cutoff_time = time.time() - (minutes * 60)
        
        # Get time series data
        time_series = {}
        
        for name, metric_history in self.metrics.items():
            recent_data = [m for m in metric_history if m.timestamp >= cutoff_time]
            
            if recent_data:
                # Group by minute
                minute_data = defaultdict(list)
                for metric in recent_data:
                    minute = int(metric.timestamp // 60) * 60
                    minute_data[minute].append(metric.value)
                
                # Calculate averages per minute
                time_series[name] = [
                    {
                        "timestamp": minute,
                        "value": statistics.mean(values)
                    }
                    for minute, values in sorted(minute_data.items())
                ]
        
        return {
            "time_series": time_series,
            "summary": self.get_metrics_summary(),
            "top_counters": self.get_top_metrics(MetricType.COUNTER),
            "top_gauges": self.get_top_metrics(MetricType.GAUGE)
        }
    
    async def cleanup_old_metrics(self) -> None:
        """Clean up old metrics beyond retention period."""
        
        cutoff_time = time.time() - self.retention_seconds
        
        for name, metric_history in self.metrics.items():
            # Remove old metrics
            while metric_history and metric_history[0].timestamp < cutoff_time:
                metric_history.popleft()
        
        # Clean up timer data
        for name, timer_values in self.timers.items():
            if len(timer_values) > 1000:
                self.timers[name] = timer_values[-1000:]
    
    def export_metrics(self, format: str = "json") -> Union[Dict[str, Any], str]:
        """Export metrics in various formats."""
        
        if format == "json":
            return {
                "metrics": {
                    name: [
                        {
                            "name": m.name,
                            "value": m.value,
                            "timestamp": m.timestamp,
                            "type": m.metric_type.value,
                            "agent_id": m.agent_id,
                            "task_id": m.task_id,
                            "metadata": m.metadata,
                            "tags": m.tags
                        }
                        for m in history
                    ]
                    for name, history in self.metrics.items()
                },
                "summaries": {
                    name: {
                        "name": s.name,
                        "count": s.count,
                        "min_value": s.min_value,
                        "max_value": s.max_value,
                        "avg_value": s.avg_value,
                        "median_value": s.median_value,
                        "last_value": s.last_value,
                        "last_timestamp": s.last_timestamp,
                        "rate_per_minute": s.rate_per_minute
                    }
                    for name, s in self.get_metrics_summary().items()
                },
                "counters": dict(self.counters),
                "gauges": dict(self.gauges)
            }
        
        elif format == "prometheus":
            # Export in Prometheus format
            lines = []
            
            for name, summary in self.get_metrics_summary().items():
                metric_type = self.metric_types.get(name, MetricType.GAUGE)
                
                if metric_type == MetricType.COUNTER:
                    lines.append(f"# TYPE {name} counter")
                    lines.append(f"{name} {summary.last_value}")
                
                elif metric_type == MetricType.GAUGE:
                    lines.append(f"# TYPE {name} gauge")
                    lines.append(f"{name} {summary.last_value}")
            
            return "\n".join(lines)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    async def shutdown(self) -> None:
        """Shutdown the metrics collector."""
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
    
    def reset_metrics(self, metric_names: Optional[List[str]] = None) -> None:
        """Reset metrics (useful for testing)."""
        
        if metric_names:
            for name in metric_names:
                if name in self.metrics:
                    self.metrics[name].clear()
                if name in self.counters:
                    self.counters[name] = 0
                if name in self.gauges:
                    del self.gauges[name]
                if name in self.timers:
                    self.timers[name].clear()
        else:
            # Reset all metrics
            self.metrics.clear()
            self.counters.clear()
            self.gauges.clear()
            self.timers.clear()
            self.rate_windows.clear()
            self.metric_types.clear()
