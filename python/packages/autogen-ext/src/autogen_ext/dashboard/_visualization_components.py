"""
Visualization components for the dashboard.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum


class ChartType(Enum):
    """Types of charts available."""
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    AREA = "area"
    SCATTER = "scatter"
    GAUGE = "gauge"
    HEATMAP = "heatmap"


class MetricWidgetType(Enum):
    """Types of metric widgets."""
    COUNTER = "counter"
    GAUGE = "gauge"
    PROGRESS = "progress"
    STATUS = "status"
    TREND = "trend"


@dataclass
class ChartDataPoint:
    """Represents a single data point in a chart."""
    x: Union[str, float, int]
    y: Union[str, float, int]
    label: Optional[str] = None
    color: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChartSeries:
    """Represents a data series in a chart."""
    name: str
    data: List[ChartDataPoint]
    color: Optional[str] = None
    type: Optional[ChartType] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChartComponent:
    """Chart component for visualizing data."""
    
    def __init__(
        self,
        chart_id: str,
        title: str,
        chart_type: ChartType = ChartType.LINE,
        width: int = 400,
        height: int = 300
    ):
        self.chart_id = chart_id
        self.title = title
        self.chart_type = chart_type
        self.width = width
        self.height = height
        
        self.series: List[ChartSeries] = []
        self.options: Dict[str, Any] = {}
        self.last_updated = time.time()
    
    def add_series(self, series: ChartSeries) -> None:
        """Add a data series to the chart."""
        self.series.append(series)
        self.last_updated = time.time()
    
    def update_series(self, series_name: str, data: List[ChartDataPoint]) -> bool:
        """Update an existing data series."""
        for series in self.series:
            if series.name == series_name:
                series.data = data
                self.last_updated = time.time()
                return True
        return False
    
    def remove_series(self, series_name: str) -> bool:
        """Remove a data series from the chart."""
        for i, series in enumerate(self.series):
            if series.name == series_name:
                del self.series[i]
                self.last_updated = time.time()
                return True
        return False
    
    def set_options(self, options: Dict[str, Any]) -> None:
        """Set chart options."""
        self.options.update(options)
        self.last_updated = time.time()
    
    def get_chart_config(self) -> Dict[str, Any]:
        """Get Chart.js configuration."""
        
        datasets = []
        labels = set()
        
        for series in self.series:
            # Collect all labels
            for point in series.data:
                labels.add(str(point.x))
            
            # Create dataset
            dataset = {
                "label": series.name,
                "data": [{"x": point.x, "y": point.y} for point in series.data],
                "borderColor": series.color or self._get_default_color(len(datasets)),
                "backgroundColor": series.color or self._get_default_color(len(datasets), alpha=0.2),
                "tension": 0.1 if self.chart_type == ChartType.LINE else 0
            }
            
            if self.chart_type == ChartType.BAR:
                dataset["type"] = "bar"
            elif self.chart_type == ChartType.AREA:
                dataset["fill"] = True
            
            datasets.append(dataset)
        
        config = {
            "type": self.chart_type.value,
            "data": {
                "labels": sorted(list(labels)),
                "datasets": datasets
            },
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "scales": {
                    "y": {
                        "beginAtZero": True
                    }
                },
                **self.options
            }
        }
        
        return config
    
    def _get_default_color(self, index: int, alpha: float = 1.0) -> str:
        """Get default color for series."""
        colors = [
            f"rgba(59, 130, 246, {alpha})",   # Blue
            f"rgba(16, 185, 129, {alpha})",   # Green
            f"rgba(245, 101, 101, {alpha})",  # Red
            f"rgba(139, 92, 246, {alpha})",   # Purple
            f"rgba(245, 158, 11, {alpha})",   # Orange
            f"rgba(236, 72, 153, {alpha})",   # Pink
            f"rgba(14, 165, 233, {alpha})",   # Light Blue
            f"rgba(34, 197, 94, {alpha})",    # Light Green
        ]
        return colors[index % len(colors)]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chart to dictionary representation."""
        return {
            "chart_id": self.chart_id,
            "title": self.title,
            "type": self.chart_type.value,
            "width": self.width,
            "height": self.height,
            "config": self.get_chart_config(),
            "last_updated": self.last_updated
        }


class MetricWidget:
    """Widget for displaying metrics."""
    
    def __init__(
        self,
        widget_id: str,
        title: str,
        widget_type: MetricWidgetType = MetricWidgetType.GAUGE,
        unit: str = "",
        format_string: str = "{value}"
    ):
        self.widget_id = widget_id
        self.title = title
        self.widget_type = widget_type
        self.unit = unit
        self.format_string = format_string
        
        self.value: Union[int, float] = 0
        self.target_value: Optional[Union[int, float]] = None
        self.min_value: Optional[Union[int, float]] = None
        self.max_value: Optional[Union[int, float]] = None
        
        self.status: str = "normal"  # normal, warning, critical
        self.color: Optional[str] = None
        self.trend: Optional[str] = None  # up, down, stable
        
        self.history: List[Dict[str, Any]] = []
        self.last_updated = time.time()
    
    def update_value(
        self,
        value: Union[int, float],
        status: Optional[str] = None,
        trend: Optional[str] = None
    ) -> None:
        """Update the widget value."""
        
        # Store history
        self.history.append({
            "value": self.value,
            "timestamp": self.last_updated
        })
        
        # Keep only recent history
        if len(self.history) > 100:
            self.history = self.history[-100:]
        
        self.value = value
        
        if status:
            self.status = status
        
        if trend:
            self.trend = trend
        
        self.last_updated = time.time()
    
    def set_thresholds(
        self,
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        target_value: Optional[Union[int, float]] = None
    ) -> None:
        """Set threshold values for the widget."""
        
        self.min_value = min_value
        self.max_value = max_value
        self.target_value = target_value
    
    def get_formatted_value(self) -> str:
        """Get formatted value string."""
        
        formatted = self.format_string.format(
            value=self.value,
            unit=self.unit
        )
        
        return formatted
    
    def get_progress_percentage(self) -> float:
        """Get progress as percentage (for progress widgets)."""
        
        if self.widget_type != MetricWidgetType.PROGRESS:
            return 0.0
        
        if self.max_value is None:
            return 0.0
        
        min_val = self.min_value or 0
        max_val = self.max_value
        
        if max_val <= min_val:
            return 0.0
        
        progress = ((self.value - min_val) / (max_val - min_val)) * 100
        return max(0, min(100, progress))
    
    def calculate_trend(self) -> str:
        """Calculate trend based on recent history."""
        
        if len(self.history) < 2:
            return "stable"
        
        recent_values = [h["value"] for h in self.history[-5:]]
        
        if len(recent_values) < 2:
            return "stable"
        
        # Simple trend calculation
        first_half = sum(recent_values[:len(recent_values)//2])
        second_half = sum(recent_values[len(recent_values)//2:])
        
        first_avg = first_half / (len(recent_values)//2)
        second_avg = second_half / (len(recent_values) - len(recent_values)//2)
        
        if second_avg > first_avg * 1.05:  # 5% increase
            return "up"
        elif second_avg < first_avg * 0.95:  # 5% decrease
            return "down"
        else:
            return "stable"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert widget to dictionary representation."""
        
        return {
            "widget_id": self.widget_id,
            "title": self.title,
            "type": self.widget_type.value,
            "value": self.value,
            "formatted_value": self.get_formatted_value(),
            "unit": self.unit,
            "status": self.status,
            "color": self.color,
            "trend": self.trend or self.calculate_trend(),
            "progress_percentage": self.get_progress_percentage(),
            "target_value": self.target_value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "last_updated": self.last_updated,
            "history": self.history[-10:]  # Last 10 values
        }


class LogViewer:
    """Log viewer component for displaying log entries."""
    
    def __init__(
        self,
        viewer_id: str,
        title: str = "Activity Log",
        max_entries: int = 1000,
        auto_scroll: bool = True
    ):
        self.viewer_id = viewer_id
        self.title = title
        self.max_entries = max_entries
        self.auto_scroll = auto_scroll
        
        self.entries: List[Dict[str, Any]] = []
        self.filters: Dict[str, Any] = {}
        self.last_updated = time.time()
    
    def add_entry(
        self,
        message: str,
        level: str = "INFO",
        source: Optional[str] = None,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a log entry."""
        
        entry = {
            "id": len(self.entries),
            "message": message,
            "level": level.upper(),
            "source": source,
            "timestamp": timestamp or time.time(),
            "metadata": metadata or {}
        }
        
        self.entries.append(entry)
        
        # Trim old entries
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]
        
        self.last_updated = time.time()
    
    def add_entries(self, entries: List[Dict[str, Any]]) -> None:
        """Add multiple log entries."""
        
        for entry in entries:
            self.add_entry(
                message=entry.get("message", ""),
                level=entry.get("level", "INFO"),
                source=entry.get("source"),
                timestamp=entry.get("timestamp"),
                metadata=entry.get("metadata")
            )
    
    def set_filters(self, filters: Dict[str, Any]) -> None:
        """Set log filters."""
        
        self.filters = filters
        self.last_updated = time.time()
    
    def get_filtered_entries(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get filtered log entries."""
        
        filtered_entries = self.entries
        
        # Apply level filter
        if "level" in self.filters:
            allowed_levels = self.filters["level"]
            if isinstance(allowed_levels, str):
                allowed_levels = [allowed_levels]
            
            filtered_entries = [
                entry for entry in filtered_entries
                if entry["level"] in allowed_levels
            ]
        
        # Apply source filter
        if "source" in self.filters:
            allowed_sources = self.filters["source"]
            if isinstance(allowed_sources, str):
                allowed_sources = [allowed_sources]
            
            filtered_entries = [
                entry for entry in filtered_entries
                if entry["source"] in allowed_sources
            ]
        
        # Apply time range filter
        if "time_range" in self.filters:
            start_time = self.filters["time_range"].get("start")
            end_time = self.filters["time_range"].get("end")
            
            if start_time:
                filtered_entries = [
                    entry for entry in filtered_entries
                    if entry["timestamp"] >= start_time
                ]
            
            if end_time:
                filtered_entries = [
                    entry for entry in filtered_entries
                    if entry["timestamp"] <= end_time
                ]
        
        # Apply text search
        if "search" in self.filters:
            search_term = self.filters["search"].lower()
            filtered_entries = [
                entry for entry in filtered_entries
                if search_term in entry["message"].lower()
            ]
        
        # Sort by timestamp (newest first)
        filtered_entries.sort(key=lambda e: e["timestamp"], reverse=True)
        
        # Apply limit
        if limit:
            filtered_entries = filtered_entries[:limit]
        
        return filtered_entries
    
    def clear_entries(self) -> None:
        """Clear all log entries."""
        
        self.entries.clear()
        self.last_updated = time.time()
    
    def export_entries(self, format: str = "json") -> Union[str, List[Dict[str, Any]]]:
        """Export log entries."""
        
        if format == "json":
            return self.entries
        
        elif format == "text":
            lines = []
            for entry in self.entries:
                timestamp_str = time.strftime(
                    "%Y-%m-%d %H:%M:%S",
                    time.localtime(entry["timestamp"])
                )
                
                line = f"[{timestamp_str}] {entry['level']}"
                
                if entry["source"]:
                    line += f" ({entry['source']})"
                
                line += f": {entry['message']}"
                lines.append(line)
            
            return "\n".join(lines)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get log statistics."""
        
        level_counts = {}
        source_counts = {}
        
        for entry in self.entries:
            level = entry["level"]
            source = entry["source"] or "unknown"
            
            level_counts[level] = level_counts.get(level, 0) + 1
            source_counts[source] = source_counts.get(source, 0) + 1
        
        return {
            "total_entries": len(self.entries),
            "level_counts": level_counts,
            "source_counts": source_counts,
            "time_range": {
                "start": min(e["timestamp"] for e in self.entries) if self.entries else None,
                "end": max(e["timestamp"] for e in self.entries) if self.entries else None
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert log viewer to dictionary representation."""
        
        return {
            "viewer_id": self.viewer_id,
            "title": self.title,
            "max_entries": self.max_entries,
            "auto_scroll": self.auto_scroll,
            "entries": self.get_filtered_entries(limit=50),  # Last 50 entries
            "filters": self.filters,
            "statistics": self.get_statistics(),
            "last_updated": self.last_updated
        }


class DashboardLayout:
    """Manages dashboard layout and components."""
    
    def __init__(self, layout_id: str, title: str = "Dashboard"):
        self.layout_id = layout_id
        self.title = title
        
        self.charts: Dict[str, ChartComponent] = {}
        self.widgets: Dict[str, MetricWidget] = {}
        self.log_viewers: Dict[str, LogViewer] = {}
        
        self.layout_config: Dict[str, Any] = {
            "grid": {
                "columns": 12,
                "row_height": 60,
                "margin": [10, 10]
            },
            "components": []
        }
    
    def add_chart(
        self,
        chart: ChartComponent,
        position: Dict[str, int] = None
    ) -> None:
        """Add a chart to the dashboard."""
        
        self.charts[chart.chart_id] = chart
        
        # Add to layout
        component_config = {
            "id": chart.chart_id,
            "type": "chart",
            "x": position.get("x", 0) if position else 0,
            "y": position.get("y", 0) if position else 0,
            "w": position.get("w", 6) if position else 6,
            "h": position.get("h", 4) if position else 4
        }
        
        self.layout_config["components"].append(component_config)
    
    def add_widget(
        self,
        widget: MetricWidget,
        position: Dict[str, int] = None
    ) -> None:
        """Add a widget to the dashboard."""
        
        self.widgets[widget.widget_id] = widget
        
        # Add to layout
        component_config = {
            "id": widget.widget_id,
            "type": "widget",
            "x": position.get("x", 0) if position else 0,
            "y": position.get("y", 0) if position else 0,
            "w": position.get("w", 3) if position else 3,
            "h": position.get("h", 2) if position else 2
        }
        
        self.layout_config["components"].append(component_config)
    
    def add_log_viewer(
        self,
        log_viewer: LogViewer,
        position: Dict[str, int] = None
    ) -> None:
        """Add a log viewer to the dashboard."""
        
        self.log_viewers[log_viewer.viewer_id] = log_viewer
        
        # Add to layout
        component_config = {
            "id": log_viewer.viewer_id,
            "type": "log_viewer",
            "x": position.get("x", 0) if position else 0,
            "y": position.get("y", 0) if position else 0,
            "w": position.get("w", 12) if position else 12,
            "h": position.get("h", 6) if position else 6
        }
        
        self.layout_config["components"].append(component_config)
    
    def remove_component(self, component_id: str) -> bool:
        """Remove a component from the dashboard."""
        
        # Remove from collections
        removed = False
        
        if component_id in self.charts:
            del self.charts[component_id]
            removed = True
        
        if component_id in self.widgets:
            del self.widgets[component_id]
            removed = True
        
        if component_id in self.log_viewers:
            del self.log_viewers[component_id]
            removed = True
        
        # Remove from layout
        self.layout_config["components"] = [
            comp for comp in self.layout_config["components"]
            if comp["id"] != component_id
        ]
        
        return removed
    
    def update_layout(self, layout_config: Dict[str, Any]) -> None:
        """Update the dashboard layout configuration."""
        
        self.layout_config.update(layout_config)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert dashboard layout to dictionary representation."""
        
        return {
            "layout_id": self.layout_id,
            "title": self.title,
            "layout_config": self.layout_config,
            "charts": {
                chart_id: chart.to_dict()
                for chart_id, chart in self.charts.items()
            },
            "widgets": {
                widget_id: widget.to_dict()
                for widget_id, widget in self.widgets.items()
            },
            "log_viewers": {
                viewer_id: viewer.to_dict()
                for viewer_id, viewer in self.log_viewers.items()
            }
        }
