"""
Security monitoring and alerting system for AutoGen enhanced system.
"""

import asyncio
import time
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable
from collections import defaultdict, deque
import hashlib


class SecurityEventType(Enum):
    """Types of security events."""
    PROMPT_INJECTION = "prompt_injection"
    INPUT_VALIDATION_FAILURE = "input_validation_failure"
    CONTENT_FILTER_VIOLATION = "content_filter_violation"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    AUTHENTICATION_FAILURE = "authentication_failure"
    AUTHORIZATION_FAILURE = "authorization_failure"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    DATA_BREACH_ATTEMPT = "data_breach_attempt"
    SYSTEM_COMPROMISE = "system_compromise"


class AlertSeverity(Enum):
    """Severity levels for security alerts."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Represents a security event."""
    event_id: str
    event_type: SecurityEventType
    severity: AlertSeverity
    timestamp: float
    source: str
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    description: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_notes: str = ""


@dataclass
class SecurityMetrics:
    """Security metrics and statistics."""
    total_events: int = 0
    events_by_type: Dict[SecurityEventType, int] = field(default_factory=lambda: defaultdict(int))
    events_by_severity: Dict[AlertSeverity, int] = field(default_factory=lambda: defaultdict(int))
    events_by_source: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    blocked_attempts: int = 0
    false_positives: int = 0
    response_times: List[float] = field(default_factory=list)
    uptime: float = 0.0
    last_updated: float = field(default_factory=time.time)


@dataclass
class AlertConfig:
    """Configuration for security alerts."""
    enable_email_alerts: bool = False
    enable_webhook_alerts: bool = False
    enable_dashboard_alerts: bool = True
    enable_log_alerts: bool = True
    
    email_recipients: List[str] = field(default_factory=list)
    webhook_urls: List[str] = field(default_factory=list)
    
    # Alert thresholds
    alert_on_severity: List[AlertSeverity] = field(default_factory=lambda: [AlertSeverity.HIGH, AlertSeverity.CRITICAL])
    rate_limit_threshold: int = 10  # Events per minute
    burst_detection_window: int = 60  # Seconds
    
    # Notification settings
    batch_alerts: bool = True
    batch_interval: int = 300  # 5 minutes
    max_alerts_per_batch: int = 50


class SecurityMonitor:
    """Monitors security events and manages alerts."""
    
    def __init__(self, alert_config: Optional[AlertConfig] = None):
        self.alert_config = alert_config or AlertConfig()
        
        # Event storage
        self.events: Dict[str, SecurityEvent] = {}
        self.event_history: deque = deque(maxlen=10000)
        
        # Metrics
        self.metrics = SecurityMetrics()
        self.start_time = time.time()
        
        # Alert management
        self.alert_handlers: Dict[str, Callable] = {}
        self.pending_alerts: List[SecurityEvent] = []
        self.alert_queue: asyncio.Queue = asyncio.Queue()
        
        # Pattern detection
        self.user_activity: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.source_activity: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Background tasks
        self.monitor_task: Optional[asyncio.Task] = None
        self.alert_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # Setup default alert handlers
        self._setup_default_handlers()
    
    def _setup_default_handlers(self):
        """Setup default alert handlers."""
        
        self.register_alert_handler("log", self._log_alert_handler)
        self.register_alert_handler("dashboard", self._dashboard_alert_handler)
        
        if self.alert_config.enable_email_alerts:
            self.register_alert_handler("email", self._email_alert_handler)
        
        if self.alert_config.enable_webhook_alerts:
            self.register_alert_handler("webhook", self._webhook_alert_handler)
    
    async def start_monitoring(self):
        """Start the security monitoring system."""
        
        self.is_running = True
        
        # Start background tasks
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        self.alert_task = asyncio.create_task(self._alert_processing_loop())
    
    async def stop_monitoring(self):
        """Stop the security monitoring system."""
        
        self.is_running = False
        
        # Cancel background tasks
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        if self.alert_task:
            self.alert_task.cancel()
            try:
                await self.alert_task
            except asyncio.CancelledError:
                pass
    
    async def log_event(
        self,
        event_type: SecurityEventType,
        severity: AlertSeverity,
        source: str,
        description: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log a security event."""
        
        # Generate event ID
        event_data = f"{event_type.value}:{source}:{time.time()}"
        event_id = hashlib.sha256(event_data.encode()).hexdigest()[:16]
        
        # Create event
        event = SecurityEvent(
            event_id=event_id,
            event_type=event_type,
            severity=severity,
            timestamp=time.time(),
            source=source,
            user_id=user_id,
            agent_id=agent_id,
            description=description,
            details=details or {},
            metadata=metadata or {}
        )
        
        # Store event
        self.events[event_id] = event
        self.event_history.append(event)
        
        # Update metrics
        self.metrics.total_events += 1
        self.metrics.events_by_type[event_type] += 1
        self.metrics.events_by_severity[severity] += 1
        self.metrics.events_by_source[source] += 1
        self.metrics.last_updated = time.time()
        
        # Track user/source activity
        if user_id:
            self.user_activity[user_id].append(time.time())
        self.source_activity[source].append(time.time())
        
        # Check if alert should be triggered
        if severity in self.alert_config.alert_on_severity:
            await self.alert_queue.put(event)
        
        # Detect patterns
        await self._detect_suspicious_patterns(event)
        
        return event_id
    
    async def _detect_suspicious_patterns(self, event: SecurityEvent):
        """Detect suspicious activity patterns."""
        
        current_time = time.time()
        
        # Check for rate limiting violations
        if event.user_id:
            user_events = self.user_activity[event.user_id]
            recent_events = [t for t in user_events if current_time - t < 60]  # Last minute
            
            if len(recent_events) > self.alert_config.rate_limit_threshold:
                await self.log_event(
                    event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
                    severity=AlertSeverity.HIGH,
                    source="security_monitor",
                    description=f"User {event.user_id} exceeded rate limit",
                    user_id=event.user_id,
                    details={"events_per_minute": len(recent_events)}
                )
        
        # Check for burst activity from source
        source_events = self.source_activity[event.source]
        recent_source_events = [t for t in source_events if current_time - t < self.alert_config.burst_detection_window]
        
        if len(recent_source_events) > 20:  # Threshold for burst detection
            await self.log_event(
                event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
                severity=AlertSeverity.MEDIUM,
                source="security_monitor",
                description=f"Burst activity detected from {event.source}",
                details={"events_in_window": len(recent_source_events)}
            )
        
        # Check for escalating severity
        if event.user_id:
            user_recent_events = [
                e for e in self.event_history
                if e.user_id == event.user_id and current_time - e.timestamp < 300  # Last 5 minutes
            ]
            
            severity_progression = [e.severity for e in user_recent_events[-5:]]  # Last 5 events
            
            if len(severity_progression) >= 3:
                severity_values = {
                    AlertSeverity.LOW: 1,
                    AlertSeverity.MEDIUM: 2,
                    AlertSeverity.HIGH: 3,
                    AlertSeverity.CRITICAL: 4
                }
                
                values = [severity_values[s] for s in severity_progression]
                if all(values[i] <= values[i+1] for i in range(len(values)-1)):  # Escalating
                    await self.log_event(
                        event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
                        severity=AlertSeverity.HIGH,
                        source="security_monitor",
                        description=f"Escalating security events for user {event.user_id}",
                        user_id=event.user_id,
                        details={"severity_progression": [s.value for s in severity_progression]}
                    )
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        
        while self.is_running:
            try:
                # Update uptime
                self.metrics.uptime = time.time() - self.start_time
                
                # Clean old events from activity tracking
                current_time = time.time()
                cutoff_time = current_time - 3600  # 1 hour
                
                for user_id, activity in self.user_activity.items():
                    while activity and activity[0] < cutoff_time:
                        activity.popleft()
                
                for source, activity in self.source_activity.items():
                    while activity and activity[0] < cutoff_time:
                        activity.popleft()
                
                # Check for system health issues
                await self._check_system_health()
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)
    
    async def _check_system_health(self):
        """Check system health and detect anomalies."""
        
        current_time = time.time()
        
        # Check for high error rates
        recent_events = [
            e for e in self.event_history
            if current_time - e.timestamp < 300  # Last 5 minutes
        ]
        
        if len(recent_events) > 100:  # High event rate
            critical_events = [e for e in recent_events if e.severity == AlertSeverity.CRITICAL]
            
            if len(critical_events) > 10:
                await self.log_event(
                    event_type=SecurityEventType.SYSTEM_COMPROMISE,
                    severity=AlertSeverity.CRITICAL,
                    source="security_monitor",
                    description="High rate of critical security events detected",
                    details={
                        "total_events": len(recent_events),
                        "critical_events": len(critical_events),
                        "time_window": "5 minutes"
                    }
                )
    
    async def _alert_processing_loop(self):
        """Process security alerts."""
        
        batch_alerts = []
        last_batch_time = time.time()
        
        while self.is_running:
            try:
                # Wait for alerts with timeout
                try:
                    event = await asyncio.wait_for(self.alert_queue.get(), timeout=30)
                    batch_alerts.append(event)
                except asyncio.TimeoutError:
                    pass  # Continue to check batch processing
                
                current_time = time.time()
                
                # Process batch if conditions are met
                should_process_batch = (
                    len(batch_alerts) >= self.alert_config.max_alerts_per_batch or
                    (batch_alerts and current_time - last_batch_time >= self.alert_config.batch_interval)
                )
                
                if should_process_batch:
                    if self.alert_config.batch_alerts and len(batch_alerts) > 1:
                        await self._process_batch_alerts(batch_alerts)
                    else:
                        for alert_event in batch_alerts:
                            await self._process_single_alert(alert_event)
                    
                    batch_alerts.clear()
                    last_batch_time = current_time
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in alert processing: {e}")
                await asyncio.sleep(1)
    
    async def _process_single_alert(self, event: SecurityEvent):
        """Process a single security alert."""
        
        # Call all registered alert handlers
        for handler_name, handler in self.alert_handlers.items():
            try:
                await handler(event)
            except Exception as e:
                print(f"Error in alert handler {handler_name}: {e}")
    
    async def _process_batch_alerts(self, events: List[SecurityEvent]):
        """Process a batch of security alerts."""
        
        # Group events by type and severity
        grouped_events = defaultdict(list)
        for event in events:
            key = (event.event_type, event.severity)
            grouped_events[key].append(event)
        
        # Create batch alert summary
        batch_summary = {
            "timestamp": time.time(),
            "total_events": len(events),
            "event_groups": {}
        }
        
        for (event_type, severity), group_events in grouped_events.items():
            batch_summary["event_groups"][f"{event_type.value}_{severity.value}"] = {
                "count": len(group_events),
                "events": [e.event_id for e in group_events]
            }
        
        # Call batch alert handlers
        for handler_name, handler in self.alert_handlers.items():
            try:
                if hasattr(handler, '__name__') and 'batch' in handler.__name__:
                    await handler(events, batch_summary)
                else:
                    # For non-batch handlers, send summary as single event
                    summary_event = SecurityEvent(
                        event_id=f"batch_{int(time.time())}",
                        event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
                        severity=max(e.severity for e in events),
                        timestamp=time.time(),
                        source="security_monitor",
                        description=f"Batch alert: {len(events)} security events",
                        details=batch_summary
                    )
                    await handler(summary_event)
            except Exception as e:
                print(f"Error in batch alert handler {handler_name}: {e}")
    
    def register_alert_handler(self, name: str, handler: Callable):
        """Register an alert handler."""
        self.alert_handlers[name] = handler
    
    def unregister_alert_handler(self, name: str) -> bool:
        """Unregister an alert handler."""
        if name in self.alert_handlers:
            del self.alert_handlers[name]
            return True
        return False
    
    # Default alert handlers
    
    async def _log_alert_handler(self, event: SecurityEvent):
        """Log alert handler."""
        
        log_message = (
            f"SECURITY ALERT [{event.severity.value.upper()}] "
            f"{event.event_type.value}: {event.description}"
        )
        
        if event.user_id:
            log_message += f" (User: {event.user_id})"
        
        if event.agent_id:
            log_message += f" (Agent: {event.agent_id})"
        
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {log_message}")
    
    async def _dashboard_alert_handler(self, event: SecurityEvent):
        """Dashboard alert handler."""
        
        # This would integrate with the dashboard system
        # For now, just store in pending alerts
        self.pending_alerts.append(event)
        
        # Keep only recent alerts
        if len(self.pending_alerts) > 100:
            self.pending_alerts = self.pending_alerts[-50:]
    
    async def _email_alert_handler(self, event: SecurityEvent):
        """Email alert handler."""
        
        # This would send actual emails
        # For now, just simulate
        print(f"EMAIL ALERT: Would send email about {event.event_type.value} to {self.alert_config.email_recipients}")
    
    async def _webhook_alert_handler(self, event: SecurityEvent):
        """Webhook alert handler."""
        
        # This would send HTTP requests to webhooks
        # For now, just simulate
        webhook_payload = {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "severity": event.severity.value,
            "timestamp": event.timestamp,
            "description": event.description,
            "source": event.source
        }
        
        print(f"WEBHOOK ALERT: Would send to {self.alert_config.webhook_urls}")
        print(f"Payload: {json.dumps(webhook_payload, indent=2)}")
    
    # Query and management methods
    
    def get_events(
        self,
        event_type: Optional[SecurityEventType] = None,
        severity: Optional[AlertSeverity] = None,
        source: Optional[str] = None,
        user_id: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: Optional[int] = None
    ) -> List[SecurityEvent]:
        """Get security events with filtering."""
        
        events = list(self.event_history)
        
        # Apply filters
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if severity:
            events = [e for e in events if e.severity == severity]
        
        if source:
            events = [e for e in events if e.source == source]
        
        if user_id:
            events = [e for e in events if e.user_id == user_id]
        
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        
        # Sort by timestamp (newest first)
        events.sort(key=lambda e: e.timestamp, reverse=True)
        
        # Apply limit
        if limit:
            events = events[:limit]
        
        return events
    
    def get_metrics(self) -> SecurityMetrics:
        """Get current security metrics."""
        
        # Update uptime
        self.metrics.uptime = time.time() - self.start_time
        
        return self.metrics
    
    def get_pending_alerts(self) -> List[SecurityEvent]:
        """Get pending dashboard alerts."""
        
        return self.pending_alerts.copy()
    
    def resolve_event(self, event_id: str, resolution_notes: str = "") -> bool:
        """Mark an event as resolved."""
        
        if event_id in self.events:
            self.events[event_id].resolved = True
            self.events[event_id].resolution_notes = resolution_notes
            return True
        
        return False
    
    def mark_false_positive(self, event_id: str) -> bool:
        """Mark an event as a false positive."""
        
        if event_id in self.events:
            self.events[event_id].metadata["false_positive"] = True
            self.metrics.false_positives += 1
            return True
        
        return False
    
    def get_statistics(self, time_window: int = 3600) -> Dict[str, Any]:
        """Get security statistics for a time window."""
        
        current_time = time.time()
        start_time = current_time - time_window
        
        recent_events = [
            e for e in self.event_history
            if e.timestamp >= start_time
        ]
        
        # Calculate statistics
        stats = {
            "time_window_hours": time_window / 3600,
            "total_events": len(recent_events),
            "events_by_type": defaultdict(int),
            "events_by_severity": defaultdict(int),
            "events_by_source": defaultdict(int),
            "unique_users": len(set(e.user_id for e in recent_events if e.user_id)),
            "unique_sources": len(set(e.source for e in recent_events)),
            "resolved_events": len([e for e in recent_events if e.resolved]),
            "false_positives": len([e for e in recent_events if e.metadata.get("false_positive")]),
        }
        
        for event in recent_events:
            stats["events_by_type"][event.event_type.value] += 1
            stats["events_by_severity"][event.severity.value] += 1
            stats["events_by_source"][event.source] += 1
        
        return dict(stats)
    
    def export_events(
        self,
        format: str = "json",
        include_resolved: bool = True,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> Union[str, List[Dict[str, Any]]]:
        """Export security events."""
        
        events = list(self.event_history)
        
        # Apply filters
        if not include_resolved:
            events = [e for e in events if not e.resolved]
        
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        
        if format == "json":
            return [
                {
                    "event_id": e.event_id,
                    "event_type": e.event_type.value,
                    "severity": e.severity.value,
                    "timestamp": e.timestamp,
                    "source": e.source,
                    "user_id": e.user_id,
                    "agent_id": e.agent_id,
                    "description": e.description,
                    "details": e.details,
                    "resolved": e.resolved,
                    "resolution_notes": e.resolution_notes
                }
                for e in events
            ]
        
        elif format == "csv":
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow([
                "event_id", "event_type", "severity", "timestamp",
                "source", "user_id", "agent_id", "description", "resolved"
            ])
            
            # Write events
            for e in events:
                writer.writerow([
                    e.event_id, e.event_type.value, e.severity.value,
                    e.timestamp, e.source, e.user_id or "", e.agent_id or "",
                    e.description, e.resolved
                ])
            
            return output.getvalue()
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def update_alert_config(self, config: AlertConfig):
        """Update alert configuration."""
        
        self.alert_config = config
        
        # Re-setup handlers based on new config
        self._setup_default_handlers()
