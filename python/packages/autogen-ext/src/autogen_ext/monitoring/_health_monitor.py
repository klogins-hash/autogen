"""
Health monitoring system for tracking agent and system health.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from autogen_core import CancellationToken


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check definition."""
    name: str
    check_function: Callable[[], bool]
    description: str
    timeout_seconds: float = 30.0
    critical: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthResult:
    """Result of a health check."""
    check_name: str
    status: HealthStatus
    message: str
    timestamp: float
    duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class HealthMonitor:
    """Monitors system and agent health with configurable checks."""
    
    def __init__(self, check_interval: float = 60.0):
        self.check_interval = check_interval
        self.health_checks: Dict[str, HealthCheck] = {}
        self.last_results: Dict[str, HealthResult] = {}
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False
        self.status_callbacks: List[Callable[[Dict[str, HealthResult]], None]] = []
        
    def add_health_check(self, health_check: HealthCheck) -> None:
        """Add a health check to the monitor."""
        self.health_checks[health_check.name] = health_check
        
    def remove_health_check(self, check_name: str) -> None:
        """Remove a health check from the monitor."""
        self.health_checks.pop(check_name, None)
        self.last_results.pop(check_name, None)
        
    def add_status_callback(self, callback: Callable[[Dict[str, HealthResult]], None]) -> None:
        """Add callback to be notified of health status changes."""
        self.status_callbacks.append(callback)
        
    async def run_health_checks(self) -> Dict[str, HealthResult]:
        """Run all health checks and return results."""
        results = {}
        
        for check_name, health_check in self.health_checks.items():
            result = await self._run_single_check(health_check)
            results[check_name] = result
            self.last_results[check_name] = result
            
        # Notify callbacks of status changes
        for callback in self.status_callbacks:
            try:
                callback(results)
            except Exception:
                pass  # Don't let callback errors break monitoring
                
        return results
        
    async def _run_single_check(self, health_check: HealthCheck) -> HealthResult:
        """Run a single health check with timeout."""
        start_time = time.time()
        
        try:
            # Run check with timeout
            check_passed = await asyncio.wait_for(
                asyncio.to_thread(health_check.check_function),
                timeout=health_check.timeout_seconds
            )
            
            duration = time.time() - start_time
            
            if check_passed:
                status = HealthStatus.HEALTHY
                message = f"Check '{health_check.name}' passed"
            else:
                status = HealthStatus.CRITICAL if health_check.critical else HealthStatus.WARNING
                message = f"Check '{health_check.name}' failed"
                
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            status = HealthStatus.CRITICAL if health_check.critical else HealthStatus.WARNING
            message = f"Check '{health_check.name}' timed out after {health_check.timeout_seconds}s"
            
        except Exception as e:
            duration = time.time() - start_time
            status = HealthStatus.CRITICAL if health_check.critical else HealthStatus.WARNING
            message = f"Check '{health_check.name}' failed with error: {str(e)}"
            
        return HealthResult(
            check_name=health_check.name,
            status=status,
            message=message,
            timestamp=start_time,
            duration=duration,
            metadata=health_check.metadata.copy()
        )
        
    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status."""
        if not self.last_results:
            return HealthStatus.UNKNOWN
            
        statuses = [result.status for result in self.last_results.values()]
        
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
            
    def get_failing_checks(self) -> List[HealthResult]:
        """Get list of currently failing health checks."""
        return [
            result for result in self.last_results.values()
            if result.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]
        ]
        
    async def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
    async def stop_monitoring(self) -> None:
        """Stop continuous health monitoring."""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None
            
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                await self.run_health_checks()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception:
                # Continue monitoring even if individual checks fail
                await asyncio.sleep(self.check_interval)


# Predefined health checks for common scenarios
def create_memory_usage_check(max_memory_mb: int = 1024) -> HealthCheck:
    """Create a health check for memory usage."""
    import psutil
    
    def check_memory():
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        return memory_mb < max_memory_mb
        
    return HealthCheck(
        name="memory_usage",
        check_function=check_memory,
        description=f"Check if memory usage is below {max_memory_mb}MB",
        critical=True,
        metadata={"max_memory_mb": max_memory_mb}
    )


def create_disk_space_check(min_free_gb: int = 1) -> HealthCheck:
    """Create a health check for disk space."""
    import shutil
    
    def check_disk_space():
        free_bytes = shutil.disk_usage("/").free
        free_gb = free_bytes / 1024 / 1024 / 1024
        return free_gb > min_free_gb
        
    return HealthCheck(
        name="disk_space",
        check_function=check_disk_space,
        description=f"Check if free disk space is above {min_free_gb}GB",
        critical=True,
        metadata={"min_free_gb": min_free_gb}
    )


def create_api_connectivity_check(url: str, timeout: float = 10.0) -> HealthCheck:
    """Create a health check for API connectivity."""
    import requests
    
    def check_api():
        try:
            response = requests.get(url, timeout=timeout)
            return response.status_code == 200
        except Exception:
            return False
            
    return HealthCheck(
        name=f"api_connectivity_{url}",
        check_function=check_api,
        description=f"Check connectivity to {url}",
        timeout_seconds=timeout + 5,
        critical=False,
        metadata={"url": url, "timeout": timeout}
    )


def create_agent_response_check(agent_name: str, max_response_time: float = 30.0) -> HealthCheck:
    """Create a health check for agent responsiveness."""
    
    def check_agent_response():
        # This would integrate with actual agent monitoring
        # For now, return True as placeholder
        return True
        
    return HealthCheck(
        name=f"agent_response_{agent_name}",
        check_function=check_agent_response,
        description=f"Check if {agent_name} responds within {max_response_time}s",
        timeout_seconds=max_response_time + 5,
        critical=True,
        metadata={"agent_name": agent_name, "max_response_time": max_response_time}
    )
