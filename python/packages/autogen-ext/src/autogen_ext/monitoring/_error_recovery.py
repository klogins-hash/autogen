"""
Error recovery and resilience management for AutoGen agents.
"""

import asyncio
import logging
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Pattern, Union
import re

from autogen_core import CancellationToken


class RecoveryStrategy(Enum):
    """Available error recovery strategies."""
    RETRY = "retry"
    FALLBACK = "fallback"
    RESTART = "restart"
    IGNORE = "ignore"
    ESCALATE = "escalate"


@dataclass
class ErrorPattern:
    """Pattern for matching and handling specific errors."""
    name: str
    pattern: Union[str, Pattern]
    strategy: RecoveryStrategy
    max_retries: int = 3
    retry_delay: float = 1.0
    fallback_action: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorEvent:
    """Record of an error occurrence."""
    timestamp: float
    error_type: str
    error_message: str
    stack_trace: str
    context: Dict[str, Any]
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None


class ErrorRecoveryManager:
    """Manages error recovery strategies and resilience patterns."""
    
    def __init__(self, max_error_history: int = 1000):
        self.max_error_history = max_error_history
        self.error_patterns: List[ErrorPattern] = []
        self.error_history: List[ErrorEvent] = []
        self.retry_counts: Dict[str, int] = {}
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
        
    def add_error_pattern(self, pattern: ErrorPattern) -> None:
        """Add an error pattern for automatic recovery."""
        self.error_patterns.append(pattern)
        
    def remove_error_pattern(self, pattern_name: str) -> None:
        """Remove an error pattern."""
        self.error_patterns = [
            p for p in self.error_patterns 
            if p.name != pattern_name
        ]
        
    async def handle_error(
        self, 
        error: Exception, 
        context: Dict[str, Any] = None,
        operation: Optional[Callable] = None
    ) -> tuple[bool, Any]:
        """
        Handle an error using configured recovery strategies.
        
        Returns:
            (recovery_successful, result_or_none)
        """
        context = context or {}
        error_event = self._create_error_event(error, context)
        self.error_history.append(error_event)
        
        # Trim error history if needed
        if len(self.error_history) > self.max_error_history:
            self.error_history = self.error_history[-self.max_error_history:]
        
        # Find matching error pattern
        matching_pattern = self._find_matching_pattern(error)
        if not matching_pattern:
            self.logger.warning(f"No recovery pattern found for error: {error}")
            return False, None
            
        error_event.recovery_attempted = True
        error_event.recovery_strategy = matching_pattern.strategy
        
        # Apply recovery strategy
        try:
            success, result = await self._apply_recovery_strategy(
                matching_pattern, error, context, operation
            )
            error_event.recovery_successful = success
            return success, result
            
        except Exception as recovery_error:
            self.logger.error(f"Recovery strategy failed: {recovery_error}")
            error_event.recovery_successful = False
            return False, None
            
    def _create_error_event(self, error: Exception, context: Dict[str, Any]) -> ErrorEvent:
        """Create an error event record."""
        return ErrorEvent(
            timestamp=time.time(),
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            context=context.copy()
        )
        
    def _find_matching_pattern(self, error: Exception) -> Optional[ErrorPattern]:
        """Find the first matching error pattern."""
        error_str = str(error)
        error_type = type(error).__name__
        
        for pattern in self.error_patterns:
            if isinstance(pattern.pattern, str):
                if pattern.pattern in error_str or pattern.pattern in error_type:
                    return pattern
            elif hasattr(pattern.pattern, 'search'):
                if pattern.pattern.search(error_str) or pattern.pattern.search(error_type):
                    return pattern
                    
        return None
        
    async def _apply_recovery_strategy(
        self,
        pattern: ErrorPattern,
        error: Exception,
        context: Dict[str, Any],
        operation: Optional[Callable]
    ) -> tuple[bool, Any]:
        """Apply the specified recovery strategy."""
        
        if pattern.strategy == RecoveryStrategy.RETRY:
            return await self._retry_operation(pattern, operation, context)
            
        elif pattern.strategy == RecoveryStrategy.FALLBACK:
            return await self._fallback_operation(pattern, context)
            
        elif pattern.strategy == RecoveryStrategy.RESTART:
            return await self._restart_operation(pattern, context)
            
        elif pattern.strategy == RecoveryStrategy.IGNORE:
            self.logger.info(f"Ignoring error as per pattern '{pattern.name}': {error}")
            return True, None
            
        elif pattern.strategy == RecoveryStrategy.ESCALATE:
            self.logger.error(f"Escalating error as per pattern '{pattern.name}': {error}")
            raise error
            
        return False, None
        
    async def _retry_operation(
        self,
        pattern: ErrorPattern,
        operation: Optional[Callable],
        context: Dict[str, Any]
    ) -> tuple[bool, Any]:
        """Retry the failed operation."""
        if not operation:
            return False, None
            
        retry_key = f"{pattern.name}_{id(operation)}"
        current_retries = self.retry_counts.get(retry_key, 0)
        
        if current_retries >= pattern.max_retries:
            self.logger.error(f"Max retries ({pattern.max_retries}) exceeded for pattern '{pattern.name}'")
            self.retry_counts.pop(retry_key, None)
            return False, None
            
        self.retry_counts[retry_key] = current_retries + 1
        
        # Wait before retry
        if pattern.retry_delay > 0:
            await asyncio.sleep(pattern.retry_delay * (2 ** current_retries))  # Exponential backoff
            
        try:
            if asyncio.iscoroutinefunction(operation):
                result = await operation()
            else:
                result = operation()
                
            # Success - reset retry count
            self.retry_counts.pop(retry_key, None)
            return True, result
            
        except Exception as retry_error:
            self.logger.warning(f"Retry {current_retries + 1} failed for pattern '{pattern.name}': {retry_error}")
            return False, None
            
    async def _fallback_operation(
        self,
        pattern: ErrorPattern,
        context: Dict[str, Any]
    ) -> tuple[bool, Any]:
        """Execute fallback operation."""
        if not pattern.fallback_action:
            return False, None
            
        try:
            if asyncio.iscoroutinefunction(pattern.fallback_action):
                result = await pattern.fallback_action(context)
            else:
                result = pattern.fallback_action(context)
                
            return True, result
            
        except Exception as fallback_error:
            self.logger.error(f"Fallback operation failed for pattern '{pattern.name}': {fallback_error}")
            return False, None
            
    async def _restart_operation(
        self,
        pattern: ErrorPattern,
        context: Dict[str, Any]
    ) -> tuple[bool, Any]:
        """Restart operation (placeholder for agent restart logic)."""
        self.logger.info(f"Restart requested for pattern '{pattern.name}'")
        # This would integrate with actual agent restart mechanisms
        return True, None
        
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error and recovery statistics."""
        if not self.error_history:
            return {}
            
        total_errors = len(self.error_history)
        recovery_attempted = sum(1 for e in self.error_history if e.recovery_attempted)
        recovery_successful = sum(1 for e in self.error_history if e.recovery_successful)
        
        error_types = {}
        for event in self.error_history:
            error_types[event.error_type] = error_types.get(event.error_type, 0) + 1
            
        recent_errors = [
            e for e in self.error_history 
            if time.time() - e.timestamp < 3600  # Last hour
        ]
        
        return {
            "total_errors": total_errors,
            "recovery_attempted": recovery_attempted,
            "recovery_successful": recovery_successful,
            "recovery_rate": recovery_successful / recovery_attempted if recovery_attempted > 0 else 0,
            "error_types": error_types,
            "recent_errors_count": len(recent_errors),
            "active_retries": len(self.retry_counts)
        }
        
    def clear_error_history(self) -> None:
        """Clear error history and reset retry counts."""
        self.error_history.clear()
        self.retry_counts.clear()
        
    def is_circuit_breaker_open(self, service_name: str) -> bool:
        """Check if circuit breaker is open for a service."""
        breaker = self.circuit_breakers.get(service_name)
        if not breaker:
            return False
            
        if breaker["state"] == "open":
            # Check if we should try to close it
            if time.time() - breaker["last_failure"] > breaker["timeout"]:
                breaker["state"] = "half_open"
                return False
            return True
            
        return False
        
    def record_service_failure(self, service_name: str) -> None:
        """Record a service failure for circuit breaker logic."""
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = {
                "failure_count": 0,
                "last_failure": 0,
                "state": "closed",
                "timeout": 60  # 1 minute timeout
            }
            
        breaker = self.circuit_breakers[service_name]
        breaker["failure_count"] += 1
        breaker["last_failure"] = time.time()
        
        # Open circuit breaker if too many failures
        if breaker["failure_count"] >= 5:  # Configurable threshold
            breaker["state"] = "open"
            self.logger.warning(f"Circuit breaker opened for service: {service_name}")
            
    def record_service_success(self, service_name: str) -> None:
        """Record a service success for circuit breaker logic."""
        if service_name in self.circuit_breakers:
            breaker = self.circuit_breakers[service_name]
            breaker["failure_count"] = 0
            breaker["state"] = "closed"


# Predefined error patterns for common scenarios
def create_network_error_pattern() -> ErrorPattern:
    """Create pattern for network-related errors."""
    return ErrorPattern(
        name="network_errors",
        pattern=re.compile(r"(ConnectionError|TimeoutError|NetworkError|HTTPError)", re.IGNORECASE),
        strategy=RecoveryStrategy.RETRY,
        max_retries=3,
        retry_delay=2.0
    )


def create_rate_limit_pattern() -> ErrorPattern:
    """Create pattern for rate limiting errors."""
    return ErrorPattern(
        name="rate_limit",
        pattern=re.compile(r"(rate.?limit|429|too.?many.?requests)", re.IGNORECASE),
        strategy=RecoveryStrategy.RETRY,
        max_retries=5,
        retry_delay=10.0
    )


def create_memory_error_pattern() -> ErrorPattern:
    """Create pattern for memory-related errors."""
    return ErrorPattern(
        name="memory_errors",
        pattern=re.compile(r"(MemoryError|OutOfMemoryError)", re.IGNORECASE),
        strategy=RecoveryStrategy.RESTART,
        max_retries=1
    )


def create_api_key_error_pattern() -> ErrorPattern:
    """Create pattern for API key/authentication errors."""
    return ErrorPattern(
        name="auth_errors",
        pattern=re.compile(r"(unauthorized|invalid.?key|authentication)", re.IGNORECASE),
        strategy=RecoveryStrategy.ESCALATE
    )
