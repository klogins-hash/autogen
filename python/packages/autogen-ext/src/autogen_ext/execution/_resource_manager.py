"""
Resource management for controlling agent execution resources and constraints.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from collections import defaultdict


class ResourceType(Enum):
    """Types of resources that can be managed."""
    CPU = "cpu"
    MEMORY = "memory"
    NETWORK = "network"
    API_CALLS = "api_calls"
    TOKENS = "tokens"
    CUSTOM = "custom"


@dataclass
class ResourceConstraint:
    """Constraint on resource usage."""
    resource_type: ResourceType
    max_value: float
    current_value: float = 0.0
    reserved_value: float = 0.0
    unit: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceReservation:
    """Reservation of resources for a task."""
    reservation_id: str
    task_id: str
    agent_name: str
    resources: Dict[ResourceType, float]
    timestamp: float = field(default_factory=time.time)
    expires_at: Optional[float] = None


class ResourcePool:
    """Manages a pool of resources with constraints and reservations."""
    
    def __init__(self, name: str):
        self.name = name
        self.constraints: Dict[ResourceType, ResourceConstraint] = {}
        self.reservations: Dict[str, ResourceReservation] = {}
        self.usage_history: List[Dict[str, Any]] = []
        
    def add_constraint(self, constraint: ResourceConstraint) -> None:
        """Add a resource constraint to the pool."""
        self.constraints[constraint.resource_type] = constraint
        
    def remove_constraint(self, resource_type: ResourceType) -> None:
        """Remove a resource constraint."""
        self.constraints.pop(resource_type, None)
        
    def can_reserve(self, resources: Dict[ResourceType, float]) -> bool:
        """Check if resources can be reserved."""
        for resource_type, amount in resources.items():
            if resource_type not in self.constraints:
                continue
                
            constraint = self.constraints[resource_type]
            available = constraint.max_value - constraint.current_value - constraint.reserved_value
            
            if amount > available:
                return False
                
        return True
        
    def reserve_resources(
        self,
        reservation_id: str,
        task_id: str,
        agent_name: str,
        resources: Dict[ResourceType, float],
        duration_seconds: Optional[float] = None
    ) -> bool:
        """Reserve resources for a task."""
        
        if not self.can_reserve(resources):
            return False
            
        # Create reservation
        expires_at = None
        if duration_seconds:
            expires_at = time.time() + duration_seconds
            
        reservation = ResourceReservation(
            reservation_id=reservation_id,
            task_id=task_id,
            agent_name=agent_name,
            resources=resources.copy(),
            expires_at=expires_at
        )
        
        # Update reserved amounts
        for resource_type, amount in resources.items():
            if resource_type in self.constraints:
                self.constraints[resource_type].reserved_value += amount
                
        self.reservations[reservation_id] = reservation
        return True
        
    def activate_reservation(self, reservation_id: str) -> bool:
        """Activate a reservation (move from reserved to current usage)."""
        
        if reservation_id not in self.reservations:
            return False
            
        reservation = self.reservations[reservation_id]
        
        # Move from reserved to current
        for resource_type, amount in reservation.resources.items():
            if resource_type in self.constraints:
                constraint = self.constraints[resource_type]
                constraint.reserved_value = max(0, constraint.reserved_value - amount)
                constraint.current_value += amount
                
        return True
        
    def release_resources(self, reservation_id: str) -> bool:
        """Release resources from a reservation."""
        
        if reservation_id not in self.reservations:
            return False
            
        reservation = self.reservations[reservation_id]
        
        # Release resources
        for resource_type, amount in reservation.resources.items():
            if resource_type in self.constraints:
                constraint = self.constraints[resource_type]
                constraint.current_value = max(0, constraint.current_value - amount)
                constraint.reserved_value = max(0, constraint.reserved_value - amount)
                
        # Record usage history
        self.usage_history.append({
            "reservation_id": reservation_id,
            "task_id": reservation.task_id,
            "agent_name": reservation.agent_name,
            "resources": reservation.resources.copy(),
            "duration": time.time() - reservation.timestamp,
            "released_at": time.time()
        })
        
        # Clean up
        del self.reservations[reservation_id]
        return True
        
    def cleanup_expired_reservations(self) -> int:
        """Clean up expired reservations and return count."""
        current_time = time.time()
        expired_ids = []
        
        for reservation_id, reservation in self.reservations.items():
            if reservation.expires_at and current_time > reservation.expires_at:
                expired_ids.append(reservation_id)
                
        for reservation_id in expired_ids:
            self.release_resources(reservation_id)
            
        return len(expired_ids)
        
    def get_utilization(self) -> Dict[ResourceType, float]:
        """Get current resource utilization percentages."""
        utilization = {}
        
        for resource_type, constraint in self.constraints.items():
            if constraint.max_value > 0:
                utilization[resource_type] = (constraint.current_value / constraint.max_value) * 100
            else:
                utilization[resource_type] = 0.0
                
        return utilization
        
    def get_available_resources(self) -> Dict[ResourceType, float]:
        """Get currently available resources."""
        available = {}
        
        for resource_type, constraint in self.constraints.items():
            available[resource_type] = constraint.max_value - constraint.current_value - constraint.reserved_value
            
        return available


class ResourceManager:
    """Manages multiple resource pools and global resource allocation."""
    
    def __init__(self):
        self.pools: Dict[str, ResourcePool] = {}
        self.global_reservations: Dict[str, str] = {}  # reservation_id -> pool_name
        self.agent_resource_usage: Dict[str, Dict[ResourceType, float]] = defaultdict(lambda: defaultdict(float))
        
    def create_pool(self, pool_name: str) -> ResourcePool:
        """Create a new resource pool."""
        pool = ResourcePool(pool_name)
        self.pools[pool_name] = pool
        return pool
        
    def get_pool(self, pool_name: str) -> Optional[ResourcePool]:
        """Get a resource pool by name."""
        return self.pools.get(pool_name)
        
    def reserve_resources_global(
        self,
        reservation_id: str,
        task_id: str,
        agent_name: str,
        resources: Dict[ResourceType, float],
        pool_name: str = "default",
        duration_seconds: Optional[float] = None
    ) -> bool:
        """Reserve resources globally across pools."""
        
        if pool_name not in self.pools:
            self.create_pool(pool_name)
            
        pool = self.pools[pool_name]
        
        if pool.reserve_resources(reservation_id, task_id, agent_name, resources, duration_seconds):
            self.global_reservations[reservation_id] = pool_name
            return True
            
        return False
        
    def activate_reservation_global(self, reservation_id: str) -> bool:
        """Activate a global reservation."""
        
        if reservation_id not in self.global_reservations:
            return False
            
        pool_name = self.global_reservations[reservation_id]
        pool = self.pools.get(pool_name)
        
        if pool:
            return pool.activate_reservation(reservation_id)
            
        return False
        
    def release_resources_global(self, reservation_id: str) -> bool:
        """Release resources globally."""
        
        if reservation_id not in self.global_reservations:
            return False
            
        pool_name = self.global_reservations[reservation_id]
        pool = self.pools.get(pool_name)
        
        if pool:
            success = pool.release_resources(reservation_id)
            if success:
                del self.global_reservations[reservation_id]
            return success
            
        return False
        
    def get_agent_resource_requirements(self, agent_name: str) -> Dict[ResourceType, float]:
        """Get estimated resource requirements for an agent based on history."""
        
        if agent_name not in self.agent_resource_usage:
            # Default requirements for unknown agents
            return {
                ResourceType.CPU: 1.0,
                ResourceType.MEMORY: 100.0,  # MB
                ResourceType.API_CALLS: 10.0,
                ResourceType.TOKENS: 1000.0
            }
            
        usage = self.agent_resource_usage[agent_name]
        
        # Add safety margin to historical usage
        requirements = {}
        for resource_type, avg_usage in usage.items():
            requirements[resource_type] = avg_usage * 1.2  # 20% safety margin
            
        return requirements
        
    def record_agent_usage(
        self,
        agent_name: str,
        resources: Dict[ResourceType, float]
    ) -> None:
        """Record actual resource usage for an agent."""
        
        for resource_type, amount in resources.items():
            current_avg = self.agent_resource_usage[agent_name][resource_type]
            # Simple moving average
            self.agent_resource_usage[agent_name][resource_type] = (current_avg * 0.9) + (amount * 0.1)
            
    def cleanup_all_expired(self) -> int:
        """Clean up expired reservations in all pools."""
        total_cleaned = 0
        
        for pool in self.pools.values():
            total_cleaned += pool.cleanup_expired_reservations()
            
        return total_cleaned
        
    def get_global_utilization(self) -> Dict[str, Dict[ResourceType, float]]:
        """Get utilization across all pools."""
        utilization = {}
        
        for pool_name, pool in self.pools.items():
            utilization[pool_name] = pool.get_utilization()
            
        return utilization
        
    def get_resource_bottlenecks(self, threshold: float = 80.0) -> List[Dict[str, Any]]:
        """Identify resource bottlenecks across pools."""
        bottlenecks = []
        
        for pool_name, pool in self.pools.items():
            utilization = pool.get_utilization()
            
            for resource_type, usage_percent in utilization.items():
                if usage_percent > threshold:
                    bottlenecks.append({
                        "pool": pool_name,
                        "resource_type": resource_type.value,
                        "utilization_percent": usage_percent,
                        "severity": "critical" if usage_percent > 95 else "warning"
                    })
                    
        return bottlenecks
        
    def optimize_resource_allocation(self) -> Dict[str, Any]:
        """Analyze and suggest resource allocation optimizations."""
        
        suggestions = []
        
        # Analyze utilization patterns
        for pool_name, pool in self.pools.items():
            utilization = pool.get_utilization()
            available = pool.get_available_resources()
            
            for resource_type, usage_percent in utilization.items():
                if usage_percent < 20:  # Under-utilized
                    suggestions.append({
                        "type": "reduce_allocation",
                        "pool": pool_name,
                        "resource": resource_type.value,
                        "current_utilization": usage_percent,
                        "suggestion": f"Consider reducing {resource_type.value} allocation in {pool_name}"
                    })
                elif usage_percent > 90:  # Over-utilized
                    suggestions.append({
                        "type": "increase_allocation",
                        "pool": pool_name,
                        "resource": resource_type.value,
                        "current_utilization": usage_percent,
                        "suggestion": f"Consider increasing {resource_type.value} allocation in {pool_name}"
                    })
                    
        return {
            "suggestions": suggestions,
            "total_pools": len(self.pools),
            "active_reservations": len(self.global_reservations)
        }
