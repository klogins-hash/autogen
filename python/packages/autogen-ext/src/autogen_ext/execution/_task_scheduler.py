"""
Task scheduling and queue management for parallel agent execution.
"""

import asyncio
import heapq
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from collections import deque

from ._parallel_executor import ExecutionTask, TaskPriority


class SchedulingStrategy(Enum):
    """Available task scheduling strategies."""
    FIFO = "fifo"  # First In, First Out
    PRIORITY = "priority"  # Priority-based scheduling
    SHORTEST_JOB_FIRST = "sjf"  # Shortest estimated job first
    ROUND_ROBIN = "round_robin"  # Round-robin by agent type
    ADAPTIVE = "adaptive"  # Adaptive based on performance


@dataclass
class ScheduledTask:
    """Task with scheduling metadata."""
    task: ExecutionTask
    priority_score: float
    estimated_duration: float = 0.0
    queue_time: float = field(default_factory=time.time)
    
    def __lt__(self, other):
        """For priority queue ordering."""
        return self.priority_score > other.priority_score  # Higher priority first


class TaskQueue:
    """Intelligent task queue with multiple scheduling strategies."""
    
    def __init__(self, strategy: SchedulingStrategy = SchedulingStrategy.PRIORITY):
        self.strategy = strategy
        self.tasks: List[ScheduledTask] = []
        self.fifo_queue: deque = deque()
        self.round_robin_queues: Dict[str, deque] = {}
        self.round_robin_index = 0
        self.task_history: Dict[str, List[float]] = {}  # Agent -> execution times
        
    def enqueue(self, task: ExecutionTask, estimated_duration: float = 0.0) -> None:
        """Add a task to the queue."""
        
        if self.strategy == SchedulingStrategy.FIFO:
            self.fifo_queue.append(task)
            
        elif self.strategy == SchedulingStrategy.PRIORITY:
            priority_score = self._calculate_priority_score(task)
            scheduled_task = ScheduledTask(
                task=task,
                priority_score=priority_score,
                estimated_duration=estimated_duration
            )
            heapq.heappush(self.tasks, scheduled_task)
            
        elif self.strategy == SchedulingStrategy.SHORTEST_JOB_FIRST:
            duration = estimated_duration or self._estimate_duration(task)
            priority_score = 1.0 / (duration + 0.1)  # Shorter jobs get higher priority
            scheduled_task = ScheduledTask(
                task=task,
                priority_score=priority_score,
                estimated_duration=duration
            )
            heapq.heappush(self.tasks, scheduled_task)
            
        elif self.strategy == SchedulingStrategy.ROUND_ROBIN:
            agent_type = task.agent.__class__.__name__
            if agent_type not in self.round_robin_queues:
                self.round_robin_queues[agent_type] = deque()
            self.round_robin_queues[agent_type].append(task)
            
        elif self.strategy == SchedulingStrategy.ADAPTIVE:
            priority_score = self._calculate_adaptive_priority(task, estimated_duration)
            scheduled_task = ScheduledTask(
                task=task,
                priority_score=priority_score,
                estimated_duration=estimated_duration
            )
            heapq.heappush(self.tasks, scheduled_task)
    
    def dequeue(self) -> Optional[ExecutionTask]:
        """Remove and return the next task from the queue."""
        
        if self.strategy == SchedulingStrategy.FIFO:
            return self.fifo_queue.popleft() if self.fifo_queue else None
            
        elif self.strategy in [
            SchedulingStrategy.PRIORITY,
            SchedulingStrategy.SHORTEST_JOB_FIRST,
            SchedulingStrategy.ADAPTIVE
        ]:
            if self.tasks:
                scheduled_task = heapq.heappop(self.tasks)
                return scheduled_task.task
            return None
            
        elif self.strategy == SchedulingStrategy.ROUND_ROBIN:
            return self._round_robin_dequeue()
            
        return None
    
    def _round_robin_dequeue(self) -> Optional[ExecutionTask]:
        """Dequeue using round-robin strategy."""
        if not self.round_robin_queues:
            return None
        
        queue_names = list(self.round_robin_queues.keys())
        attempts = 0
        
        while attempts < len(queue_names):
            current_queue_name = queue_names[self.round_robin_index % len(queue_names)]
            current_queue = self.round_robin_queues[current_queue_name]
            
            if current_queue:
                task = current_queue.popleft()
                self.round_robin_index = (self.round_robin_index + 1) % len(queue_names)
                return task
            
            self.round_robin_index = (self.round_robin_index + 1) % len(queue_names)
            attempts += 1
        
        return None
    
    def _calculate_priority_score(self, task: ExecutionTask) -> float:
        """Calculate priority score for a task."""
        base_priority = task.priority.value
        
        # Age factor - older tasks get higher priority
        age_factor = (time.time() - task.created_at) / 3600  # Hours
        
        # Dependency factor - tasks with more dependents get higher priority
        dependency_factor = len(task.dependencies) * 0.1
        
        return base_priority + age_factor + dependency_factor
    
    def _calculate_adaptive_priority(self, task: ExecutionTask, estimated_duration: float) -> float:
        """Calculate adaptive priority based on historical performance."""
        base_priority = self._calculate_priority_score(task)
        
        # Performance factor based on agent's historical success rate
        agent_name = task.agent.name
        if agent_name in self.task_history:
            avg_duration = sum(self.task_history[agent_name]) / len(self.task_history[agent_name])
            performance_factor = 1.0 / (avg_duration + 0.1)
        else:
            performance_factor = 1.0
        
        # Resource utilization factor
        estimated_duration = estimated_duration or self._estimate_duration(task)
        resource_factor = 1.0 / (estimated_duration + 0.1)
        
        return base_priority * performance_factor * resource_factor
    
    def _estimate_duration(self, task: ExecutionTask) -> float:
        """Estimate task duration based on historical data."""
        agent_name = task.agent.name
        
        if agent_name in self.task_history and self.task_history[agent_name]:
            return sum(self.task_history[agent_name]) / len(self.task_history[agent_name])
        
        # Default estimate based on message count and complexity
        message_count = len(task.messages)
        return max(5.0, message_count * 2.0)  # Minimum 5 seconds, 2 seconds per message
    
    def record_execution_time(self, agent_name: str, duration: float) -> None:
        """Record execution time for adaptive scheduling."""
        if agent_name not in self.task_history:
            self.task_history[agent_name] = []
        
        self.task_history[agent_name].append(duration)
        
        # Keep only recent history (last 100 executions)
        if len(self.task_history[agent_name]) > 100:
            self.task_history[agent_name] = self.task_history[agent_name][-100:]
    
    def size(self) -> int:
        """Get the number of tasks in the queue."""
        if self.strategy == SchedulingStrategy.FIFO:
            return len(self.fifo_queue)
        elif self.strategy == SchedulingStrategy.ROUND_ROBIN:
            return sum(len(q) for q in self.round_robin_queues.values())
        else:
            return len(self.tasks)
    
    def is_empty(self) -> bool:
        """Check if the queue is empty."""
        return self.size() == 0
    
    def peek(self) -> Optional[ExecutionTask]:
        """Peek at the next task without removing it."""
        if self.strategy == SchedulingStrategy.FIFO:
            return self.fifo_queue[0] if self.fifo_queue else None
        elif self.strategy == SchedulingStrategy.ROUND_ROBIN:
            # This is more complex for round-robin, simplified implementation
            for queue in self.round_robin_queues.values():
                if queue:
                    return queue[0]
            return None
        else:
            return self.tasks[0].task if self.tasks else None


class TaskScheduler:
    """Advanced task scheduler with load balancing and optimization."""
    
    def __init__(
        self,
        strategy: SchedulingStrategy = SchedulingStrategy.ADAPTIVE,
        max_queue_size: int = 1000,
        load_balancing: bool = True
    ):
        self.strategy = strategy
        self.max_queue_size = max_queue_size
        self.load_balancing = load_balancing
        
        self.task_queue = TaskQueue(strategy)
        self.agent_loads: Dict[str, int] = {}  # Agent -> current task count
        self.agent_performance: Dict[str, Dict[str, float]] = {}  # Agent -> metrics
        
    async def schedule_task(
        self,
        task: ExecutionTask,
        estimated_duration: float = 0.0
    ) -> bool:
        """Schedule a task for execution."""
        
        if self.task_queue.size() >= self.max_queue_size:
            return False  # Queue is full
        
        # Apply load balancing if enabled
        if self.load_balancing:
            if not self._can_schedule_for_agent(task.agent.name):
                return False  # Agent is overloaded
        
        self.task_queue.enqueue(task, estimated_duration)
        return True
    
    async def get_next_task(self) -> Optional[ExecutionTask]:
        """Get the next task to execute."""
        return self.task_queue.dequeue()
    
    def _can_schedule_for_agent(self, agent_name: str) -> bool:
        """Check if we can schedule more tasks for an agent."""
        current_load = self.agent_loads.get(agent_name, 0)
        max_load_per_agent = 5  # Configurable limit
        
        return current_load < max_load_per_agent
    
    def record_task_start(self, task: ExecutionTask) -> None:
        """Record that a task has started."""
        agent_name = task.agent.name
        self.agent_loads[agent_name] = self.agent_loads.get(agent_name, 0) + 1
    
    def record_task_completion(
        self,
        task: ExecutionTask,
        duration: float,
        success: bool
    ) -> None:
        """Record task completion for performance tracking."""
        agent_name = task.agent.name
        
        # Update load tracking
        if agent_name in self.agent_loads:
            self.agent_loads[agent_name] = max(0, self.agent_loads[agent_name] - 1)
        
        # Update performance metrics
        if agent_name not in self.agent_performance:
            self.agent_performance[agent_name] = {
                "total_tasks": 0,
                "successful_tasks": 0,
                "total_duration": 0.0,
                "avg_duration": 0.0,
                "success_rate": 0.0
            }
        
        metrics = self.agent_performance[agent_name]
        metrics["total_tasks"] += 1
        if success:
            metrics["successful_tasks"] += 1
        metrics["total_duration"] += duration
        metrics["avg_duration"] = metrics["total_duration"] / metrics["total_tasks"]
        metrics["success_rate"] = metrics["successful_tasks"] / metrics["total_tasks"]
        
        # Record for adaptive scheduling
        self.task_queue.record_execution_time(agent_name, duration)
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            "strategy": self.strategy.value,
            "queue_size": self.task_queue.size(),
            "max_queue_size": self.max_queue_size,
            "agent_loads": self.agent_loads.copy(),
            "agent_performance": self.agent_performance.copy()
        }
    
    def optimize_scheduling(self) -> None:
        """Optimize scheduling based on performance data."""
        if self.strategy != SchedulingStrategy.ADAPTIVE:
            return
        
        # Analyze performance patterns and adjust scheduling
        for agent_name, metrics in self.agent_performance.items():
            if metrics["success_rate"] < 0.8:  # Low success rate
                # Could implement agent-specific optimizations here
                pass
            
            if metrics["avg_duration"] > 60.0:  # Slow agent
                # Could adjust priority calculations
                pass
