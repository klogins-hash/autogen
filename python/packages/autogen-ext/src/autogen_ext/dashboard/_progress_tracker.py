"""
Progress tracking system for monitoring task execution.
"""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable
import uuid


class ProgressStatus(Enum):
    """Status of task progress."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskProgress:
    """Represents progress of a task."""
    task_id: str
    name: str
    progress: float = 0.0  # 0-100
    status: ProgressStatus = ProgressStatus.PENDING
    agent_id: Optional[str] = None
    parent_task_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    estimated_duration: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    subtasks: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.status == ProgressStatus.RUNNING and not self.started_at:
            self.started_at = time.time()
        elif self.status in [ProgressStatus.COMPLETED, ProgressStatus.FAILED] and not self.completed_at:
            self.completed_at = time.time()


@dataclass
class TaskMetrics:
    """Metrics for task execution."""
    total_tasks: int = 0
    active_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    cancelled_tasks: int = 0
    average_duration: float = 0.0
    success_rate: float = 0.0
    throughput_per_hour: float = 0.0


class ProgressTracker:
    """Tracks progress of tasks and agents."""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        
        # Task storage
        self.tasks: Dict[str, TaskProgress] = {}
        self.task_history: deque = deque(maxlen=max_history)
        
        # Agent tracking
        self.agent_tasks: Dict[str, Set[str]] = defaultdict(set)
        self.agent_status: Dict[str, Dict[str, Any]] = {}
        
        # Task relationships
        self.task_dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.task_children: Dict[str, Set[str]] = defaultdict(set)
        
        # Callbacks
        self.task_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Background monitoring
        self.monitor_task: Optional[asyncio.Task] = None
        self.start_monitoring()
    
    def start_monitoring(self):
        """Start background monitoring task."""
        
        async def monitor_loop():
            while True:
                try:
                    await asyncio.sleep(5)  # Check every 5 seconds
                    await self._check_task_timeouts()
                    await self._update_agent_metrics()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    print(f"Error in progress monitor: {e}")
        
        self.monitor_task = asyncio.create_task(monitor_loop())
    
    async def create_task(
        self,
        name: str,
        agent_id: Optional[str] = None,
        parent_task_id: Optional[str] = None,
        estimated_duration: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new task."""
        
        task_id = str(uuid.uuid4())
        
        task = TaskProgress(
            task_id=task_id,
            name=name,
            agent_id=agent_id,
            parent_task_id=parent_task_id,
            estimated_duration=estimated_duration,
            metadata=metadata or {}
        )
        
        self.tasks[task_id] = task
        
        # Update agent tracking
        if agent_id:
            self.agent_tasks[agent_id].add(task_id)
        
        # Update parent-child relationships
        if parent_task_id and parent_task_id in self.tasks:
            self.tasks[parent_task_id].subtasks.append(task_id)
            self.task_children[parent_task_id].add(task_id)
        
        # Trigger callbacks
        await self._trigger_callbacks("task_created", task)
        
        return task_id
    
    async def update_task_progress(self, task_progress: TaskProgress) -> None:
        """Update task progress."""
        
        task_id = task_progress.task_id
        
        if task_id not in self.tasks:
            # Create new task if it doesn't exist
            self.tasks[task_id] = task_progress
        else:
            # Update existing task
            existing_task = self.tasks[task_id]
            
            # Update fields
            existing_task.progress = task_progress.progress
            existing_task.status = task_progress.status
            existing_task.updated_at = time.time()
            existing_task.metadata.update(task_progress.metadata)
            
            if task_progress.error_message:
                existing_task.error_message = task_progress.error_message
            
            # Update status-specific timestamps
            if task_progress.status == ProgressStatus.RUNNING and not existing_task.started_at:
                existing_task.started_at = time.time()
            elif task_progress.status in [ProgressStatus.COMPLETED, ProgressStatus.FAILED, ProgressStatus.CANCELLED]:
                if not existing_task.completed_at:
                    existing_task.completed_at = time.time()
                
                # Move to history
                self.task_history.append(existing_task)
        
        # Update agent tracking
        if task_progress.agent_id:
            self.agent_tasks[task_progress.agent_id].add(task_id)
        
        # Trigger callbacks
        await self._trigger_callbacks("task_updated", self.tasks[task_id])
        
        # Check if task is completed
        if task_progress.status in [ProgressStatus.COMPLETED, ProgressStatus.FAILED, ProgressStatus.CANCELLED]:
            await self._handle_task_completion(task_id)
    
    async def start_task(self, task_id: str) -> bool:
        """Start a task."""
        
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        # Check dependencies
        if not await self._check_dependencies(task_id):
            return False
        
        task.status = ProgressStatus.RUNNING
        task.started_at = time.time()
        task.updated_at = time.time()
        
        await self._trigger_callbacks("task_started", task)
        
        return True
    
    async def complete_task(self, task_id: str, success: bool = True, error_message: Optional[str] = None) -> bool:
        """Complete a task."""
        
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        task.status = ProgressStatus.COMPLETED if success else ProgressStatus.FAILED
        task.progress = 100.0 if success else task.progress
        task.completed_at = time.time()
        task.updated_at = time.time()
        
        if error_message:
            task.error_message = error_message
        
        # Move to history
        self.task_history.append(task)
        
        await self._trigger_callbacks("task_completed", task)
        await self._handle_task_completion(task_id)
        
        return True
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        task.status = ProgressStatus.CANCELLED
        task.completed_at = time.time()
        task.updated_at = time.time()
        
        # Cancel subtasks
        for subtask_id in task.subtasks:
            await self.cancel_task(subtask_id)
        
        # Move to history
        self.task_history.append(task)
        
        await self._trigger_callbacks("task_cancelled", task)
        await self._handle_task_completion(task_id)
        
        return True
    
    async def pause_task(self, task_id: str) -> bool:
        """Pause a task."""
        
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        if task.status == ProgressStatus.RUNNING:
            task.status = ProgressStatus.PAUSED
            task.updated_at = time.time()
            
            await self._trigger_callbacks("task_paused", task)
            return True
        
        return False
    
    async def resume_task(self, task_id: str) -> bool:
        """Resume a paused task."""
        
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        if task.status == ProgressStatus.PAUSED:
            task.status = ProgressStatus.RUNNING
            task.updated_at = time.time()
            
            await self._trigger_callbacks("task_resumed", task)
            return True
        
        return False
    
    def get_task(self, task_id: str) -> Optional[TaskProgress]:
        """Get a task by ID."""
        return self.tasks.get(task_id)
    
    def get_active_tasks(self) -> List[TaskProgress]:
        """Get all active tasks."""
        return [
            task for task in self.tasks.values()
            if task.status in [ProgressStatus.PENDING, ProgressStatus.RUNNING, ProgressStatus.PAUSED]
        ]
    
    def get_completed_tasks(self, limit: Optional[int] = None) -> List[TaskProgress]:
        """Get completed tasks from history."""
        
        completed = [
            task for task in self.task_history
            if task.status in [ProgressStatus.COMPLETED, ProgressStatus.FAILED, ProgressStatus.CANCELLED]
        ]
        
        # Sort by completion time (most recent first)
        completed.sort(key=lambda t: t.completed_at or 0, reverse=True)
        
        if limit:
            completed = completed[:limit]
        
        return completed
    
    def get_task_history(self, limit: Optional[int] = None) -> List[TaskProgress]:
        """Get task history."""
        
        history = list(self.task_history)
        history.sort(key=lambda t: t.updated_at, reverse=True)
        
        if limit:
            history = history[:limit]
        
        return history
    
    def get_agent_tasks(self, agent_id: str) -> List[TaskProgress]:
        """Get tasks for a specific agent."""
        
        task_ids = self.agent_tasks.get(agent_id, set())
        return [self.tasks[task_id] for task_id in task_ids if task_id in self.tasks]
    
    def get_task_tree(self, root_task_id: str) -> Dict[str, Any]:
        """Get task tree starting from a root task."""
        
        if root_task_id not in self.tasks:
            return {}
        
        def build_tree(task_id: str) -> Dict[str, Any]:
            task = self.tasks[task_id]
            
            tree = {
                "task": task,
                "children": []
            }
            
            for child_id in task.subtasks:
                if child_id in self.tasks:
                    tree["children"].append(build_tree(child_id))
            
            return tree
        
        return build_tree(root_task_id)
    
    def get_task_metrics(self) -> TaskMetrics:
        """Get task execution metrics."""
        
        active_tasks = self.get_active_tasks()
        completed_tasks = self.get_completed_tasks()
        
        total_tasks = len(self.tasks) + len(self.task_history)
        
        # Calculate success rate
        successful_tasks = len([t for t in completed_tasks if t.status == ProgressStatus.COMPLETED])
        failed_tasks = len([t for t in completed_tasks if t.status == ProgressStatus.FAILED])
        cancelled_tasks = len([t for t in completed_tasks if t.status == ProgressStatus.CANCELLED])
        
        success_rate = 0.0
        if completed_tasks:
            success_rate = (successful_tasks / len(completed_tasks)) * 100
        
        # Calculate average duration
        durations = []
        for task in completed_tasks:
            if task.started_at and task.completed_at:
                durations.append(task.completed_at - task.started_at)
        
        average_duration = sum(durations) / len(durations) if durations else 0.0
        
        # Calculate throughput (tasks per hour)
        throughput_per_hour = 0.0
        if completed_tasks:
            # Get tasks completed in the last hour
            one_hour_ago = time.time() - 3600
            recent_completions = [
                t for t in completed_tasks
                if t.completed_at and t.completed_at >= one_hour_ago
            ]
            throughput_per_hour = len(recent_completions)
        
        return TaskMetrics(
            total_tasks=total_tasks,
            active_tasks=len(active_tasks),
            completed_tasks=successful_tasks,
            failed_tasks=failed_tasks,
            cancelled_tasks=cancelled_tasks,
            average_duration=average_duration,
            success_rate=success_rate,
            throughput_per_hour=throughput_per_hour
        )
    
    def get_agent_metrics(self, agent_id: str) -> Dict[str, Any]:
        """Get metrics for a specific agent."""
        
        agent_tasks = self.get_agent_tasks(agent_id)
        
        active = [t for t in agent_tasks if t.status in [ProgressStatus.PENDING, ProgressStatus.RUNNING, ProgressStatus.PAUSED]]
        completed = [t for t in agent_tasks if t.status == ProgressStatus.COMPLETED]
        failed = [t for t in agent_tasks if t.status == ProgressStatus.FAILED]
        
        # Calculate average progress
        total_progress = sum(t.progress for t in agent_tasks)
        average_progress = total_progress / len(agent_tasks) if agent_tasks else 0.0
        
        return {
            "agent_id": agent_id,
            "total_tasks": len(agent_tasks),
            "active_tasks": len(active),
            "completed_tasks": len(completed),
            "failed_tasks": len(failed),
            "average_progress": average_progress,
            "success_rate": (len(completed) / len(agent_tasks) * 100) if agent_tasks else 0.0
        }
    
    async def add_task_dependency(self, task_id: str, dependency_id: str) -> bool:
        """Add a dependency between tasks."""
        
        if task_id not in self.tasks or dependency_id not in self.tasks:
            return False
        
        self.task_dependencies[task_id].add(dependency_id)
        self.tasks[task_id].dependencies.append(dependency_id)
        
        return True
    
    async def _check_dependencies(self, task_id: str) -> bool:
        """Check if task dependencies are satisfied."""
        
        dependencies = self.task_dependencies.get(task_id, set())
        
        for dep_id in dependencies:
            if dep_id in self.tasks:
                dep_task = self.tasks[dep_id]
                if dep_task.status != ProgressStatus.COMPLETED:
                    return False
        
        return True
    
    async def _handle_task_completion(self, task_id: str) -> None:
        """Handle task completion and check dependent tasks."""
        
        # Check if any pending tasks can now start
        for pending_task_id, task in self.tasks.items():
            if (task.status == ProgressStatus.PENDING and 
                task_id in self.task_dependencies.get(pending_task_id, set())):
                
                if await self._check_dependencies(pending_task_id):
                    await self.start_task(pending_task_id)
        
        # Update parent task progress if this is a subtask
        task = self.tasks.get(task_id)
        if task and task.parent_task_id:
            await self._update_parent_progress(task.parent_task_id)
    
    async def _update_parent_progress(self, parent_task_id: str) -> None:
        """Update parent task progress based on subtasks."""
        
        if parent_task_id not in self.tasks:
            return
        
        parent_task = self.tasks[parent_task_id]
        subtask_ids = parent_task.subtasks
        
        if not subtask_ids:
            return
        
        # Calculate average progress of subtasks
        total_progress = 0.0
        completed_subtasks = 0
        
        for subtask_id in subtask_ids:
            if subtask_id in self.tasks:
                subtask = self.tasks[subtask_id]
                total_progress += subtask.progress
                
                if subtask.status == ProgressStatus.COMPLETED:
                    completed_subtasks += 1
        
        # Update parent progress
        parent_task.progress = total_progress / len(subtask_ids)
        parent_task.updated_at = time.time()
        
        # Check if all subtasks are completed
        if completed_subtasks == len(subtask_ids):
            parent_task.status = ProgressStatus.COMPLETED
            parent_task.completed_at = time.time()
            parent_task.progress = 100.0
            
            # Move to history
            self.task_history.append(parent_task)
            
            await self._trigger_callbacks("task_completed", parent_task)
    
    async def _check_task_timeouts(self) -> None:
        """Check for task timeouts."""
        
        current_time = time.time()
        
        for task in self.tasks.values():
            if (task.status == ProgressStatus.RUNNING and 
                task.estimated_duration and 
                task.started_at):
                
                elapsed = current_time - task.started_at
                
                if elapsed > task.estimated_duration * 1.5:  # 50% overtime
                    # Mark as potentially stuck
                    task.metadata["timeout_warning"] = True
                    await self._trigger_callbacks("task_timeout", task)
    
    async def _update_agent_metrics(self) -> None:
        """Update agent metrics."""
        
        for agent_id in self.agent_tasks:
            metrics = self.get_agent_metrics(agent_id)
            self.agent_status[agent_id] = {
                "last_updated": time.time(),
                "metrics": metrics
            }
    
    def register_callback(self, event: str, callback: Callable) -> None:
        """Register a callback for task events."""
        
        self.task_callbacks[event].append(callback)
    
    async def _trigger_callbacks(self, event: str, task: TaskProgress) -> None:
        """Trigger callbacks for an event."""
        
        callbacks = self.task_callbacks.get(event, [])
        
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(task)
                else:
                    callback(task)
            except Exception as e:
                print(f"Error in callback {callback}: {e}")
    
    async def shutdown(self) -> None:
        """Shutdown the progress tracker."""
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
    
    def export_data(self) -> Dict[str, Any]:
        """Export all tracking data."""
        
        return {
            "active_tasks": [
                {
                    "task_id": t.task_id,
                    "name": t.name,
                    "progress": t.progress,
                    "status": t.status.value,
                    "agent_id": t.agent_id,
                    "created_at": t.created_at,
                    "updated_at": t.updated_at,
                    "started_at": t.started_at,
                    "metadata": t.metadata
                }
                for t in self.get_active_tasks()
            ],
            "task_history": [
                {
                    "task_id": t.task_id,
                    "name": t.name,
                    "progress": t.progress,
                    "status": t.status.value,
                    "agent_id": t.agent_id,
                    "created_at": t.created_at,
                    "completed_at": t.completed_at,
                    "metadata": t.metadata
                }
                for t in self.get_task_history()
            ],
            "metrics": self.get_task_metrics().__dict__,
            "agent_metrics": {
                agent_id: self.get_agent_metrics(agent_id)
                for agent_id in self.agent_tasks
            }
        }
