"""
Parallel execution framework for running multiple agents concurrently.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union
from concurrent.futures import ThreadPoolExecutor

from autogen_core import CancellationToken
from autogen_agentchat.base import ChatAgent, Response
from autogen_agentchat.messages import BaseChatMessage


class TaskPriority(Enum):
    """Task priority levels for execution scheduling."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ExecutionTask:
    """Task for parallel execution."""
    task_id: str
    agent: ChatAgent
    messages: List[BaseChatMessage]
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: Optional[float] = None
    dependencies: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class ExecutionResult:
    """Result of task execution."""
    task_id: str
    agent_name: str
    success: bool
    response: Optional[Response] = None
    error: Optional[Exception] = None
    execution_time: float = 0.0
    started_at: float = 0.0
    completed_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ParallelAgentExecutor:
    """Executes multiple agent tasks in parallel with dependency management."""
    
    def __init__(
        self,
        max_concurrent_tasks: int = 5,
        default_timeout: float = 300.0,  # 5 minutes
        enable_thread_pool: bool = True,
        thread_pool_size: int = 10
    ):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.default_timeout = default_timeout
        self.enable_thread_pool = enable_thread_pool
        
        # Task management
        self.pending_tasks: Dict[str, ExecutionTask] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.completed_tasks: Dict[str, ExecutionResult] = {}
        self.task_dependencies: Dict[str, Set[str]] = {}
        
        # Execution control
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self.thread_pool = ThreadPoolExecutor(max_workers=thread_pool_size) if enable_thread_pool else None
        
        # Callbacks
        self.task_started_callbacks: List[Callable[[ExecutionTask], None]] = []
        self.task_completed_callbacks: List[Callable[[ExecutionResult], None]] = []
        
    async def submit_task(
        self,
        task_id: str,
        agent: ChatAgent,
        messages: List[BaseChatMessage],
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: Optional[float] = None,
        dependencies: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Submit a task for parallel execution."""
        
        if task_id in self.pending_tasks or task_id in self.running_tasks:
            raise ValueError(f"Task with ID '{task_id}' already exists")
        
        task = ExecutionTask(
            task_id=task_id,
            agent=agent,
            messages=messages,
            priority=priority,
            timeout=timeout or self.default_timeout,
            dependencies=dependencies or set(),
            metadata=metadata or {}
        )
        
        self.pending_tasks[task_id] = task
        self.task_dependencies[task_id] = task.dependencies.copy()
        
        # Try to start the task if dependencies are met
        await self._try_start_task(task_id)
        
        return task_id
    
    async def submit_batch(
        self,
        tasks: List[Dict[str, Any]]
    ) -> List[str]:
        """Submit multiple tasks as a batch."""
        task_ids = []
        
        for task_config in tasks:
            task_id = await self.submit_task(**task_config)
            task_ids.append(task_id)
            
        return task_ids
    
    async def wait_for_task(
        self,
        task_id: str,
        timeout: Optional[float] = None
    ) -> ExecutionResult:
        """Wait for a specific task to complete."""
        
        # Check if already completed
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        
        # Check if currently running
        if task_id in self.running_tasks:
            try:
                await asyncio.wait_for(
                    self.running_tasks[task_id],
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                await self.cancel_task(task_id)
                raise
            
            return self.completed_tasks[task_id]
        
        # Task not found or not started
        if task_id not in self.pending_tasks:
            raise ValueError(f"Task '{task_id}' not found")
        
        # Wait for task to start and complete
        start_time = time.time()
        while task_id not in self.completed_tasks:
            if timeout and (time.time() - start_time) > timeout:
                raise asyncio.TimeoutError(f"Timeout waiting for task '{task_id}'")
            
            await asyncio.sleep(0.1)
        
        return self.completed_tasks[task_id]
    
    async def wait_for_all(
        self,
        task_ids: Optional[List[str]] = None,
        timeout: Optional[float] = None
    ) -> Dict[str, ExecutionResult]:
        """Wait for all specified tasks (or all tasks) to complete."""
        
        if task_ids is None:
            task_ids = list(self.pending_tasks.keys()) + list(self.running_tasks.keys())
        
        results = {}
        
        # Wait for all tasks
        for task_id in task_ids:
            try:
                result = await self.wait_for_task(task_id, timeout)
                results[task_id] = result
            except Exception as e:
                # Create error result for failed tasks
                results[task_id] = ExecutionResult(
                    task_id=task_id,
                    agent_name="unknown",
                    success=False,
                    error=e,
                    completed_at=time.time()
                )
        
        return results
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task."""
        
        # Cancel running task
        if task_id in self.running_tasks:
            self.running_tasks[task_id].cancel()
            try:
                await self.running_tasks[task_id]
            except asyncio.CancelledError:
                pass
            
            # Create cancelled result
            self.completed_tasks[task_id] = ExecutionResult(
                task_id=task_id,
                agent_name=self.pending_tasks.get(task_id, ExecutionTask("", None, [])).agent.name if task_id in self.pending_tasks else "unknown",
                success=False,
                error=asyncio.CancelledError("Task was cancelled"),
                completed_at=time.time()
            )
            
            return True
        
        # Remove pending task
        if task_id in self.pending_tasks:
            del self.pending_tasks[task_id]
            self.task_dependencies.pop(task_id, None)
            return True
        
        return False
    
    async def _try_start_task(self, task_id: str) -> bool:
        """Try to start a task if its dependencies are met."""
        
        if task_id not in self.pending_tasks:
            return False
        
        task = self.pending_tasks[task_id]
        
        # Check if dependencies are satisfied
        unsatisfied_deps = self.task_dependencies[task_id] - set(self.completed_tasks.keys())
        if unsatisfied_deps:
            return False
        
        # Check if any dependencies failed
        for dep_id in task.dependencies:
            if dep_id in self.completed_tasks and not self.completed_tasks[dep_id].success:
                # Dependency failed, mark this task as failed too
                self.completed_tasks[task_id] = ExecutionResult(
                    task_id=task_id,
                    agent_name=task.agent.name,
                    success=False,
                    error=Exception(f"Dependency '{dep_id}' failed"),
                    completed_at=time.time()
                )
                del self.pending_tasks[task_id]
                self.task_dependencies.pop(task_id, None)
                return False
        
        # Start the task
        execution_task = asyncio.create_task(self._execute_task(task))
        self.running_tasks[task_id] = execution_task
        del self.pending_tasks[task_id]
        
        # Notify callbacks
        for callback in self.task_started_callbacks:
            try:
                callback(task)
            except Exception:
                pass  # Don't let callback errors break execution
        
        return True
    
    async def _execute_task(self, task: ExecutionTask) -> None:
        """Execute a single task."""
        
        async with self.semaphore:  # Limit concurrent executions
            start_time = time.time()
            
            try:
                # Execute the agent task
                if self.enable_thread_pool and self.thread_pool:
                    # Run in thread pool for CPU-bound operations
                    response = await asyncio.get_event_loop().run_in_executor(
                        self.thread_pool,
                        self._sync_agent_execution,
                        task
                    )
                else:
                    # Run directly in async context
                    response = await task.agent.on_messages(
                        task.messages,
                        CancellationToken()
                    )
                
                execution_time = time.time() - start_time
                
                # Create success result
                result = ExecutionResult(
                    task_id=task.task_id,
                    agent_name=task.agent.name,
                    success=True,
                    response=response,
                    execution_time=execution_time,
                    started_at=start_time,
                    completed_at=time.time(),
                    metadata=task.metadata.copy()
                )
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Create error result
                result = ExecutionResult(
                    task_id=task.task_id,
                    agent_name=task.agent.name,
                    success=False,
                    error=e,
                    execution_time=execution_time,
                    started_at=start_time,
                    completed_at=time.time(),
                    metadata=task.metadata.copy()
                )
            
            # Store result and cleanup
            self.completed_tasks[task.task_id] = result
            self.running_tasks.pop(task.task_id, None)
            self.task_dependencies.pop(task.task_id, None)
            
            # Notify callbacks
            for callback in self.task_completed_callbacks:
                try:
                    callback(result)
                except Exception:
                    pass
            
            # Try to start dependent tasks
            await self._start_dependent_tasks(task.task_id)
    
    def _sync_agent_execution(self, task: ExecutionTask) -> Response:
        """Synchronous wrapper for agent execution in thread pool."""
        # This would need to be adapted based on the actual agent interface
        # For now, this is a placeholder
        return Response(chat_message=task.messages[-1] if task.messages else None)
    
    async def _start_dependent_tasks(self, completed_task_id: str) -> None:
        """Start tasks that were waiting for the completed task."""
        
        tasks_to_check = []
        for task_id, dependencies in self.task_dependencies.items():
            if completed_task_id in dependencies and task_id in self.pending_tasks:
                tasks_to_check.append(task_id)
        
        for task_id in tasks_to_check:
            await self._try_start_task(task_id)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current execution status."""
        return {
            "pending_tasks": len(self.pending_tasks),
            "running_tasks": len(self.running_tasks),
            "completed_tasks": len(self.completed_tasks),
            "max_concurrent": self.max_concurrent_tasks,
            "current_concurrent": len(self.running_tasks),
            "success_rate": self._calculate_success_rate()
        }
    
    def _calculate_success_rate(self) -> float:
        """Calculate success rate of completed tasks."""
        if not self.completed_tasks:
            return 0.0
        
        successful = sum(1 for result in self.completed_tasks.values() if result.success)
        return successful / len(self.completed_tasks)
    
    def add_task_started_callback(self, callback: Callable[[ExecutionTask], None]) -> None:
        """Add callback for when tasks start."""
        self.task_started_callbacks.append(callback)
    
    def add_task_completed_callback(self, callback: Callable[[ExecutionResult], None]) -> None:
        """Add callback for when tasks complete."""
        self.task_completed_callbacks.append(callback)
    
    async def shutdown(self) -> None:
        """Shutdown the executor and cleanup resources."""
        
        # Cancel all running tasks
        for task_id in list(self.running_tasks.keys()):
            await self.cancel_task(task_id)
        
        # Clear pending tasks
        self.pending_tasks.clear()
        self.task_dependencies.clear()
        
        # Shutdown thread pool
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
