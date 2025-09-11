"""
Example usage of the dashboard system with AutoGen enhanced agents.
"""

import asyncio
import time
from typing import Dict, Any, Optional

from autogen_agentchat import ConversableAgent
from autogen_core import CancellationToken

from ._dashboard_integration import DashboardIntegration, AgentDashboardConfig, dashboard_enabled
from ._dashboard_server import DashboardConfig
from ._metrics_collector import MetricType
from ._progress_tracker import ProgressStatus


# Example of using the dashboard decorator
@dashboard_enabled(AgentDashboardConfig(
    enable_metrics=True,
    enable_progress_tracking=True,
    enable_logging=True,
    auto_start_dashboard=False  # We'll start it manually in this example
))
class DashboardEnabledAgent(ConversableAgent):
    """Example agent with dashboard integration."""
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, **kwargs)
        self.task_counter = 0
    
    async def process_task(self, task_description: str) -> str:
        """Process a task with dashboard tracking."""
        
        self.task_counter += 1
        task_id = f"task_{self.task_counter}"
        
        # Track task start
        self.track_task(
            task_id=task_id,
            task_name=task_description,
            progress=0.0,
            status=ProgressStatus.RUNNING
        )
        
        # Track message
        self.track_message(
            message_type="task_start",
            content=f"Starting task: {task_description}"
        )
        
        # Simulate task processing with progress updates
        for i in range(5):
            await asyncio.sleep(1)  # Simulate work
            
            progress = (i + 1) * 20  # 20%, 40%, 60%, 80%, 100%
            
            self.track_task(
                task_id=task_id,
                task_name=task_description,
                progress=progress,
                status=ProgressStatus.RUNNING
            )
            
            # Record some metrics
            self.record_metric(f"processing_step_{i}", i + 1, MetricType.COUNTER)
        
        # Complete the task
        self.track_task(
            task_id=task_id,
            task_name=task_description,
            progress=100.0,
            status=ProgressStatus.COMPLETED
        )
        
        self.track_message(
            message_type="task_complete",
            content=f"Completed task: {task_description}"
        )
        
        return f"Task '{task_description}' completed successfully"


class DashboardExample:
    """Example demonstrating dashboard usage with AutoGen agents."""
    
    def __init__(self):
        # Dashboard configuration
        dashboard_config = DashboardConfig(
            host="localhost",
            port=8080,
            debug=True
        )
        
        agent_config = AgentDashboardConfig(
            enable_metrics=True,
            enable_progress_tracking=True,
            enable_logging=True,
            auto_start_dashboard=False,
            dashboard_config=dashboard_config
        )
        
        # Create dashboard integration
        self.dashboard = DashboardIntegration(agent_config)
        
        # Create agents
        self.agents = []
    
    async def setup(self):
        """Setup the dashboard and agents."""
        
        # Initialize dashboard
        await self.dashboard.initialize()
        
        # Create some example agents
        agent1 = DashboardEnabledAgent(
            name="WebSearchAgent",
            system_message="You are a web search agent that helps find information online."
        )
        
        agent2 = DashboardEnabledAgent(
            name="DataAnalysisAgent", 
            system_message="You are a data analysis agent that processes and analyzes data."
        )
        
        agent3 = DashboardEnabledAgent(
            name="ReportGeneratorAgent",
            system_message="You are a report generator that creates comprehensive reports."
        )
        
        self.agents = [agent1, agent2, agent3]
        
        # Register agents with dashboard
        for i, agent in enumerate(self.agents):
            await self.dashboard.register_agent(
                agent_id=f"agent_{i}",
                agent_name=agent.name,
                agent_type="ConversableAgent",
                metadata={"system_message": agent.system_message}
            )
    
    async def run_example_workflow(self):
        """Run an example workflow with dashboard tracking."""
        
        print("Starting example workflow...")
        
        # Start dashboard server in background
        dashboard_task = asyncio.create_task(self.dashboard.start_dashboard())
        
        # Wait a moment for server to start
        await asyncio.sleep(2)
        
        print(f"Dashboard available at: {self.dashboard.get_dashboard_url()}")
        
        # Run some example tasks
        tasks = [
            "Search for recent AI developments",
            "Analyze market trends data", 
            "Generate quarterly report",
            "Process customer feedback",
            "Create data visualization"
        ]
        
        # Run tasks concurrently across agents
        task_futures = []
        
        for i, task in enumerate(tasks):
            agent = self.agents[i % len(self.agents)]
            future = asyncio.create_task(agent.process_task(task))
            task_futures.append(future)
            
            # Stagger task starts
            await asyncio.sleep(0.5)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*task_futures)
        
        print("All tasks completed:")
        for i, result in enumerate(results):
            print(f"  {i+1}. {result}")
        
        # Record some final metrics
        await self.dashboard.record_metric("workflow_completed", 1, MetricType.COUNTER)
        await self.dashboard.record_metric("total_tasks", len(tasks), MetricType.GAUGE)
        
        # Keep dashboard running for a bit to see the results
        print("\nDashboard will continue running for 30 seconds...")
        print("You can view the dashboard in your browser to see the metrics and logs.")
        
        await asyncio.sleep(30)
        
        # Stop dashboard
        await self.dashboard.stop_dashboard()
        dashboard_task.cancel()
        
        try:
            await dashboard_task
        except asyncio.CancelledError:
            pass
    
    async def run_interactive_demo(self):
        """Run an interactive demo where users can see real-time updates."""
        
        print("Starting interactive dashboard demo...")
        
        # Start dashboard
        dashboard_task = asyncio.create_task(self.dashboard.start_dashboard())
        await asyncio.sleep(2)
        
        print(f"\n🚀 Dashboard is now running at: {self.dashboard.get_dashboard_url()}")
        print("\nOpen the URL in your browser to see real-time updates!")
        print("\nThe demo will run various agent activities to demonstrate the dashboard features:")
        print("- Agent connections and disconnections")
        print("- Task progress tracking")
        print("- Metrics collection")
        print("- Activity logging")
        print("\nPress Ctrl+C to stop the demo.\n")
        
        try:
            # Simulate ongoing agent activity
            while True:
                # Randomly select an agent and task
                import random
                
                agent = random.choice(self.agents)
                tasks = [
                    "Process incoming data batch",
                    "Generate analysis report", 
                    "Search for relevant information",
                    "Validate data quality",
                    "Create summary document",
                    "Perform system check",
                    "Update knowledge base",
                    "Generate recommendations"
                ]
                
                task = random.choice(tasks)
                
                # Run the task
                await agent.process_task(task)
                
                # Record some random metrics
                await self.dashboard.record_metric(
                    "cpu_usage", 
                    random.uniform(20, 80), 
                    MetricType.GAUGE
                )
                
                await self.dashboard.record_metric(
                    "memory_usage",
                    random.uniform(40, 90),
                    MetricType.GAUGE
                )
                
                await self.dashboard.record_metric(
                    "requests_processed",
                    random.randint(1, 5),
                    MetricType.COUNTER
                )
                
                # Wait before next activity
                await asyncio.sleep(random.uniform(2, 8))
        
        except KeyboardInterrupt:
            print("\n\nStopping dashboard demo...")
        
        finally:
            # Clean shutdown
            await self.dashboard.stop_dashboard()
            dashboard_task.cancel()
            
            try:
                await dashboard_task
            except asyncio.CancelledError:
                pass
            
            print("Dashboard demo stopped.")


async def run_basic_example():
    """Run a basic dashboard example."""
    
    example = DashboardExample()
    await example.setup()
    await example.run_example_workflow()


async def run_interactive_example():
    """Run an interactive dashboard example."""
    
    example = DashboardExample()
    await example.setup()
    await example.run_interactive_demo()


def main():
    """Main function to run dashboard examples."""
    
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        print("Running interactive dashboard demo...")
        asyncio.run(run_interactive_example())
    else:
        print("Running basic dashboard example...")
        asyncio.run(run_basic_example())


if __name__ == "__main__":
    main()
