"""
Token optimization and usage tracking utilities.
"""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from autogen_core.models import ChatCompletionClient, LLMMessage


@dataclass
class TokenUsage:
    """Token usage statistics."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_estimate: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class OptimizationMetrics:
    """Metrics for token optimization performance."""
    original_tokens: int
    optimized_tokens: int
    savings: int
    savings_percentage: float
    optimization_time: float


class TokenUsageTracker:
    """Tracks token usage across conversations and agents."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.usage_history: deque = deque(maxlen=max_history)
        self.agent_usage: Dict[str, List[TokenUsage]] = defaultdict(list)
        self.session_usage: Dict[str, TokenUsage] = {}
        
    def record_usage(
        self, 
        agent_name: str, 
        session_id: str, 
        usage: TokenUsage
    ) -> None:
        """Record token usage for an agent in a session."""
        self.usage_history.append((agent_name, session_id, usage))
        self.agent_usage[agent_name].append(usage)
        
        if session_id not in self.session_usage:
            self.session_usage[session_id] = TokenUsage()
        
        session_usage = self.session_usage[session_id]
        session_usage.prompt_tokens += usage.prompt_tokens
        session_usage.completion_tokens += usage.completion_tokens
        session_usage.total_tokens += usage.total_tokens
        session_usage.cost_estimate += usage.cost_estimate
    
    def get_agent_stats(self, agent_name: str) -> Dict[str, float]:
        """Get usage statistics for a specific agent."""
        if agent_name not in self.agent_usage:
            return {}
        
        usages = self.agent_usage[agent_name]
        total_tokens = sum(u.total_tokens for u in usages)
        total_cost = sum(u.cost_estimate for u in usages)
        avg_tokens = total_tokens / len(usages) if usages else 0
        
        return {
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "average_tokens_per_request": avg_tokens,
            "request_count": len(usages)
        }
    
    def get_session_stats(self, session_id: str) -> Optional[TokenUsage]:
        """Get usage statistics for a specific session."""
        return self.session_usage.get(session_id)
    
    def get_recent_usage(self, minutes: int = 60) -> List[Tuple[str, str, TokenUsage]]:
        """Get usage from the last N minutes."""
        cutoff_time = time.time() - (minutes * 60)
        return [
            (agent, session, usage) for agent, session, usage in self.usage_history
            if usage.timestamp >= cutoff_time
        ]


class TokenOptimizer:
    """Optimizes token usage through various strategies."""
    
    def __init__(self, model_client: ChatCompletionClient):
        self.model_client = model_client
        self.usage_tracker = TokenUsageTracker()
        
        # Model-specific token costs (per 1K tokens)
        self.token_costs = {
            "gpt-4": {"prompt": 0.03, "completion": 0.06},
            "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
            "gpt-3.5-turbo": {"prompt": 0.0015, "completion": 0.002},
            "claude-3": {"prompt": 0.015, "completion": 0.075},
        }
    
    def estimate_cost(self, prompt_tokens: int, completion_tokens: int, model: str) -> float:
        """Estimate cost based on token usage and model."""
        model_key = self._normalize_model_name(model)
        costs = self.token_costs.get(model_key, {"prompt": 0.01, "completion": 0.03})
        
        prompt_cost = (prompt_tokens / 1000) * costs["prompt"]
        completion_cost = (completion_tokens / 1000) * costs["completion"]
        
        return prompt_cost + completion_cost
    
    def _normalize_model_name(self, model: str) -> str:
        """Normalize model name for cost lookup."""
        model_lower = model.lower()
        if "gpt-4-turbo" in model_lower:
            return "gpt-4-turbo"
        elif "gpt-4" in model_lower:
            return "gpt-4"
        elif "gpt-3.5" in model_lower:
            return "gpt-3.5-turbo"
        elif "claude" in model_lower:
            return "claude-3"
        return "gpt-4"  # Default fallback
    
    async def optimize_messages(
        self, 
        messages: List[LLMMessage],
        target_token_limit: Optional[int] = None
    ) -> Tuple[List[LLMMessage], OptimizationMetrics]:
        """Optimize messages to reduce token usage."""
        start_time = time.time()
        original_tokens = self._estimate_tokens(messages)
        
        if target_token_limit is None:
            # Use model's context window with safety margin
            context_limit = self._get_context_limit()
            target_token_limit = int(context_limit * 0.8)  # 80% of context window
        
        optimized_messages = messages
        
        if original_tokens > target_token_limit:
            # Apply optimization strategies
            optimized_messages = await self._apply_optimizations(
                messages, target_token_limit
            )
        
        optimized_tokens = self._estimate_tokens(optimized_messages)
        optimization_time = time.time() - start_time
        
        savings = original_tokens - optimized_tokens
        savings_percentage = (savings / original_tokens * 100) if original_tokens > 0 else 0
        
        metrics = OptimizationMetrics(
            original_tokens=original_tokens,
            optimized_tokens=optimized_tokens,
            savings=savings,
            savings_percentage=savings_percentage,
            optimization_time=optimization_time
        )
        
        return optimized_messages, metrics
    
    async def _apply_optimizations(
        self, 
        messages: List[LLMMessage], 
        target_limit: int
    ) -> List[LLMMessage]:
        """Apply various optimization strategies."""
        current_messages = messages
        current_tokens = self._estimate_tokens(current_messages)
        
        # Strategy 1: Remove redundant whitespace and formatting
        if current_tokens > target_limit:
            current_messages = self._clean_message_content(current_messages)
            current_tokens = self._estimate_tokens(current_messages)
        
        # Strategy 2: Truncate very long individual messages
        if current_tokens > target_limit:
            current_messages = self._truncate_long_messages(current_messages, target_limit)
            current_tokens = self._estimate_tokens(current_messages)
        
        # Strategy 3: Remove older messages (sliding window)
        if current_tokens > target_limit:
            current_messages = self._apply_sliding_window(current_messages, target_limit)
        
        return current_messages
    
    def _clean_message_content(self, messages: List[LLMMessage]) -> List[LLMMessage]:
        """Clean message content to reduce tokens."""
        cleaned_messages = []
        
        for msg in messages:
            if isinstance(msg.content, str):
                # Remove extra whitespace
                cleaned_content = " ".join(msg.content.split())
                # Remove common redundant phrases
                cleaned_content = self._remove_redundant_phrases(cleaned_content)
                
                # Create new message with cleaned content
                if hasattr(msg, 'source'):
                    new_msg = type(msg)(content=cleaned_content, source=msg.source)
                else:
                    new_msg = type(msg)(content=cleaned_content)
                cleaned_messages.append(new_msg)
            else:
                cleaned_messages.append(msg)
        
        return cleaned_messages
    
    def _remove_redundant_phrases(self, content: str) -> str:
        """Remove common redundant phrases to save tokens."""
        redundant_patterns = [
            r'\b(please|kindly|if you would|if you could)\b',
            r'\b(thank you|thanks)\b',
            r'\b(I think|I believe|in my opinion)\b',
            r'\b(actually|basically|essentially)\b',
        ]
        
        cleaned = content
        for pattern in redundant_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Clean up extra spaces
        cleaned = " ".join(cleaned.split())
        return cleaned
    
    def _truncate_long_messages(
        self, 
        messages: List[LLMMessage], 
        target_limit: int
    ) -> List[LLMMessage]:
        """Truncate individual messages that are too long."""
        max_message_tokens = target_limit // max(len(messages), 1)
        truncated_messages = []
        
        for msg in messages:
            if isinstance(msg.content, str):
                msg_tokens = self._estimate_tokens([msg])
                if msg_tokens > max_message_tokens:
                    # Truncate to fit within limit
                    target_chars = max_message_tokens * 4  # Rough estimation
                    truncated_content = msg.content[:target_chars] + "... [truncated]"
                    
                    if hasattr(msg, 'source'):
                        new_msg = type(msg)(content=truncated_content, source=msg.source)
                    else:
                        new_msg = type(msg)(content=truncated_content)
                    truncated_messages.append(new_msg)
                else:
                    truncated_messages.append(msg)
            else:
                truncated_messages.append(msg)
        
        return truncated_messages
    
    def _apply_sliding_window(
        self, 
        messages: List[LLMMessage], 
        target_limit: int
    ) -> List[LLMMessage]:
        """Apply sliding window to keep most recent messages."""
        # Always keep system messages
        system_messages = [msg for msg in messages if msg.__class__.__name__ == 'SystemMessage']
        other_messages = [msg for msg in messages if msg.__class__.__name__ != 'SystemMessage']
        
        # Calculate how many recent messages we can keep
        system_tokens = self._estimate_tokens(system_messages)
        remaining_tokens = target_limit - system_tokens
        
        # Keep as many recent messages as possible
        kept_messages = []
        current_tokens = 0
        
        for msg in reversed(other_messages):
            msg_tokens = self._estimate_tokens([msg])
            if current_tokens + msg_tokens <= remaining_tokens:
                kept_messages.insert(0, msg)
                current_tokens += msg_tokens
            else:
                break
        
        return system_messages + kept_messages
    
    def _estimate_tokens(self, messages: List[LLMMessage]) -> int:
        """Estimate token count for messages."""
        total_chars = 0
        for msg in messages:
            if isinstance(msg.content, str):
                total_chars += len(msg.content)
            # Add overhead for message structure
            total_chars += 50  # Rough estimate for message metadata
        
        # Rough approximation: 4 characters ≈ 1 token
        return total_chars // 4
    
    def _get_context_limit(self) -> int:
        """Get context limit for the current model."""
        model_info = self.model_client.model_info
        return model_info.get("context_window", 4096)  # Default to 4K if unknown
