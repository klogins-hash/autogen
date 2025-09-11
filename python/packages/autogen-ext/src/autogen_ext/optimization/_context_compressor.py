"""
Context compression utilities for reducing token usage while preserving essential information.
"""

import asyncio
import hashlib
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    LLMMessage,
    SystemMessage,
    UserMessage,
)


class CompressionStrategy(Enum):
    """Available compression strategies."""
    SUMMARIZATION = "summarization"
    SLIDING_WINDOW = "sliding_window"
    RELEVANCE_FILTERING = "relevance_filtering"
    HYBRID = "hybrid"


@dataclass
class CompressionResult:
    """Result of context compression operation."""
    compressed_messages: List[LLMMessage]
    original_token_count: int
    compressed_token_count: int
    compression_ratio: float
    preserved_information: Dict[str, Any]


class BaseCompressor(ABC):
    """Base class for context compression strategies."""
    
    @abstractmethod
    async def compress(
        self, 
        messages: List[LLMMessage], 
        target_tokens: Optional[int] = None
    ) -> CompressionResult:
        """Compress a list of messages."""
        pass


class SummarizationCompressor(BaseCompressor):
    """Compresses context using LLM-based summarization."""
    
    def __init__(self, model_client: ChatCompletionClient, summary_ratio: float = 0.3):
        self.model_client = model_client
        self.summary_ratio = summary_ratio
        
    async def compress(
        self, 
        messages: List[LLMMessage], 
        target_tokens: Optional[int] = None
    ) -> CompressionResult:
        """Compress messages using summarization."""
        original_tokens = self._estimate_tokens(messages)
        
        if len(messages) <= 3:  # Keep minimal conversations intact
            return CompressionResult(
                compressed_messages=messages,
                original_token_count=original_tokens,
                compressed_token_count=original_tokens,
                compression_ratio=1.0,
                preserved_information={}
            )
        
        # Keep system message and last few messages
        system_msgs = [msg for msg in messages if isinstance(msg, SystemMessage)]
        recent_msgs = messages[-2:]  # Keep last 2 messages
        middle_msgs = messages[len(system_msgs):-2]
        
        if not middle_msgs:
            return CompressionResult(
                compressed_messages=messages,
                original_token_count=original_tokens,
                compressed_token_count=original_tokens,
                compression_ratio=1.0,
                preserved_information={}
            )
        
        # Summarize middle messages
        summary_prompt = self._create_summary_prompt(middle_msgs)
        summary_response = await self.model_client.create([
            SystemMessage(content="You are a helpful assistant that creates concise summaries."),
            UserMessage(content=summary_prompt)
        ])
        
        summary_content = summary_response.content
        if isinstance(summary_content, str):
            summary_msg = UserMessage(
                content=f"[SUMMARY] Previous conversation summary: {summary_content}",
                source="context_compressor"
            )
            
            compressed_messages = system_msgs + [summary_msg] + recent_msgs
            compressed_tokens = self._estimate_tokens(compressed_messages)
            
            return CompressionResult(
                compressed_messages=compressed_messages,
                original_token_count=original_tokens,
                compressed_token_count=compressed_tokens,
                compression_ratio=compressed_tokens / original_tokens if original_tokens > 0 else 1.0,
                preserved_information={
                    "summarized_messages": len(middle_msgs),
                    "summary_method": "llm_summarization"
                }
            )
        
        # Fallback to original if summarization fails
        return CompressionResult(
            compressed_messages=messages,
            original_token_count=original_tokens,
            compressed_token_count=original_tokens,
            compression_ratio=1.0,
            preserved_information={"error": "summarization_failed"}
        )
    
    def _create_summary_prompt(self, messages: List[LLMMessage]) -> str:
        """Create a prompt for summarizing messages."""
        conversation_text = "\n".join([
            f"{msg.source}: {msg.content}" for msg in messages 
            if isinstance(msg.content, str)
        ])
        
        return f"""Please create a concise summary of the following conversation that preserves:
1. Key decisions made
2. Important facts discovered
3. Current task progress
4. Any errors or issues encountered

Conversation:
{conversation_text}

Summary (max 200 words):"""
    
    def _estimate_tokens(self, messages: List[LLMMessage]) -> int:
        """Rough token estimation (4 chars ≈ 1 token)."""
        total_chars = sum(
            len(str(msg.content)) for msg in messages 
            if isinstance(msg.content, str)
        )
        return total_chars // 4


class SlidingWindowCompressor(BaseCompressor):
    """Maintains a sliding window of recent messages."""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
    
    async def compress(
        self, 
        messages: List[LLMMessage], 
        target_tokens: Optional[int] = None
    ) -> CompressionResult:
        """Keep only recent messages within window size."""
        original_tokens = self._estimate_tokens(messages)
        
        if len(messages) <= self.window_size:
            return CompressionResult(
                compressed_messages=messages,
                original_token_count=original_tokens,
                compressed_token_count=original_tokens,
                compression_ratio=1.0,
                preserved_information={}
            )
        
        # Keep system messages and recent messages
        system_msgs = [msg for msg in messages if isinstance(msg, SystemMessage)]
        non_system_msgs = [msg for msg in messages if not isinstance(msg, SystemMessage)]
        
        recent_msgs = non_system_msgs[-self.window_size:]
        compressed_messages = system_msgs + recent_msgs
        compressed_tokens = self._estimate_tokens(compressed_messages)
        
        return CompressionResult(
            compressed_messages=compressed_messages,
            original_token_count=original_tokens,
            compressed_token_count=compressed_tokens,
            compression_ratio=compressed_tokens / original_tokens if original_tokens > 0 else 1.0,
            preserved_information={
                "window_size": self.window_size,
                "dropped_messages": len(messages) - len(compressed_messages)
            }
        )
    
    def _estimate_tokens(self, messages: List[LLMMessage]) -> int:
        """Rough token estimation."""
        total_chars = sum(
            len(str(msg.content)) for msg in messages 
            if isinstance(msg.content, str)
        )
        return total_chars // 4


class RelevanceFilteringCompressor(BaseCompressor):
    """Filters messages based on relevance to current task."""
    
    def __init__(self, model_client: ChatCompletionClient, relevance_threshold: float = 0.7):
        self.model_client = model_client
        self.relevance_threshold = relevance_threshold
    
    async def compress(
        self, 
        messages: List[LLMMessage], 
        target_tokens: Optional[int] = None
    ) -> CompressionResult:
        """Filter messages based on relevance scoring."""
        original_tokens = self._estimate_tokens(messages)
        
        if len(messages) <= 5:  # Keep short conversations intact
            return CompressionResult(
                compressed_messages=messages,
                original_token_count=original_tokens,
                compressed_token_count=original_tokens,
                compression_ratio=1.0,
                preserved_information={}
            )
        
        # Always keep system messages and last message
        system_msgs = [msg for msg in messages if isinstance(msg, SystemMessage)]
        last_msg = messages[-1:]
        middle_msgs = messages[len(system_msgs):-1]
        
        # Score relevance of middle messages
        relevant_msgs = await self._filter_relevant_messages(middle_msgs, last_msg[0])
        
        compressed_messages = system_msgs + relevant_msgs + last_msg
        compressed_tokens = self._estimate_tokens(compressed_messages)
        
        return CompressionResult(
            compressed_messages=compressed_messages,
            original_token_count=original_tokens,
            compressed_token_count=compressed_tokens,
            compression_ratio=compressed_tokens / original_tokens if original_tokens > 0 else 1.0,
            preserved_information={
                "relevance_threshold": self.relevance_threshold,
                "filtered_messages": len(middle_msgs) - len(relevant_msgs)
            }
        )
    
    async def _filter_relevant_messages(
        self, 
        messages: List[LLMMessage], 
        current_message: LLMMessage
    ) -> List[LLMMessage]:
        """Filter messages based on relevance to current context."""
        if not messages:
            return messages
        
        # Simple keyword-based relevance for now
        # In production, this could use embedding similarity
        current_content = str(current_message.content).lower()
        current_keywords = set(re.findall(r'\b\w+\b', current_content))
        
        relevant_messages = []
        for msg in messages:
            msg_content = str(msg.content).lower()
            msg_keywords = set(re.findall(r'\b\w+\b', msg_content))
            
            # Calculate keyword overlap
            overlap = len(current_keywords.intersection(msg_keywords))
            total_keywords = len(current_keywords.union(msg_keywords))
            
            if total_keywords > 0:
                relevance_score = overlap / total_keywords
                if relevance_score >= self.relevance_threshold:
                    relevant_messages.append(msg)
        
        return relevant_messages
    
    def _estimate_tokens(self, messages: List[LLMMessage]) -> int:
        """Rough token estimation."""
        total_chars = sum(
            len(str(msg.content)) for msg in messages 
            if isinstance(msg.content, str)
        )
        return total_chars // 4


class ContextCompressor:
    """Main context compression orchestrator."""
    
    def __init__(
        self, 
        model_client: ChatCompletionClient,
        default_strategy: CompressionStrategy = CompressionStrategy.HYBRID
    ):
        self.model_client = model_client
        self.default_strategy = default_strategy
        
        # Initialize compressors
        self.compressors = {
            CompressionStrategy.SUMMARIZATION: SummarizationCompressor(model_client),
            CompressionStrategy.SLIDING_WINDOW: SlidingWindowCompressor(),
            CompressionStrategy.RELEVANCE_FILTERING: RelevanceFilteringCompressor(model_client),
        }
    
    async def compress_context(
        self,
        messages: List[LLMMessage],
        target_tokens: Optional[int] = None,
        strategy: Optional[CompressionStrategy] = None
    ) -> CompressionResult:
        """Compress context using specified or default strategy."""
        strategy = strategy or self.default_strategy
        
        if strategy == CompressionStrategy.HYBRID:
            return await self._hybrid_compression(messages, target_tokens)
        
        compressor = self.compressors.get(strategy)
        if not compressor:
            raise ValueError(f"Unknown compression strategy: {strategy}")
        
        return await compressor.compress(messages, target_tokens)
    
    async def _hybrid_compression(
        self, 
        messages: List[LLMMessage], 
        target_tokens: Optional[int] = None
    ) -> CompressionResult:
        """Apply multiple compression strategies in sequence."""
        current_messages = messages
        total_original_tokens = self._estimate_tokens(messages)
        
        # Step 1: Relevance filtering
        relevance_result = await self.compressors[CompressionStrategy.RELEVANCE_FILTERING].compress(
            current_messages, target_tokens
        )
        current_messages = relevance_result.compressed_messages
        
        # Step 2: If still too long, apply sliding window
        if target_tokens and self._estimate_tokens(current_messages) > target_tokens:
            window_result = await self.compressors[CompressionStrategy.SLIDING_WINDOW].compress(
                current_messages, target_tokens
            )
            current_messages = window_result.compressed_messages
        
        # Step 3: If still too long, apply summarization
        if target_tokens and self._estimate_tokens(current_messages) > target_tokens:
            summary_result = await self.compressors[CompressionStrategy.SUMMARIZATION].compress(
                current_messages, target_tokens
            )
            current_messages = summary_result.compressed_messages
        
        final_tokens = self._estimate_tokens(current_messages)
        
        return CompressionResult(
            compressed_messages=current_messages,
            original_token_count=total_original_tokens,
            compressed_token_count=final_tokens,
            compression_ratio=final_tokens / total_original_tokens if total_original_tokens > 0 else 1.0,
            preserved_information={
                "strategy": "hybrid",
                "steps_applied": ["relevance_filtering", "sliding_window", "summarization"]
            }
        )
    
    def _estimate_tokens(self, messages: List[LLMMessage]) -> int:
        """Rough token estimation."""
        total_chars = sum(
            len(str(msg.content)) for msg in messages 
            if isinstance(msg.content, str)
        )
        return total_chars // 4
