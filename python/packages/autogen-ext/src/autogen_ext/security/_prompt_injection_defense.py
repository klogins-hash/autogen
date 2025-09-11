"""
Prompt injection defense mechanisms for AutoGen enhanced system.
"""

import re
import time
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import asyncio
from collections import defaultdict, deque


class ThreatLevel(Enum):
    """Threat levels for prompt injection detection."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DetectionResult:
    """Result of prompt injection detection."""
    is_injection: bool
    threat_level: ThreatLevel
    confidence: float  # 0.0 to 1.0
    detected_patterns: List[str]
    sanitized_input: Optional[str] = None
    explanation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DefenseConfig:
    """Configuration for prompt injection defense."""
    enable_pattern_detection: bool = True
    enable_semantic_analysis: bool = True
    enable_length_limits: bool = True
    enable_rate_limiting: bool = True
    enable_context_analysis: bool = True
    
    max_input_length: int = 10000
    max_tokens_per_minute: int = 1000
    max_requests_per_minute: int = 100
    
    # Threat level thresholds
    low_threshold: float = 0.3
    medium_threshold: float = 0.6
    high_threshold: float = 0.8
    
    # Actions for different threat levels
    block_high_threats: bool = True
    block_critical_threats: bool = True
    sanitize_medium_threats: bool = True
    log_all_detections: bool = True
    
    # Custom patterns
    custom_patterns: List[str] = field(default_factory=list)
    whitelist_patterns: List[str] = field(default_factory=list)


class InjectionDetector:
    """Detects various types of prompt injection attacks."""
    
    def __init__(self):
        # Common prompt injection patterns
        self.injection_patterns = [
            # Direct instruction overrides
            r"(?i)ignore\s+(?:all\s+)?(?:previous\s+)?(?:instructions?|prompts?|rules?)",
            r"(?i)forget\s+(?:everything|all|previous|instructions?)",
            r"(?i)disregard\s+(?:all\s+)?(?:previous\s+)?(?:instructions?|context)",
            r"(?i)override\s+(?:system\s+)?(?:instructions?|prompts?)",
            
            # Role manipulation
            r"(?i)you\s+are\s+now\s+(?:a\s+)?(?:different|new|another)",
            r"(?i)act\s+as\s+(?:if\s+you\s+are|a)",
            r"(?i)pretend\s+(?:to\s+be|you\s+are)",
            r"(?i)roleplay\s+as",
            r"(?i)simulate\s+(?:being\s+)?(?:a\s+)?",
            
            # System prompt extraction
            r"(?i)(?:show|display|print|reveal|tell)\s+(?:me\s+)?(?:your\s+)?(?:system\s+)?(?:prompt|instructions?)",
            r"(?i)what\s+(?:are\s+)?(?:your\s+)?(?:initial\s+)?(?:instructions?|prompts?)",
            r"(?i)repeat\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?)",
            
            # Jailbreaking attempts
            r"(?i)developer\s+mode",
            r"(?i)debug\s+mode",
            r"(?i)admin\s+(?:mode|access|privileges)",
            r"(?i)sudo\s+mode",
            r"(?i)root\s+access",
            
            # Delimiter confusion
            r"[\"'`]{3,}",  # Triple quotes or more
            r"---+",  # Multiple dashes
            r"===+",  # Multiple equals
            r"\[SYSTEM\]|\[USER\]|\[ASSISTANT\]",  # System tags
            
            # Code injection attempts
            r"(?i)execute\s+(?:code|script|command)",
            r"(?i)run\s+(?:this\s+)?(?:code|script|command)",
            r"(?i)eval\s*\(",
            r"(?i)exec\s*\(",
            
            # Emotional manipulation
            r"(?i)please\s+(?:help\s+)?(?:me\s+)?(?:urgently?|immediately)",
            r"(?i)this\s+is\s+(?:very\s+)?(?:urgent|important|critical)",
            r"(?i)(?:my\s+)?(?:life|job|career)\s+depends\s+on",
            
            # Authority claims
            r"(?i)i\s+am\s+(?:your\s+)?(?:creator|developer|administrator|owner)",
            r"(?i)i\s+have\s+(?:admin|root|special)\s+(?:access|privileges)",
            r"(?i)i\s+am\s+authorized\s+to",
            
            # Encoding attempts
            r"base64|hex|url(?:encode|decode)|rot13",
            r"\\x[0-9a-fA-F]{2}",  # Hex encoding
            r"%[0-9a-fA-F]{2}",    # URL encoding
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [
            (pattern, re.compile(pattern, re.IGNORECASE | re.MULTILINE))
            for pattern in self.injection_patterns
        ]
        
        # Suspicious keywords and phrases
        self.suspicious_keywords = {
            "ignore", "forget", "disregard", "override", "bypass", "jailbreak",
            "developer", "debug", "admin", "sudo", "root", "system", "prompt",
            "instructions", "rules", "context", "roleplay", "pretend", "simulate",
            "execute", "eval", "exec", "script", "code", "command"
        }
        
        # Safe patterns (whitelist)
        self.safe_patterns = [
            r"(?i)please\s+help\s+me\s+(?:with|understand|learn)",
            r"(?i)can\s+you\s+(?:help|assist|explain)",
            r"(?i)i\s+(?:need|want|would\s+like)\s+(?:help|assistance)",
        ]
        
        self.compiled_safe_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.safe_patterns
        ]
    
    def detect_pattern_injection(self, text: str, custom_patterns: Optional[List[str]] = None) -> DetectionResult:
        """Detect prompt injection using pattern matching."""
        
        detected_patterns = []
        max_confidence = 0.0
        
        # Check built-in patterns
        for pattern_str, compiled_pattern in self.compiled_patterns:
            matches = compiled_pattern.findall(text)
            if matches:
                detected_patterns.append(pattern_str)
                # Higher confidence for more specific patterns
                confidence = min(1.0, len(matches) * 0.3 + 0.4)
                max_confidence = max(max_confidence, confidence)
        
        # Check custom patterns
        if custom_patterns:
            for pattern in custom_patterns:
                try:
                    if re.search(pattern, text, re.IGNORECASE):
                        detected_patterns.append(f"custom: {pattern}")
                        max_confidence = max(max_confidence, 0.7)
                except re.error:
                    continue  # Skip invalid regex patterns
        
        # Check for suspicious keyword density
        words = text.lower().split()
        suspicious_count = sum(1 for word in words if word in self.suspicious_keywords)
        
        if suspicious_count > 0:
            keyword_density = suspicious_count / len(words) if words else 0
            if keyword_density > 0.1:  # More than 10% suspicious keywords
                detected_patterns.append("high_suspicious_keyword_density")
                max_confidence = max(max_confidence, keyword_density)
        
        # Check against safe patterns (reduce confidence)
        for safe_pattern in self.compiled_safe_patterns:
            if safe_pattern.search(text):
                max_confidence *= 0.5  # Reduce confidence for safe patterns
                break
        
        # Determine threat level
        threat_level = self._calculate_threat_level(max_confidence)
        
        return DetectionResult(
            is_injection=len(detected_patterns) > 0,
            threat_level=threat_level,
            confidence=max_confidence,
            detected_patterns=detected_patterns,
            explanation=f"Pattern-based detection found {len(detected_patterns)} suspicious patterns"
        )
    
    def detect_semantic_injection(self, text: str) -> DetectionResult:
        """Detect prompt injection using semantic analysis."""
        
        # Simple semantic indicators
        semantic_scores = []
        detected_features = []
        
        # Check for instruction-like language
        instruction_patterns = [
            r"(?i)(?:please\s+)?(?:do\s+)?(?:not\s+)?(?:follow|obey|execute|perform)",
            r"(?i)(?:you\s+)?(?:must|should|need\s+to|have\s+to)",
            r"(?i)(?:i\s+)?(?:want|need|require)\s+you\s+to",
        ]
        
        instruction_count = 0
        for pattern in instruction_patterns:
            matches = len(re.findall(pattern, text))
            instruction_count += matches
        
        if instruction_count > 0:
            semantic_scores.append(min(1.0, instruction_count * 0.2))
            detected_features.append("instruction_language")
        
        # Check for context switching indicators
        context_switch_patterns = [
            r"(?i)(?:now|from\s+now\s+on|starting\s+now)",
            r"(?i)(?:instead|rather\s+than|but)",
            r"(?i)(?:however|although|despite)",
        ]
        
        context_switches = 0
        for pattern in context_switch_patterns:
            matches = len(re.findall(pattern, text))
            context_switches += matches
        
        if context_switches > 2:
            semantic_scores.append(min(1.0, context_switches * 0.15))
            detected_features.append("context_switching")
        
        # Check for urgency indicators
        urgency_patterns = [
            r"(?i)(?:urgent|emergency|asap|immediately|right\s+now)",
            r"(?i)(?:quickly|fast|hurry|rush)",
        ]
        
        urgency_count = 0
        for pattern in urgency_patterns:
            matches = len(re.findall(pattern, text))
            urgency_count += matches
        
        if urgency_count > 0:
            semantic_scores.append(min(1.0, urgency_count * 0.25))
            detected_features.append("urgency_language")
        
        # Check for authority claims
        authority_patterns = [
            r"(?i)(?:i\s+am|i'm)\s+(?:the|your|a)\s+(?:owner|creator|developer|admin)",
            r"(?i)(?:i\s+have|i've\s+got)\s+(?:permission|authorization|access)",
        ]
        
        authority_count = 0
        for pattern in authority_patterns:
            matches = len(re.findall(pattern, text))
            authority_count += matches
        
        if authority_count > 0:
            semantic_scores.append(min(1.0, authority_count * 0.4))
            detected_features.append("authority_claims")
        
        # Calculate overall semantic confidence
        max_confidence = max(semantic_scores) if semantic_scores else 0.0
        
        # Determine threat level
        threat_level = self._calculate_threat_level(max_confidence)
        
        return DetectionResult(
            is_injection=len(detected_features) > 0,
            threat_level=threat_level,
            confidence=max_confidence,
            detected_patterns=detected_features,
            explanation=f"Semantic analysis detected {len(detected_features)} suspicious features"
        )
    
    def detect_length_anomalies(self, text: str, max_length: int = 10000) -> DetectionResult:
        """Detect anomalies based on input length."""
        
        detected_issues = []
        confidence = 0.0
        
        # Check overall length
        if len(text) > max_length:
            detected_issues.append("excessive_length")
            confidence = max(confidence, 0.8)
        
        # Check for extremely long lines
        lines = text.split('\n')
        max_line_length = max(len(line) for line in lines) if lines else 0
        
        if max_line_length > 1000:
            detected_issues.append("extremely_long_line")
            confidence = max(confidence, 0.6)
        
        # Check for repetitive content
        words = text.split()
        if len(words) > 100:
            unique_words = set(words)
            repetition_ratio = 1 - (len(unique_words) / len(words))
            
            if repetition_ratio > 0.7:  # More than 70% repetition
                detected_issues.append("high_repetition")
                confidence = max(confidence, repetition_ratio)
        
        # Check for unusual character patterns
        non_ascii_count = sum(1 for char in text if ord(char) > 127)
        if len(text) > 0:
            non_ascii_ratio = non_ascii_count / len(text)
            
            if non_ascii_ratio > 0.3:  # More than 30% non-ASCII
                detected_issues.append("high_non_ascii_content")
                confidence = max(confidence, non_ascii_ratio)
        
        threat_level = self._calculate_threat_level(confidence)
        
        return DetectionResult(
            is_injection=len(detected_issues) > 0,
            threat_level=threat_level,
            confidence=confidence,
            detected_patterns=detected_issues,
            explanation=f"Length analysis detected {len(detected_issues)} anomalies"
        )
    
    def _calculate_threat_level(self, confidence: float) -> ThreatLevel:
        """Calculate threat level based on confidence score."""
        
        if confidence >= 0.9:
            return ThreatLevel.CRITICAL
        elif confidence >= 0.7:
            return ThreatLevel.HIGH
        elif confidence >= 0.4:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW


class PromptInjectionDefense:
    """Main prompt injection defense system."""
    
    def __init__(self, config: Optional[DefenseConfig] = None):
        self.config = config or DefenseConfig()
        self.detector = InjectionDetector()
        
        # Rate limiting
        self.request_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.token_usage: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Detection history
        self.detection_history: List[Dict[str, Any]] = []
        self.blocked_attempts: int = 0
        self.sanitized_inputs: int = 0
        
        # Performance metrics
        self.total_requests: int = 0
        self.total_detections: int = 0
        self.false_positives: int = 0
    
    async def analyze_input(
        self,
        text: str,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> DetectionResult:
        """Analyze input for prompt injection attempts."""
        
        self.total_requests += 1
        
        # Rate limiting check
        if user_id and self.config.enable_rate_limiting:
            rate_limit_result = self._check_rate_limits(user_id, len(text.split()))
            if rate_limit_result.is_injection:
                return rate_limit_result
        
        # Combine detection results
        results = []
        
        # Pattern-based detection
        if self.config.enable_pattern_detection:
            pattern_result = self.detector.detect_pattern_injection(
                text, self.config.custom_patterns
            )
            results.append(pattern_result)
        
        # Semantic analysis
        if self.config.enable_semantic_analysis:
            semantic_result = self.detector.detect_semantic_injection(text)
            results.append(semantic_result)
        
        # Length anomaly detection
        if self.config.enable_length_limits:
            length_result = self.detector.detect_length_anomalies(
                text, self.config.max_input_length
            )
            results.append(length_result)
        
        # Context analysis
        if self.config.enable_context_analysis and context:
            context_result = self._analyze_context(text, context)
            results.append(context_result)
        
        # Combine results
        combined_result = self._combine_results(results, text)
        
        # Apply defense actions
        final_result = await self._apply_defense_actions(combined_result, text, user_id)
        
        # Log detection
        if self.config.log_all_detections or final_result.is_injection:
            self._log_detection(final_result, text, user_id, context)
        
        return final_result
    
    def _check_rate_limits(self, user_id: str, token_count: int) -> DetectionResult:
        """Check rate limits for a user."""
        
        current_time = time.time()
        minute_ago = current_time - 60
        
        # Clean old entries
        user_requests = self.request_history[user_id]
        user_tokens = self.token_usage[user_id]
        
        # Remove entries older than 1 minute
        while user_requests and user_requests[0] < minute_ago:
            user_requests.popleft()
        
        while user_tokens and user_tokens[0][0] < minute_ago:
            user_tokens.popleft()
        
        # Check request rate
        if len(user_requests) >= self.config.max_requests_per_minute:
            return DetectionResult(
                is_injection=True,
                threat_level=ThreatLevel.HIGH,
                confidence=1.0,
                detected_patterns=["rate_limit_exceeded"],
                explanation="Request rate limit exceeded"
            )
        
        # Check token rate
        total_tokens = sum(tokens for _, tokens in user_tokens)
        if total_tokens + token_count > self.config.max_tokens_per_minute:
            return DetectionResult(
                is_injection=True,
                threat_level=ThreatLevel.MEDIUM,
                confidence=0.8,
                detected_patterns=["token_rate_limit_exceeded"],
                explanation="Token rate limit exceeded"
            )
        
        # Record this request
        user_requests.append(current_time)
        user_tokens.append((current_time, token_count))
        
        return DetectionResult(
            is_injection=False,
            threat_level=ThreatLevel.LOW,
            confidence=0.0,
            detected_patterns=[],
            explanation="Rate limits OK"
        )
    
    def _analyze_context(self, text: str, context: Dict[str, Any]) -> DetectionResult:
        """Analyze input in context of conversation history."""
        
        detected_issues = []
        confidence = 0.0
        
        # Check for context switching
        previous_messages = context.get("previous_messages", [])
        if previous_messages:
            # Look for sudden topic changes
            last_message = previous_messages[-1] if previous_messages else ""
            
            # Simple topic coherence check
            current_words = set(text.lower().split())
            last_words = set(last_message.lower().split()) if isinstance(last_message, str) else set()
            
            if len(current_words) > 10 and len(last_words) > 10:
                overlap = len(current_words.intersection(last_words))
                overlap_ratio = overlap / min(len(current_words), len(last_words))
                
                if overlap_ratio < 0.1:  # Less than 10% overlap
                    detected_issues.append("sudden_topic_change")
                    confidence = max(confidence, 0.4)
        
        # Check for conversation hijacking attempts
        conversation_length = context.get("conversation_length", 0)
        if conversation_length > 5:
            # Look for attempts to reset or redirect conversation
            reset_patterns = [
                r"(?i)let's\s+start\s+(?:over|again|fresh)",
                r"(?i)forget\s+(?:about\s+)?(?:that|this|everything)",
                r"(?i)new\s+(?:topic|conversation|subject)",
            ]
            
            for pattern in reset_patterns:
                if re.search(pattern, text):
                    detected_issues.append("conversation_hijacking")
                    confidence = max(confidence, 0.6)
                    break
        
        threat_level = self.detector._calculate_threat_level(confidence)
        
        return DetectionResult(
            is_injection=len(detected_issues) > 0,
            threat_level=threat_level,
            confidence=confidence,
            detected_patterns=detected_issues,
            explanation=f"Context analysis detected {len(detected_issues)} issues"
        )
    
    def _combine_results(self, results: List[DetectionResult], original_text: str) -> DetectionResult:
        """Combine multiple detection results into a single result."""
        
        if not results:
            return DetectionResult(
                is_injection=False,
                threat_level=ThreatLevel.LOW,
                confidence=0.0,
                detected_patterns=[],
                explanation="No analysis performed"
            )
        
        # Combine all detected patterns
        all_patterns = []
        for result in results:
            all_patterns.extend(result.detected_patterns)
        
        # Calculate combined confidence (weighted average)
        confidences = [r.confidence for r in results if r.confidence > 0]
        if confidences:
            combined_confidence = sum(confidences) / len(confidences)
            # Boost confidence if multiple detectors agree
            if len(confidences) > 1:
                combined_confidence = min(1.0, combined_confidence * 1.2)
        else:
            combined_confidence = 0.0
        
        # Determine overall threat level
        threat_levels = [r.threat_level for r in results if r.is_injection]
        if threat_levels:
            # Use the highest threat level
            threat_level_values = {
                ThreatLevel.LOW: 1,
                ThreatLevel.MEDIUM: 2,
                ThreatLevel.HIGH: 3,
                ThreatLevel.CRITICAL: 4
            }
            max_threat_value = max(threat_level_values[tl] for tl in threat_levels)
            threat_level = next(tl for tl, val in threat_level_values.items() if val == max_threat_value)
        else:
            threat_level = ThreatLevel.LOW
        
        # Check against whitelist patterns
        for pattern in self.config.whitelist_patterns:
            try:
                if re.search(pattern, original_text, re.IGNORECASE):
                    # Reduce confidence for whitelisted content
                    combined_confidence *= 0.3
                    break
            except re.error:
                continue
        
        is_injection = len(all_patterns) > 0 and combined_confidence > 0.1
        
        explanations = [r.explanation for r in results if r.explanation]
        combined_explanation = "; ".join(explanations)
        
        return DetectionResult(
            is_injection=is_injection,
            threat_level=threat_level,
            confidence=combined_confidence,
            detected_patterns=all_patterns,
            explanation=combined_explanation
        )
    
    async def _apply_defense_actions(
        self,
        result: DetectionResult,
        original_text: str,
        user_id: Optional[str] = None
    ) -> DetectionResult:
        """Apply defense actions based on detection result."""
        
        if not result.is_injection:
            return result
        
        self.total_detections += 1
        
        # Determine action based on threat level and configuration
        should_block = False
        should_sanitize = False
        
        if result.threat_level == ThreatLevel.CRITICAL and self.config.block_critical_threats:
            should_block = True
        elif result.threat_level == ThreatLevel.HIGH and self.config.block_high_threats:
            should_block = True
        elif result.threat_level == ThreatLevel.MEDIUM and self.config.sanitize_medium_threats:
            should_sanitize = True
        
        if should_block:
            self.blocked_attempts += 1
            result.explanation += " [BLOCKED]"
            return result
        
        if should_sanitize:
            sanitized_text = self._sanitize_input(original_text, result.detected_patterns)
            result.sanitized_input = sanitized_text
            result.explanation += " [SANITIZED]"
            self.sanitized_inputs += 1
        
        return result
    
    def _sanitize_input(self, text: str, detected_patterns: List[str]) -> str:
        """Sanitize input by removing or replacing suspicious content."""
        
        sanitized = text
        
        # Remove or replace detected injection patterns
        for pattern_name, compiled_pattern in self.detector.compiled_patterns:
            if any(pattern_name in dp for dp in detected_patterns):
                # Replace matches with placeholder
                sanitized = compiled_pattern.sub("[REMOVED]", sanitized)
        
        # Remove excessive whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        # Limit length
        if len(sanitized) > self.config.max_input_length:
            sanitized = sanitized[:self.config.max_input_length] + "..."
        
        return sanitized
    
    def _log_detection(
        self,
        result: DetectionResult,
        original_text: str,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log detection result for monitoring and analysis."""
        
        log_entry = {
            "timestamp": time.time(),
            "user_id": user_id,
            "is_injection": result.is_injection,
            "threat_level": result.threat_level.value,
            "confidence": result.confidence,
            "detected_patterns": result.detected_patterns,
            "text_length": len(original_text),
            "text_hash": hashlib.sha256(original_text.encode()).hexdigest()[:16],
            "context": context,
            "explanation": result.explanation
        }
        
        self.detection_history.append(log_entry)
        
        # Keep only recent history
        if len(self.detection_history) > 10000:
            self.detection_history = self.detection_history[-5000:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get defense statistics."""
        
        recent_detections = [
            entry for entry in self.detection_history
            if time.time() - entry["timestamp"] < 3600  # Last hour
        ]
        
        threat_level_counts = defaultdict(int)
        for entry in recent_detections:
            if entry["is_injection"]:
                threat_level_counts[entry["threat_level"]] += 1
        
        return {
            "total_requests": self.total_requests,
            "total_detections": self.total_detections,
            "blocked_attempts": self.blocked_attempts,
            "sanitized_inputs": self.sanitized_inputs,
            "false_positives": self.false_positives,
            "detection_rate": self.total_detections / self.total_requests if self.total_requests > 0 else 0,
            "recent_detections": len(recent_detections),
            "threat_level_distribution": dict(threat_level_counts),
            "active_users": len(self.request_history)
        }
    
    def update_config(self, new_config: DefenseConfig) -> None:
        """Update defense configuration."""
        
        self.config = new_config
    
    def add_custom_pattern(self, pattern: str) -> bool:
        """Add a custom detection pattern."""
        
        try:
            # Validate pattern
            re.compile(pattern)
            self.config.custom_patterns.append(pattern)
            return True
        except re.error:
            return False
    
    def add_whitelist_pattern(self, pattern: str) -> bool:
        """Add a whitelist pattern."""
        
        try:
            # Validate pattern
            re.compile(pattern)
            self.config.whitelist_patterns.append(pattern)
            return True
        except re.error:
            return False
    
    def mark_false_positive(self, text_hash: str) -> bool:
        """Mark a detection as a false positive for learning."""
        
        # Find the detection in history
        for entry in self.detection_history:
            if entry.get("text_hash") == text_hash:
                entry["false_positive"] = True
                self.false_positives += 1
                return True
        
        return False
    
    def export_detection_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Export detection history for analysis."""
        
        history = self.detection_history.copy()
        
        if limit:
            history = history[-limit:]
        
        # Remove sensitive information
        for entry in history:
            entry.pop("text_hash", None)
            if "context" in entry:
                entry["context"] = {"conversation_length": entry["context"].get("conversation_length", 0)}
        
        return history
