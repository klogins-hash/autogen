"""
Security integration for AutoGen enhanced system.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

from autogen_core import AgentId, MessageContext
from autogen_agentchat import ConversableAgent, ChatCompletionClient

from ._prompt_injection_defense import PromptInjectionDefense, DefenseConfig, ThreatLevel
from ._input_validator import InputValidator, ValidationSeverity, SanitizationConfig
from ._content_filter import ContentFilter, FilterAction, ContentCategory
from ._security_monitor import SecurityMonitor, SecurityEventType, AlertSeverity, AlertConfig


@dataclass
class SecurityConfig:
    """Configuration for integrated security system."""
    enable_prompt_injection_defense: bool = True
    enable_input_validation: bool = True
    enable_content_filtering: bool = True
    enable_security_monitoring: bool = True
    
    # Component configurations
    defense_config: Optional[DefenseConfig] = None
    sanitization_config: Optional[SanitizationConfig] = None
    alert_config: Optional[AlertConfig] = None
    
    # Integration settings
    auto_block_critical_threats: bool = True
    auto_sanitize_medium_threats: bool = True
    log_all_security_events: bool = True
    
    # Performance settings
    async_processing: bool = True
    cache_results: bool = True
    cache_ttl: int = 300  # 5 minutes


class SecurityIntegration:
    """Integrated security system for AutoGen enhanced agents."""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        
        # Initialize security components
        self.prompt_defense: Optional[PromptInjectionDefense] = None
        self.input_validator: Optional[InputValidator] = None
        self.content_filter: Optional[ContentFilter] = None
        self.security_monitor: Optional[SecurityMonitor] = None
        
        # Result cache
        self.result_cache: Dict[str, Dict[str, Any]] = {}
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize security components based on configuration."""
        
        if self.config.enable_prompt_injection_defense:
            defense_config = self.config.defense_config or DefenseConfig()
            self.prompt_defense = PromptInjectionDefense(defense_config)
        
        if self.config.enable_input_validation:
            sanitization_config = self.config.sanitization_config or SanitizationConfig()
            self.input_validator = InputValidator(sanitization_config)
        
        if self.config.enable_content_filtering:
            self.content_filter = ContentFilter()
        
        if self.config.enable_security_monitoring:
            alert_config = self.config.alert_config or AlertConfig()
            self.security_monitor = SecurityMonitor(alert_config)
    
    async def start(self):
        """Start the security integration system."""
        
        if self.security_monitor:
            await self.security_monitor.start_monitoring()
    
    async def stop(self):
        """Stop the security integration system."""
        
        if self.security_monitor:
            await self.security_monitor.stop_monitoring()
    
    async def analyze_input(
        self,
        text: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Comprehensive security analysis of input text."""
        
        # Check cache first
        if self.config.cache_results:
            cache_key = self._generate_cache_key(text, user_id, context)
            if cache_key in self.result_cache:
                cached_result = self.result_cache[cache_key]
                if time.time() - cached_result["timestamp"] < self.config.cache_ttl:
                    return cached_result["result"]
        
        # Initialize result
        analysis_result = {
            "is_safe": True,
            "action": "allow",
            "sanitized_input": None,
            "security_events": [],
            "analysis_details": {}
        }
        
        # Run security analyses
        if self.config.async_processing:
            tasks = []
            
            if self.prompt_defense:
                tasks.append(self._analyze_prompt_injection(text, user_id, context))
            
            if self.input_validator:
                tasks.append(self._analyze_input_validation(text))
            
            if self.content_filter:
                tasks.append(self._analyze_content_filtering(text, context))
            
            # Wait for all analyses to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"Error in security analysis task {i}: {result}")
                    continue
                
                analysis_result["analysis_details"].update(result)
        
        else:
            # Sequential processing
            if self.prompt_defense:
                prompt_result = await self._analyze_prompt_injection(text, user_id, context)
                analysis_result["analysis_details"].update(prompt_result)
            
            if self.input_validator:
                validation_result = await self._analyze_input_validation(text)
                analysis_result["analysis_details"].update(validation_result)
            
            if self.content_filter:
                filter_result = await self._analyze_content_filtering(text, context)
                analysis_result["analysis_details"].update(filter_result)
        
        # Determine overall safety and action
        overall_result = self._determine_overall_action(analysis_result["analysis_details"])
        analysis_result.update(overall_result)
        
        # Log security events
        if self.security_monitor and self.config.log_all_security_events:
            await self._log_security_events(analysis_result, text, user_id, agent_id, context)
        
        # Cache result
        if self.config.cache_results:
            self.result_cache[cache_key] = {
                "result": analysis_result,
                "timestamp": time.time()
            }
            
            # Clean old cache entries
            self._clean_cache()
        
        return analysis_result
    
    async def _analyze_prompt_injection(
        self,
        text: str,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze for prompt injection attacks."""
        
        if not self.prompt_defense:
            return {"prompt_injection": {"enabled": False}}
        
        try:
            result = await self.prompt_defense.analyze_input(text, user_id, context)
            
            return {
                "prompt_injection": {
                    "enabled": True,
                    "is_injection": result.is_injection,
                    "threat_level": result.threat_level.value,
                    "confidence": result.confidence,
                    "detected_patterns": result.detected_patterns,
                    "sanitized_input": result.sanitized_input,
                    "explanation": result.explanation
                }
            }
        
        except Exception as e:
            return {
                "prompt_injection": {
                    "enabled": True,
                    "error": str(e)
                }
            }
    
    async def _analyze_input_validation(self, text: str) -> Dict[str, Any]:
        """Analyze input validation."""
        
        if not self.input_validator:
            return {"input_validation": {"enabled": False}}
        
        try:
            result = self.input_validator.validate_and_sanitize(text)
            
            return {
                "input_validation": {
                    "enabled": True,
                    "is_valid": result.is_valid,
                    "severity": result.severity.value,
                    "issues": result.issues,
                    "sanitized_input": result.sanitized_input
                }
            }
        
        except Exception as e:
            return {
                "input_validation": {
                    "enabled": True,
                    "error": str(e)
                }
            }
    
    async def _analyze_content_filtering(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze content filtering."""
        
        if not self.content_filter:
            return {"content_filtering": {"enabled": False}}
        
        try:
            result = self.content_filter.filter_content(text, context)
            
            return {
                "content_filtering": {
                    "enabled": True,
                    "is_filtered": result.is_filtered,
                    "matched_rules": result.matched_rules,
                    "categories": [cat.value for cat in result.categories],
                    "action": result.action.value,
                    "severity": result.severity,
                    "sanitized_content": result.sanitized_content,
                    "explanation": result.explanation
                }
            }
        
        except Exception as e:
            return {
                "content_filtering": {
                    "enabled": True,
                    "error": str(e)
                }
            }
    
    def _determine_overall_action(self, analysis_details: Dict[str, Any]) -> Dict[str, Any]:
        """Determine overall safety assessment and action."""
        
        is_safe = True
        action = "allow"
        sanitized_input = None
        reasons = []
        
        # Check prompt injection results
        prompt_analysis = analysis_details.get("prompt_injection", {})
        if prompt_analysis.get("enabled") and prompt_analysis.get("is_injection"):
            threat_level = prompt_analysis.get("threat_level", "low")
            
            if threat_level in ["critical", "high"] and self.config.auto_block_critical_threats:
                is_safe = False
                action = "block"
                reasons.append(f"Prompt injection detected (threat level: {threat_level})")
            
            elif threat_level == "medium" and self.config.auto_sanitize_medium_threats:
                action = "sanitize"
                sanitized_input = prompt_analysis.get("sanitized_input")
                reasons.append(f"Prompt injection detected (threat level: {threat_level}) - sanitized")
        
        # Check input validation results
        validation_analysis = analysis_details.get("input_validation", {})
        if validation_analysis.get("enabled") and not validation_analysis.get("is_valid"):
            severity = validation_analysis.get("severity", "info")
            
            if severity in ["critical", "error"]:
                if action != "block":  # Don't override block action
                    action = "sanitize"
                    sanitized_input = validation_analysis.get("sanitized_input") or sanitized_input
                reasons.append(f"Input validation failed (severity: {severity})")
        
        # Check content filtering results
        filter_analysis = analysis_details.get("content_filtering", {})
        if filter_analysis.get("enabled") and filter_analysis.get("is_filtered"):
            filter_action = filter_analysis.get("action", "allow")
            
            if filter_action == "block":
                is_safe = False
                action = "block"
                reasons.append("Content filtering blocked input")
            
            elif filter_action == "sanitize" and action != "block":
                action = "sanitize"
                sanitized_input = filter_analysis.get("sanitized_content") or sanitized_input
                reasons.append("Content filtering sanitized input")
        
        return {
            "is_safe": is_safe,
            "action": action,
            "sanitized_input": sanitized_input,
            "reasons": reasons
        }
    
    async def _log_security_events(
        self,
        analysis_result: Dict[str, Any],
        text: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """Log security events to the security monitor."""
        
        if not self.security_monitor:
            return
        
        analysis_details = analysis_result.get("analysis_details", {})
        
        # Log prompt injection events
        prompt_analysis = analysis_details.get("prompt_injection", {})
        if prompt_analysis.get("enabled") and prompt_analysis.get("is_injection"):
            threat_level = prompt_analysis.get("threat_level", "low")
            
            severity_mapping = {
                "low": AlertSeverity.LOW,
                "medium": AlertSeverity.MEDIUM,
                "high": AlertSeverity.HIGH,
                "critical": AlertSeverity.CRITICAL
            }
            
            await self.security_monitor.log_event(
                event_type=SecurityEventType.PROMPT_INJECTION,
                severity=severity_mapping.get(threat_level, AlertSeverity.MEDIUM),
                source="prompt_injection_defense",
                description=f"Prompt injection detected: {prompt_analysis.get('explanation', '')}",
                user_id=user_id,
                agent_id=agent_id,
                details={
                    "threat_level": threat_level,
                    "confidence": prompt_analysis.get("confidence", 0),
                    "detected_patterns": prompt_analysis.get("detected_patterns", []),
                    "text_length": len(text)
                }
            )
        
        # Log input validation events
        validation_analysis = analysis_details.get("input_validation", {})
        if validation_analysis.get("enabled") and not validation_analysis.get("is_valid"):
            severity = validation_analysis.get("severity", "info")
            
            severity_mapping = {
                "info": AlertSeverity.LOW,
                "warning": AlertSeverity.LOW,
                "error": AlertSeverity.MEDIUM,
                "critical": AlertSeverity.HIGH
            }
            
            await self.security_monitor.log_event(
                event_type=SecurityEventType.INPUT_VALIDATION_FAILURE,
                severity=severity_mapping.get(severity, AlertSeverity.MEDIUM),
                source="input_validator",
                description=f"Input validation failed: {', '.join(validation_analysis.get('issues', []))}",
                user_id=user_id,
                agent_id=agent_id,
                details={
                    "validation_severity": severity,
                    "issues": validation_analysis.get("issues", []),
                    "text_length": len(text)
                }
            )
        
        # Log content filtering events
        filter_analysis = analysis_details.get("content_filtering", {})
        if filter_analysis.get("enabled") and filter_analysis.get("is_filtered"):
            filter_severity = filter_analysis.get("severity", 1)
            
            # Map filter severity (1-10) to alert severity
            if filter_severity >= 8:
                alert_severity = AlertSeverity.CRITICAL
            elif filter_severity >= 6:
                alert_severity = AlertSeverity.HIGH
            elif filter_severity >= 4:
                alert_severity = AlertSeverity.MEDIUM
            else:
                alert_severity = AlertSeverity.LOW
            
            await self.security_monitor.log_event(
                event_type=SecurityEventType.CONTENT_FILTER_VIOLATION,
                severity=alert_severity,
                source="content_filter",
                description=f"Content filtered: {filter_analysis.get('explanation', '')}",
                user_id=user_id,
                agent_id=agent_id,
                details={
                    "matched_rules": filter_analysis.get("matched_rules", []),
                    "categories": filter_analysis.get("categories", []),
                    "filter_action": filter_analysis.get("action", "unknown"),
                    "filter_severity": filter_severity,
                    "text_length": len(text)
                }
            )
    
    def _generate_cache_key(
        self,
        text: str,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate cache key for analysis results."""
        
        import hashlib
        
        # Create a hash of the input parameters
        cache_data = f"{text}:{user_id}:{str(context)}"
        return hashlib.sha256(cache_data.encode()).hexdigest()[:16]
    
    def _clean_cache(self):
        """Clean expired cache entries."""
        
        current_time = time.time()
        expired_keys = [
            key for key, data in self.result_cache.items()
            if current_time - data["timestamp"] > self.config.cache_ttl
        ]
        
        for key in expired_keys:
            del self.result_cache[key]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get security integration statistics."""
        
        stats = {
            "cache_size": len(self.result_cache),
            "components": {}
        }
        
        if self.prompt_defense:
            stats["components"]["prompt_defense"] = self.prompt_defense.get_statistics()
        
        if self.input_validator:
            stats["components"]["input_validator"] = {
                "rules": len(self.input_validator.validation_rules),
                "enabled_rules": len([r for r in self.input_validator.validation_rules if r.enabled])
            }
        
        if self.content_filter:
            stats["components"]["content_filter"] = self.content_filter.get_statistics()
        
        if self.security_monitor:
            stats["components"]["security_monitor"] = self.security_monitor.get_statistics()
        
        return stats
    
    def update_config(self, config: SecurityConfig):
        """Update security configuration."""
        
        self.config = config
        
        # Reinitialize components with new config
        self._initialize_components()
    
    async def test_input(self, test_cases: List[str]) -> List[Dict[str, Any]]:
        """Test multiple inputs for security analysis."""
        
        results = []
        
        for test_case in test_cases:
            result = await self.analyze_input(test_case)
            results.append({
                "input": test_case[:100] + "..." if len(test_case) > 100 else test_case,
                "result": result
            })
        
        return results


# Decorator for easy agent integration
def security_enabled(config: Optional[SecurityConfig] = None):
    """Decorator to enable security integration for an agent class."""
    
    def decorator(agent_class):
        original_init = agent_class.__init__
        
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            
            # Add security integration
            self._security_integration = SecurityIntegration(config)
            
            # Start security monitoring
            asyncio.create_task(self._security_integration.start())
        
        agent_class.__init__ = new_init
        
        # Add security methods to the agent class
        async def analyze_input_security(self, text: str, user_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
            """Analyze input for security threats."""
            agent_id = getattr(self, 'id', str(id(self)))
            return await self._security_integration.analyze_input(text, user_id, agent_id, context)
        
        def get_security_statistics(self):
            """Get security statistics."""
            return self._security_integration.get_statistics()
        
        agent_class.analyze_input_security = analyze_input_security
        agent_class.get_security_statistics = get_security_statistics
        
        return agent_class
    
    return decorator


# Security middleware for message processing
class SecurityMiddleware:
    """Middleware for processing messages through security checks."""
    
    def __init__(self, security_integration: SecurityIntegration):
        self.security_integration = security_integration
    
    async def process_message(
        self,
        message: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a message through security checks."""
        
        # Analyze the message
        analysis_result = await self.security_integration.analyze_input(
            message, user_id, agent_id, context
        )
        
        # Determine how to handle the message
        if analysis_result["action"] == "block":
            return {
                "allowed": False,
                "message": None,
                "reason": "Message blocked by security system",
                "details": analysis_result
            }
        
        elif analysis_result["action"] == "sanitize":
            sanitized_message = analysis_result.get("sanitized_input", message)
            return {
                "allowed": True,
                "message": sanitized_message,
                "reason": "Message sanitized by security system",
                "details": analysis_result
            }
        
        else:  # allow
            return {
                "allowed": True,
                "message": message,
                "reason": "Message passed security checks",
                "details": analysis_result
            }
