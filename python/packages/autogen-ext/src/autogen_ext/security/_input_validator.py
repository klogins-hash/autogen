"""
Input validation and sanitization for AutoGen enhanced system.
"""

import re
import html
import json
import base64
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable
import unicodedata


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    severity: ValidationSeverity
    issues: List[str] = field(default_factory=list)
    sanitized_input: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationRule:
    """Defines a validation rule."""
    name: str
    description: str
    validator: Callable[[str], ValidationResult]
    enabled: bool = True
    severity: ValidationSeverity = ValidationSeverity.ERROR


@dataclass
class SanitizationConfig:
    """Configuration for input sanitization."""
    remove_html: bool = True
    decode_entities: bool = True
    normalize_unicode: bool = True
    remove_control_chars: bool = True
    max_length: Optional[int] = None
    allowed_chars: Optional[str] = None
    forbidden_chars: Optional[str] = None
    preserve_whitespace: bool = False


class InputValidator:
    """Validates and sanitizes user input."""
    
    def __init__(self, sanitization_config: Optional[SanitizationConfig] = None):
        self.sanitization_config = sanitization_config or SanitizationConfig()
        self.validation_rules: List[ValidationRule] = []
        
        # Initialize default validation rules
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default validation rules."""
        
        # Length validation
        self.add_rule(ValidationRule(
            name="max_length",
            description="Check maximum input length",
            validator=self._validate_length,
            severity=ValidationSeverity.ERROR
        ))
        
        # Character validation
        self.add_rule(ValidationRule(
            name="allowed_characters",
            description="Check for allowed characters only",
            validator=self._validate_characters,
            severity=ValidationSeverity.WARNING
        ))
        
        # HTML/Script injection
        self.add_rule(ValidationRule(
            name="html_injection",
            description="Check for HTML/script injection",
            validator=self._validate_html_injection,
            severity=ValidationSeverity.CRITICAL
        ))
        
        # SQL injection patterns
        self.add_rule(ValidationRule(
            name="sql_injection",
            description="Check for SQL injection patterns",
            validator=self._validate_sql_injection,
            severity=ValidationSeverity.CRITICAL
        ))
        
        # Command injection
        self.add_rule(ValidationRule(
            name="command_injection",
            description="Check for command injection patterns",
            validator=self._validate_command_injection,
            severity=ValidationSeverity.CRITICAL
        ))
        
        # Path traversal
        self.add_rule(ValidationRule(
            name="path_traversal",
            description="Check for path traversal attempts",
            validator=self._validate_path_traversal,
            severity=ValidationSeverity.ERROR
        ))
        
        # Encoding attacks
        self.add_rule(ValidationRule(
            name="encoding_attacks",
            description="Check for encoding-based attacks",
            validator=self._validate_encoding_attacks,
            severity=ValidationSeverity.WARNING
        ))
        
        # Control characters
        self.add_rule(ValidationRule(
            name="control_characters",
            description="Check for control characters",
            validator=self._validate_control_characters,
            severity=ValidationSeverity.WARNING
        ))
    
    def add_rule(self, rule: ValidationRule) -> None:
        """Add a validation rule."""
        self.validation_rules.append(rule)
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove a validation rule by name."""
        for i, rule in enumerate(self.validation_rules):
            if rule.name == rule_name:
                del self.validation_rules[i]
                return True
        return False
    
    def enable_rule(self, rule_name: str) -> bool:
        """Enable a validation rule."""
        for rule in self.validation_rules:
            if rule.name == rule_name:
                rule.enabled = True
                return True
        return False
    
    def disable_rule(self, rule_name: str) -> bool:
        """Disable a validation rule."""
        for rule in self.validation_rules:
            if rule.name == rule_name:
                rule.enabled = False
                return True
        return False
    
    def validate(self, input_text: str) -> ValidationResult:
        """Validate input against all enabled rules."""
        
        all_issues = []
        max_severity = ValidationSeverity.INFO
        
        # Run all enabled validation rules
        for rule in self.validation_rules:
            if not rule.enabled:
                continue
            
            try:
                result = rule.validator(input_text)
                
                if not result.is_valid:
                    all_issues.extend([f"{rule.name}: {issue}" for issue in result.issues])
                    
                    # Update max severity
                    severity_order = {
                        ValidationSeverity.INFO: 0,
                        ValidationSeverity.WARNING: 1,
                        ValidationSeverity.ERROR: 2,
                        ValidationSeverity.CRITICAL: 3
                    }
                    
                    if severity_order[result.severity] > severity_order[max_severity]:
                        max_severity = result.severity
            
            except Exception as e:
                all_issues.append(f"{rule.name}: Validation error - {str(e)}")
                max_severity = ValidationSeverity.ERROR
        
        is_valid = len(all_issues) == 0 or max_severity in [ValidationSeverity.INFO, ValidationSeverity.WARNING]
        
        return ValidationResult(
            is_valid=is_valid,
            severity=max_severity,
            issues=all_issues
        )
    
    def sanitize(self, input_text: str) -> str:
        """Sanitize input according to configuration."""
        
        sanitized = input_text
        
        # Remove HTML tags
        if self.sanitization_config.remove_html:
            sanitized = re.sub(r'<[^>]+>', '', sanitized)
        
        # Decode HTML entities
        if self.sanitization_config.decode_entities:
            sanitized = html.unescape(sanitized)
        
        # Normalize Unicode
        if self.sanitization_config.normalize_unicode:
            sanitized = unicodedata.normalize('NFKC', sanitized)
        
        # Remove control characters
        if self.sanitization_config.remove_control_chars:
            sanitized = ''.join(char for char in sanitized if not unicodedata.category(char).startswith('C'))
        
        # Apply character filters
        if self.sanitization_config.forbidden_chars:
            for char in self.sanitization_config.forbidden_chars:
                sanitized = sanitized.replace(char, '')
        
        if self.sanitization_config.allowed_chars:
            sanitized = ''.join(char for char in sanitized if char in self.sanitization_config.allowed_chars)
        
        # Handle whitespace
        if not self.sanitization_config.preserve_whitespace:
            sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        # Apply length limit
        if self.sanitization_config.max_length:
            if len(sanitized) > self.sanitization_config.max_length:
                sanitized = sanitized[:self.sanitization_config.max_length]
        
        return sanitized
    
    def validate_and_sanitize(self, input_text: str) -> ValidationResult:
        """Validate input and return sanitized version."""
        
        # First validate
        validation_result = self.validate(input_text)
        
        # Then sanitize
        sanitized = self.sanitize(input_text)
        validation_result.sanitized_input = sanitized
        
        # Re-validate sanitized input for critical issues
        if validation_result.severity == ValidationSeverity.CRITICAL:
            sanitized_validation = self.validate(sanitized)
            if sanitized_validation.severity != ValidationSeverity.CRITICAL:
                validation_result.is_valid = True
                validation_result.severity = sanitized_validation.severity
                validation_result.issues.append("Input sanitized to resolve critical issues")
        
        return validation_result
    
    # Validation rule implementations
    
    def _validate_length(self, input_text: str) -> ValidationResult:
        """Validate input length."""
        
        if not self.sanitization_config.max_length:
            return ValidationResult(is_valid=True, severity=ValidationSeverity.INFO)
        
        if len(input_text) > self.sanitization_config.max_length:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                issues=[f"Input length {len(input_text)} exceeds maximum {self.sanitization_config.max_length}"]
            )
        
        return ValidationResult(is_valid=True, severity=ValidationSeverity.INFO)
    
    def _validate_characters(self, input_text: str) -> ValidationResult:
        """Validate allowed/forbidden characters."""
        
        issues = []
        
        # Check forbidden characters
        if self.sanitization_config.forbidden_chars:
            found_forbidden = []
            for char in self.sanitization_config.forbidden_chars:
                if char in input_text:
                    found_forbidden.append(char)
            
            if found_forbidden:
                issues.append(f"Contains forbidden characters: {', '.join(found_forbidden)}")
        
        # Check allowed characters
        if self.sanitization_config.allowed_chars:
            invalid_chars = []
            for char in input_text:
                if char not in self.sanitization_config.allowed_chars:
                    invalid_chars.append(char)
            
            if invalid_chars:
                unique_invalid = list(set(invalid_chars))
                issues.append(f"Contains disallowed characters: {', '.join(unique_invalid[:10])}")
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            severity=ValidationSeverity.WARNING if issues else ValidationSeverity.INFO,
            issues=issues
        )
    
    def _validate_html_injection(self, input_text: str) -> ValidationResult:
        """Validate for HTML/script injection."""
        
        issues = []
        
        # Check for script tags
        script_patterns = [
            r'<script[^>]*>',
            r'</script>',
            r'javascript:',
            r'vbscript:',
            r'onload\s*=',
            r'onerror\s*=',
            r'onclick\s*=',
            r'onmouseover\s*=',
        ]
        
        for pattern in script_patterns:
            if re.search(pattern, input_text, re.IGNORECASE):
                issues.append(f"Potential script injection detected: {pattern}")
        
        # Check for HTML tags that could be dangerous
        dangerous_tags = [
            r'<iframe[^>]*>',
            r'<object[^>]*>',
            r'<embed[^>]*>',
            r'<form[^>]*>',
            r'<input[^>]*>',
            r'<link[^>]*>',
            r'<meta[^>]*>',
        ]
        
        for pattern in dangerous_tags:
            if re.search(pattern, input_text, re.IGNORECASE):
                issues.append(f"Potentially dangerous HTML tag detected: {pattern}")
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            severity=ValidationSeverity.CRITICAL if issues else ValidationSeverity.INFO,
            issues=issues
        )
    
    def _validate_sql_injection(self, input_text: str) -> ValidationResult:
        """Validate for SQL injection patterns."""
        
        issues = []
        
        # Common SQL injection patterns
        sql_patterns = [
            r"(?i)(\bUNION\b.*\bSELECT\b)",
            r"(?i)(\bSELECT\b.*\bFROM\b)",
            r"(?i)(\bINSERT\b.*\bINTO\b)",
            r"(?i)(\bUPDATE\b.*\bSET\b)",
            r"(?i)(\bDELETE\b.*\bFROM\b)",
            r"(?i)(\bDROP\b.*\bTABLE\b)",
            r"(?i)(\bCREATE\b.*\bTABLE\b)",
            r"(?i)(\bALTER\b.*\bTABLE\b)",
            r"(?i)(--\s*$)",
            r"(?i)(/\*.*\*/)",
            r"(?i)(\bOR\b.*=.*)",
            r"(?i)(\bAND\b.*=.*)",
            r"(?i)(';.*--)",
            r"(?i)(\bEXEC\b|\bEXECUTE\b)",
        ]
        
        for pattern in sql_patterns:
            matches = re.findall(pattern, input_text)
            if matches:
                issues.append(f"Potential SQL injection pattern: {pattern}")
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            severity=ValidationSeverity.CRITICAL if issues else ValidationSeverity.INFO,
            issues=issues
        )
    
    def _validate_command_injection(self, input_text: str) -> ValidationResult:
        """Validate for command injection patterns."""
        
        issues = []
        
        # Command injection patterns
        command_patterns = [
            r'[;&|`$(){}[\]\\]',  # Shell metacharacters
            r'(?i)\b(cat|ls|dir|type|echo|ping|wget|curl|nc|netcat)\b',
            r'(?i)\b(rm|del|rmdir|mkdir|touch|chmod|chown)\b',
            r'(?i)\b(ps|kill|killall|pkill|top|htop)\b',
            r'(?i)\b(sudo|su|passwd|useradd|userdel)\b',
            r'(?i)\b(python|perl|ruby|php|node|java)\b\s+',
            r'(?i)\b(bash|sh|cmd|powershell|pwsh)\b',
        ]
        
        for pattern in command_patterns:
            if re.search(pattern, input_text):
                issues.append(f"Potential command injection pattern: {pattern}")
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            severity=ValidationSeverity.CRITICAL if issues else ValidationSeverity.INFO,
            issues=issues
        )
    
    def _validate_path_traversal(self, input_text: str) -> ValidationResult:
        """Validate for path traversal attempts."""
        
        issues = []
        
        # Path traversal patterns
        traversal_patterns = [
            r'\.\./+',
            r'\.\.\\+',
            r'%2e%2e%2f',
            r'%2e%2e%5c',
            r'\.\.%2f',
            r'\.\.%5c',
            r'/etc/passwd',
            r'/etc/shadow',
            r'C:\\Windows\\System32',
            r'C:/Windows/System32',
        ]
        
        for pattern in traversal_patterns:
            if re.search(pattern, input_text, re.IGNORECASE):
                issues.append(f"Potential path traversal pattern: {pattern}")
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            severity=ValidationSeverity.ERROR if issues else ValidationSeverity.INFO,
            issues=issues
        )
    
    def _validate_encoding_attacks(self, input_text: str) -> ValidationResult:
        """Validate for encoding-based attacks."""
        
        issues = []
        
        # Check for various encoding patterns
        encoding_patterns = [
            (r'%[0-9a-fA-F]{2}', 'URL encoding'),
            (r'&#x?[0-9a-fA-F]+;', 'HTML entity encoding'),
            (r'\\u[0-9a-fA-F]{4}', 'Unicode escape'),
            (r'\\x[0-9a-fA-F]{2}', 'Hex escape'),
            (r'\\[0-7]{3}', 'Octal escape'),
        ]
        
        for pattern, encoding_type in encoding_patterns:
            matches = re.findall(pattern, input_text)
            if len(matches) > 10:  # Threshold for suspicious encoding
                issues.append(f"Excessive {encoding_type} detected ({len(matches)} instances)")
        
        # Check for base64-like patterns
        base64_pattern = r'[A-Za-z0-9+/]{20,}={0,2}'
        base64_matches = re.findall(base64_pattern, input_text)
        
        for match in base64_matches:
            try:
                decoded = base64.b64decode(match, validate=True)
                # Check if decoded content contains suspicious patterns
                decoded_str = decoded.decode('utf-8', errors='ignore')
                if any(keyword in decoded_str.lower() for keyword in ['script', 'eval', 'exec', 'system']):
                    issues.append(f"Suspicious base64 encoded content detected")
            except Exception:
                pass  # Not valid base64 or not decodable
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            severity=ValidationSeverity.WARNING if issues else ValidationSeverity.INFO,
            issues=issues
        )
    
    def _validate_control_characters(self, input_text: str) -> ValidationResult:
        """Validate for control characters."""
        
        issues = []
        control_chars = []
        
        for char in input_text:
            if unicodedata.category(char).startswith('C') and char not in ['\n', '\r', '\t']:
                control_chars.append(f"U+{ord(char):04X}")
        
        if control_chars:
            unique_control_chars = list(set(control_chars))
            issues.append(f"Control characters detected: {', '.join(unique_control_chars[:10])}")
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            severity=ValidationSeverity.WARNING if issues else ValidationSeverity.INFO,
            issues=issues
        )
    
    def get_rule_info(self) -> List[Dict[str, Any]]:
        """Get information about all validation rules."""
        
        return [
            {
                "name": rule.name,
                "description": rule.description,
                "enabled": rule.enabled,
                "severity": rule.severity.value
            }
            for rule in self.validation_rules
        ]
    
    def update_sanitization_config(self, config: SanitizationConfig) -> None:
        """Update sanitization configuration."""
        
        self.sanitization_config = config


# Predefined validator configurations

def create_strict_validator() -> InputValidator:
    """Create a validator with strict security settings."""
    
    config = SanitizationConfig(
        remove_html=True,
        decode_entities=True,
        normalize_unicode=True,
        remove_control_chars=True,
        max_length=1000,
        forbidden_chars="<>&\"'`$;|&(){}[]\\",
        preserve_whitespace=False
    )
    
    return InputValidator(config)


def create_permissive_validator() -> InputValidator:
    """Create a validator with permissive settings."""
    
    config = SanitizationConfig(
        remove_html=False,
        decode_entities=True,
        normalize_unicode=True,
        remove_control_chars=True,
        max_length=10000,
        preserve_whitespace=True
    )
    
    validator = InputValidator(config)
    
    # Disable some strict rules
    validator.disable_rule("html_injection")
    validator.disable_rule("command_injection")
    
    return validator


def create_code_validator() -> InputValidator:
    """Create a validator suitable for code input."""
    
    config = SanitizationConfig(
        remove_html=True,
        decode_entities=False,
        normalize_unicode=False,
        remove_control_chars=False,
        max_length=50000,
        preserve_whitespace=True
    )
    
    validator = InputValidator(config)
    
    # Disable rules that might interfere with code
    validator.disable_rule("command_injection")  # Code might contain command-like syntax
    validator.disable_rule("sql_injection")      # Code might contain SQL-like syntax
    
    return validator
