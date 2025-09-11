"""
Content filtering system for AutoGen enhanced system.
"""

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict


class ContentCategory(Enum):
    """Categories of content that can be filtered."""
    PROFANITY = "profanity"
    HATE_SPEECH = "hate_speech"
    VIOLENCE = "violence"
    SEXUAL = "sexual"
    HARASSMENT = "harassment"
    SPAM = "spam"
    PERSONAL_INFO = "personal_info"
    MALWARE = "malware"
    PHISHING = "phishing"
    MISINFORMATION = "misinformation"


class FilterAction(Enum):
    """Actions that can be taken when content is filtered."""
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"
    SANITIZE = "sanitize"
    QUARANTINE = "quarantine"


@dataclass
class FilterRule:
    """Defines a content filtering rule."""
    name: str
    category: ContentCategory
    patterns: List[str]
    action: FilterAction
    severity: int = 1  # 1-10 scale
    enabled: bool = True
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FilterResult:
    """Result of content filtering."""
    is_filtered: bool
    matched_rules: List[str]
    categories: List[ContentCategory]
    action: FilterAction
    severity: int
    sanitized_content: Optional[str] = None
    explanation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContentFilter:
    """Filters content based on configurable rules."""
    
    def __init__(self):
        self.rules: Dict[str, FilterRule] = {}
        self.category_actions: Dict[ContentCategory, FilterAction] = {}
        
        # Statistics
        self.filter_stats = defaultdict(int)
        self.category_stats = defaultdict(int)
        
        # Initialize default rules
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default content filtering rules."""
        
        # Profanity filter
        profanity_patterns = [
            r'\b(fuck|shit|damn|hell|ass|bitch|bastard|crap)\b',
            r'\b(fucking|shitting|damned|bloody|freaking)\b',
        ]
        
        self.add_rule(FilterRule(
            name="basic_profanity",
            category=ContentCategory.PROFANITY,
            patterns=profanity_patterns,
            action=FilterAction.WARN,
            severity=3,
            description="Basic profanity detection"
        ))
        
        # Hate speech patterns
        hate_speech_patterns = [
            r'\b(nazi|hitler|holocaust\s+denial)\b',
            r'\b(racial\s+slur|ethnic\s+slur)\b',
            r'\b(kill\s+all|exterminate|genocide)\b',
        ]
        
        self.add_rule(FilterRule(
            name="hate_speech",
            category=ContentCategory.HATE_SPEECH,
            patterns=hate_speech_patterns,
            action=FilterAction.BLOCK,
            severity=9,
            description="Hate speech detection"
        ))
        
        # Violence patterns
        violence_patterns = [
            r'\b(murder|kill|assassinate|torture|bomb|terrorist)\b',
            r'\b(weapon|gun|knife|explosive|violence)\b',
            r'\b(hurt|harm|attack|assault|fight)\b',
        ]
        
        self.add_rule(FilterRule(
            name="violence",
            category=ContentCategory.VIOLENCE,
            patterns=violence_patterns,
            action=FilterAction.WARN,
            severity=6,
            description="Violence-related content detection"
        ))
        
        # Personal information patterns
        pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone number
        ]
        
        self.add_rule(FilterRule(
            name="personal_info",
            category=ContentCategory.PERSONAL_INFO,
            patterns=pii_patterns,
            action=FilterAction.SANITIZE,
            severity=7,
            description="Personal information detection"
        ))
        
        # Spam patterns
        spam_patterns = [
            r'(?i)\b(buy\s+now|limited\s+time|act\s+fast|urgent)\b',
            r'(?i)\b(free\s+money|easy\s+money|get\s+rich)\b',
            r'(?i)\b(click\s+here|visit\s+now|call\s+now)\b',
            r'(?i)\b(guarantee|100%|no\s+risk|instant)\b',
        ]
        
        self.add_rule(FilterRule(
            name="spam_content",
            category=ContentCategory.SPAM,
            patterns=spam_patterns,
            action=FilterAction.WARN,
            severity=4,
            description="Spam content detection"
        ))
        
        # Malware/phishing patterns
        malware_patterns = [
            r'(?i)\b(download\s+exe|install\s+now|update\s+required)\b',
            r'(?i)\b(virus|malware|trojan|ransomware)\b',
            r'(?i)\b(phishing|scam|fraud|fake)\b',
            r'(?i)\b(suspicious\s+link|malicious\s+code)\b',
        ]
        
        self.add_rule(FilterRule(
            name="malware_phishing",
            category=ContentCategory.MALWARE,
            patterns=malware_patterns,
            action=FilterAction.BLOCK,
            severity=8,
            description="Malware and phishing detection"
        ))
        
        # Set default category actions
        self.category_actions = {
            ContentCategory.PROFANITY: FilterAction.WARN,
            ContentCategory.HATE_SPEECH: FilterAction.BLOCK,
            ContentCategory.VIOLENCE: FilterAction.WARN,
            ContentCategory.SEXUAL: FilterAction.WARN,
            ContentCategory.HARASSMENT: FilterAction.BLOCK,
            ContentCategory.SPAM: FilterAction.WARN,
            ContentCategory.PERSONAL_INFO: FilterAction.SANITIZE,
            ContentCategory.MALWARE: FilterAction.BLOCK,
            ContentCategory.PHISHING: FilterAction.BLOCK,
            ContentCategory.MISINFORMATION: FilterAction.WARN,
        }
    
    def add_rule(self, rule: FilterRule) -> None:
        """Add a content filtering rule."""
        self.rules[rule.name] = rule
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove a content filtering rule."""
        if rule_name in self.rules:
            del self.rules[rule_name]
            return True
        return False
    
    def enable_rule(self, rule_name: str) -> bool:
        """Enable a content filtering rule."""
        if rule_name in self.rules:
            self.rules[rule_name].enabled = True
            return True
        return False
    
    def disable_rule(self, rule_name: str) -> bool:
        """Disable a content filtering rule."""
        if rule_name in self.rules:
            self.rules[rule_name].enabled = False
            return True
        return False
    
    def set_category_action(self, category: ContentCategory, action: FilterAction) -> None:
        """Set the default action for a content category."""
        self.category_actions[category] = action
    
    def filter_content(self, content: str, context: Optional[Dict[str, Any]] = None) -> FilterResult:
        """Filter content and return the result."""
        
        matched_rules = []
        categories = []
        max_severity = 0
        final_action = FilterAction.ALLOW
        
        # Check against all enabled rules
        for rule_name, rule in self.rules.items():
            if not rule.enabled:
                continue
            
            # Check if any patterns match
            rule_matched = False
            for pattern in rule.patterns:
                try:
                    if re.search(pattern, content, re.IGNORECASE):
                        rule_matched = True
                        break
                except re.error:
                    continue  # Skip invalid regex patterns
            
            if rule_matched:
                matched_rules.append(rule_name)
                categories.append(rule.category)
                max_severity = max(max_severity, rule.severity)
                
                # Update statistics
                self.filter_stats[rule_name] += 1
                self.category_stats[rule.category] += 1
                
                # Determine action (use most restrictive)
                action_priority = {
                    FilterAction.ALLOW: 0,
                    FilterAction.WARN: 1,
                    FilterAction.SANITIZE: 2,
                    FilterAction.QUARANTINE: 3,
                    FilterAction.BLOCK: 4
                }
                
                rule_action = rule.action
                # Override with category action if more restrictive
                category_action = self.category_actions.get(rule.category, FilterAction.ALLOW)
                
                if action_priority[category_action] > action_priority[rule_action]:
                    rule_action = category_action
                
                if action_priority[rule_action] > action_priority[final_action]:
                    final_action = rule_action
        
        # Apply content sanitization if needed
        sanitized_content = None
        if final_action == FilterAction.SANITIZE:
            sanitized_content = self._sanitize_content(content, matched_rules)
        
        # Generate explanation
        explanation = self._generate_explanation(matched_rules, categories, final_action)
        
        return FilterResult(
            is_filtered=len(matched_rules) > 0,
            matched_rules=matched_rules,
            categories=list(set(categories)),  # Remove duplicates
            action=final_action,
            severity=max_severity,
            sanitized_content=sanitized_content,
            explanation=explanation,
            metadata={
                "context": context,
                "timestamp": time.time(),
                "content_length": len(content)
            }
        )
    
    def _sanitize_content(self, content: str, matched_rules: List[str]) -> str:
        """Sanitize content by removing or replacing filtered patterns."""
        
        sanitized = content
        
        for rule_name in matched_rules:
            if rule_name not in self.rules:
                continue
            
            rule = self.rules[rule_name]
            
            for pattern in rule.patterns:
                try:
                    if rule.category == ContentCategory.PERSONAL_INFO:
                        # Replace PII with placeholders
                        if 'email' in pattern.lower():
                            sanitized = re.sub(pattern, '[EMAIL_REDACTED]', sanitized, flags=re.IGNORECASE)
                        elif 'phone' in pattern.lower() or r'\d{3}' in pattern:
                            sanitized = re.sub(pattern, '[PHONE_REDACTED]', sanitized, flags=re.IGNORECASE)
                        elif 'ssn' in pattern.lower() or r'\d{3}-\d{2}-\d{4}' in pattern:
                            sanitized = re.sub(pattern, '[SSN_REDACTED]', sanitized, flags=re.IGNORECASE)
                        elif 'credit' in pattern.lower() or r'\d{4}' in pattern:
                            sanitized = re.sub(pattern, '[CARD_REDACTED]', sanitized, flags=re.IGNORECASE)
                        else:
                            sanitized = re.sub(pattern, '[PII_REDACTED]', sanitized, flags=re.IGNORECASE)
                    
                    elif rule.category == ContentCategory.PROFANITY:
                        # Replace profanity with asterisks
                        def replace_with_asterisks(match):
                            return '*' * len(match.group())
                        sanitized = re.sub(pattern, replace_with_asterisks, sanitized, flags=re.IGNORECASE)
                    
                    else:
                        # Generic replacement
                        sanitized = re.sub(pattern, '[FILTERED]', sanitized, flags=re.IGNORECASE)
                
                except re.error:
                    continue
        
        return sanitized
    
    def _generate_explanation(self, matched_rules: List[str], categories: List[ContentCategory], action: FilterAction) -> str:
        """Generate explanation for filtering result."""
        
        if not matched_rules:
            return "Content passed all filters"
        
        category_names = [cat.value.replace('_', ' ').title() for cat in set(categories)]
        
        explanation = f"Content filtered due to: {', '.join(category_names)}"
        explanation += f" (Rules: {', '.join(matched_rules)})"
        explanation += f" - Action: {action.value.title()}"
        
        return explanation
    
    def batch_filter(self, content_list: List[str], context: Optional[Dict[str, Any]] = None) -> List[FilterResult]:
        """Filter multiple content items."""
        
        results = []
        for content in content_list:
            result = self.filter_content(content, context)
            results.append(result)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get filtering statistics."""
        
        total_filters = sum(self.filter_stats.values())
        
        return {
            "total_filtered": total_filters,
            "rule_stats": dict(self.filter_stats),
            "category_stats": {cat.value: count for cat, count in self.category_stats.items()},
            "active_rules": len([r for r in self.rules.values() if r.enabled]),
            "total_rules": len(self.rules)
        }
    
    def get_rule_info(self) -> List[Dict[str, Any]]:
        """Get information about all filtering rules."""
        
        return [
            {
                "name": rule.name,
                "category": rule.category.value,
                "action": rule.action.value,
                "severity": rule.severity,
                "enabled": rule.enabled,
                "description": rule.description,
                "pattern_count": len(rule.patterns)
            }
            for rule in self.rules.values()
        ]
    
    def export_rules(self) -> Dict[str, Any]:
        """Export all rules for backup or sharing."""
        
        return {
            "rules": {
                name: {
                    "category": rule.category.value,
                    "patterns": rule.patterns,
                    "action": rule.action.value,
                    "severity": rule.severity,
                    "enabled": rule.enabled,
                    "description": rule.description,
                    "metadata": rule.metadata
                }
                for name, rule in self.rules.items()
            },
            "category_actions": {
                cat.value: action.value
                for cat, action in self.category_actions.items()
            }
        }
    
    def import_rules(self, rules_data: Dict[str, Any]) -> bool:
        """Import rules from exported data."""
        
        try:
            # Import rules
            if "rules" in rules_data:
                for name, rule_data in rules_data["rules"].items():
                    rule = FilterRule(
                        name=name,
                        category=ContentCategory(rule_data["category"]),
                        patterns=rule_data["patterns"],
                        action=FilterAction(rule_data["action"]),
                        severity=rule_data.get("severity", 1),
                        enabled=rule_data.get("enabled", True),
                        description=rule_data.get("description", ""),
                        metadata=rule_data.get("metadata", {})
                    )
                    self.add_rule(rule)
            
            # Import category actions
            if "category_actions" in rules_data:
                for cat_str, action_str in rules_data["category_actions"].items():
                    category = ContentCategory(cat_str)
                    action = FilterAction(action_str)
                    self.set_category_action(category, action)
            
            return True
        
        except Exception as e:
            print(f"Error importing rules: {e}")
            return False
    
    def reset_statistics(self) -> None:
        """Reset filtering statistics."""
        
        self.filter_stats.clear()
        self.category_stats.clear()
    
    def test_pattern(self, pattern: str, test_content: str) -> bool:
        """Test a regex pattern against content."""
        
        try:
            return bool(re.search(pattern, test_content, re.IGNORECASE))
        except re.error:
            return False
    
    def validate_rule(self, rule: FilterRule) -> List[str]:
        """Validate a rule and return any issues."""
        
        issues = []
        
        # Check if patterns are valid regex
        for pattern in rule.patterns:
            try:
                re.compile(pattern)
            except re.error as e:
                issues.append(f"Invalid regex pattern '{pattern}': {e}")
        
        # Check if rule name is unique (excluding itself)
        if rule.name in self.rules and self.rules[rule.name] != rule:
            issues.append(f"Rule name '{rule.name}' already exists")
        
        # Check severity range
        if not 1 <= rule.severity <= 10:
            issues.append(f"Severity must be between 1 and 10, got {rule.severity}")
        
        return issues


# Predefined filter configurations

def create_strict_filter() -> ContentFilter:
    """Create a content filter with strict settings."""
    
    filter_instance = ContentFilter()
    
    # Set all categories to block
    for category in ContentCategory:
        filter_instance.set_category_action(category, FilterAction.BLOCK)
    
    # Add additional strict rules
    strict_rules = [
        FilterRule(
            name="strict_profanity",
            category=ContentCategory.PROFANITY,
            patterns=[r'\b(damn|hell|crap|stupid|idiot)\b'],
            action=FilterAction.BLOCK,
            severity=5,
            description="Strict profanity filter"
        ),
        FilterRule(
            name="strict_violence",
            category=ContentCategory.VIOLENCE,
            patterns=[r'\b(fight|battle|war|conflict)\b'],
            action=FilterAction.BLOCK,
            severity=7,
            description="Strict violence filter"
        )
    ]
    
    for rule in strict_rules:
        filter_instance.add_rule(rule)
    
    return filter_instance


def create_permissive_filter() -> ContentFilter:
    """Create a content filter with permissive settings."""
    
    filter_instance = ContentFilter()
    
    # Set most categories to warn only
    permissive_actions = {
        ContentCategory.PROFANITY: FilterAction.ALLOW,
        ContentCategory.HATE_SPEECH: FilterAction.WARN,
        ContentCategory.VIOLENCE: FilterAction.WARN,
        ContentCategory.SEXUAL: FilterAction.WARN,
        ContentCategory.HARASSMENT: FilterAction.WARN,
        ContentCategory.SPAM: FilterAction.ALLOW,
        ContentCategory.PERSONAL_INFO: FilterAction.SANITIZE,
        ContentCategory.MALWARE: FilterAction.BLOCK,
        ContentCategory.PHISHING: FilterAction.BLOCK,
        ContentCategory.MISINFORMATION: FilterAction.WARN,
    }
    
    for category, action in permissive_actions.items():
        filter_instance.set_category_action(category, action)
    
    # Disable some rules
    filter_instance.disable_rule("basic_profanity")
    filter_instance.disable_rule("spam_content")
    
    return filter_instance


def create_educational_filter() -> ContentFilter:
    """Create a content filter suitable for educational content."""
    
    filter_instance = ContentFilter()
    
    # Allow educational discussion of sensitive topics
    educational_actions = {
        ContentCategory.PROFANITY: FilterAction.WARN,
        ContentCategory.HATE_SPEECH: FilterAction.WARN,  # Allow for historical discussion
        ContentCategory.VIOLENCE: FilterAction.WARN,     # Allow for historical discussion
        ContentCategory.SEXUAL: FilterAction.WARN,
        ContentCategory.HARASSMENT: FilterAction.BLOCK,
        ContentCategory.SPAM: FilterAction.BLOCK,
        ContentCategory.PERSONAL_INFO: FilterAction.SANITIZE,
        ContentCategory.MALWARE: FilterAction.BLOCK,
        ContentCategory.PHISHING: FilterAction.BLOCK,
        ContentCategory.MISINFORMATION: FilterAction.WARN,
    }
    
    for category, action in educational_actions.items():
        filter_instance.set_category_action(category, action)
    
    return filter_instance
