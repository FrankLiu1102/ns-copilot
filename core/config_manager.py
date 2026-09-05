"""
Configuration Manager for NS-Copilot

Provides runtime configuration management with persistence.
Allows users to configure models, API keys, and other settings
through the web interface without modifying code.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict


# Default config file location
CONFIG_DIR = Path(os.environ.get("NEURO_COPILOT_CONFIG_DIR", Path.home() / ".neuro_copilot"))
CONFIG_FILE = CONFIG_DIR / "config.json"


@dataclass
class DebateMoEConfig:
    """Debate MoE specific configuration"""
    enabled: bool = False
    max_rounds: int = 5
    consensus_threshold: float = 0.9
    
    # Expert models
    advocate_model: str = "gpt-4o"
    critic_model: str = "gpt-4o"
    synthesizer_model: str = "gpt-4o"
    judge_model: str = "gpt-4o"
    
    # Expert temperatures
    advocate_temperature: float = 0.7
    critic_temperature: float = 0.7
    synthesizer_temperature: float = 0.6
    judge_temperature: float = 0.3


@dataclass
class APIConfig:
    """API configuration"""
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"


@dataclass
class PipelineConfig:
    """Pipeline configuration (QA-related)"""
    router_model: str = "gpt-4o"
    task_agent_model: str = "gpt-4o"
    controller_model: str = "gpt-4o"
    
    # KB settings
    kb_top_k: int = 5
    
    # Online search settings
    enable_online_search: bool = True
    online_search_top_k: int = 3


@dataclass
class DAConfig:
    """Data Analysis pipeline configuration"""
    planner_model: str = "gpt-5.1"
    planner_temperature: float = 0.2
    code_generator_model: str = "gpt-5.1"
    code_generator_temperature: float = 0.2
    controller_diagnosis_model: str = "gpt-5.1"
    controller_diagnosis_temperature: float = 0.1
    controller_prompt_refine_model: str = "gpt-5.1"
    controller_prompt_refine_temperature: float = 0.2
    analysis_model: str = "gpt-5.1"
    analysis_temperature: float = 0.3
    max_steps: int = 50
    seed: int = 0
    # Controller settings
    controller_mode: str = "force"          # "off" | "auto" | "force"
    controller_force_count: int = 2         # Number of forced retries (force mode)
    controller_auto_max_count: int = 2      # Max retries Controller may choose (auto mode)


@dataclass 
class NeuroCopilotConfig:
    """Complete NS-Copilot configuration"""
    api: APIConfig = field(default_factory=APIConfig)
    debate_moe: DebateMoEConfig = field(default_factory=DebateMoEConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    da: DAConfig = field(default_factory=DAConfig)
    
    # Metadata
    version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NeuroCopilotConfig":
        """Create from dictionary"""
        config = cls()
        
        if "api" in data:
            config.api = APIConfig(**data["api"])
        if "debate_moe" in data:
            config.debate_moe = DebateMoEConfig(**data["debate_moe"])
        if "pipeline" in data:
            config.pipeline = PipelineConfig(**data["pipeline"])
        if "da" in data:
            config.da = DAConfig(**data["da"])
        if "version" in data:
            config.version = data["version"]
            
        return config


class ConfigManager:
    """
    Manages NS-Copilot configuration with persistence.
    
    Features:
    - Load/save configuration from JSON file
    - Apply configuration to runtime environment
    - Validate configuration values
    
    Usage:
        manager = ConfigManager()
        config = manager.load()
        config.debate_moe.enabled = True
        manager.save(config)
        manager.apply(config)
    """
    
    def __init__(self, config_file: Optional[Path] = None):
        """Initialize config manager"""
        self.config_file = config_file or CONFIG_FILE
        self._current_config: Optional[NeuroCopilotConfig] = None
    
    def load(self) -> NeuroCopilotConfig:
        """
        Load configuration from file.
        Falls back to defaults if file doesn't exist.
        """
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                self._current_config = NeuroCopilotConfig.from_dict(data)
            except Exception as e:
                print(f"Warning: Could not load config file: {e}")
                self._current_config = self._create_default_config()
        else:
            self._current_config = self._create_default_config()
        
        return self._current_config
    
    def _create_default_config(self) -> NeuroCopilotConfig:
        """Create default configuration from current environment/config.py values"""
        from neuro_copilot.core import config as cfg
        
        return NeuroCopilotConfig(
            api=APIConfig(
                openai_api_key=getattr(cfg, 'OPENAI_API_KEY', ''),
                openai_base_url=getattr(cfg, 'OPENAI_BASE_URL', 'https://api.openai.com/v1'),
                anthropic_api_key=getattr(cfg, 'ANTHROPIC_API_KEY', ''),
                ollama_base_url=getattr(cfg, 'OLLAMA_BASE_URL', 'http://localhost:11434'),
            ),
            debate_moe=DebateMoEConfig(
                enabled=getattr(cfg, 'ENABLE_DEBATE_MOE', False),
                max_rounds=getattr(cfg, 'DEBATE_MAX_ROUNDS', 5),
                consensus_threshold=getattr(cfg, 'DEBATE_CONSENSUS_THRESHOLD', 0.9),
                advocate_model=getattr(cfg, 'ADVOCATE_MODEL', 'gpt-4o'),
                critic_model=getattr(cfg, 'CRITIC_MODEL', 'gpt-4o'),
                synthesizer_model=getattr(cfg, 'SYNTHESIZER_MODEL', 'gpt-4o'),
                judge_model=getattr(cfg, 'JUDGE_MODEL', 'gpt-4o'),
                advocate_temperature=getattr(cfg, 'ADVOCATE_TEMPERATURE', 0.7),
                critic_temperature=getattr(cfg, 'CRITIC_TEMPERATURE', 0.7),
                synthesizer_temperature=getattr(cfg, 'SYNTHESIZER_TEMPERATURE', 0.6),
                judge_temperature=getattr(cfg, 'JUDGE_TEMPERATURE', 0.3),
            ),
            pipeline=PipelineConfig(
                router_model=getattr(cfg, 'ROUTER_MODEL', 'gpt-4o'),
                task_agent_model=getattr(cfg, 'TASK_AGENT_MODEL', 'gpt-4o'),
                controller_model=getattr(cfg, 'CONTROLLER_MODEL', 'gpt-4o'),
                kb_top_k=getattr(cfg, 'KB_TOP_K', 5),
            ),
            da=DAConfig(
                planner_model=getattr(cfg, 'PLANNER_MODEL', 'gpt-4o'),
                planner_temperature=getattr(cfg, 'PLANNER_TEMPERATURE', 0.2),
                code_generator_model=getattr(cfg, 'CODE_GENERATOR_MODEL', 'gpt-4o'),
                code_generator_temperature=getattr(cfg, 'CODE_GENERATOR_TEMPERATURE', 0.2),
                controller_diagnosis_model=getattr(cfg, 'CONTROLLER_DIAGNOSIS_MODEL', 'gpt-4o'),
                controller_diagnosis_temperature=getattr(cfg, 'CONTROLLER_DIAGNOSIS_TEMPERATURE', 0.1),
                controller_prompt_refine_model=getattr(cfg, 'CONTROLLER_PROMPT_REFINE_MODEL', 'gpt-4o'),
                controller_prompt_refine_temperature=getattr(cfg, 'CONTROLLER_PROMPT_REFINE_TEMPERATURE', 0.2),
                analysis_model=getattr(cfg, 'ANALYSIS_MODEL', 'gpt-4o'),
                analysis_temperature=getattr(cfg, 'ANALYSIS_TEMPERATURE', 0.3),
                max_steps=getattr(cfg, 'DA_MAX_STEPS', 50),
                seed=getattr(cfg, 'DA_SEED', 0),
                controller_mode=getattr(cfg, 'CONTROLLER_MODE', 'force'),
                controller_force_count=getattr(cfg, 'FORCE_IMPROVEMENT_COUNT', 2),
                controller_auto_max_count=getattr(cfg, 'AUTO_MAX_IMPROVEMENT_COUNT', 2),
            ),
        )
    
    def save(self, config: NeuroCopilotConfig) -> bool:
        """
        Save configuration to file.
        
        Returns:
            True if save was successful
        """
        try:
            # Ensure config directory exists
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_file, 'w') as f:
                json.dump(config.to_dict(), f, indent=2)
            
            self._current_config = config
            return True
            
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
    
    def apply(self, config: NeuroCopilotConfig):
        """
        Apply configuration to runtime environment.
        Updates environment variables and module-level config values.
        """
        # Apply API keys to environment
        if config.api.openai_api_key:
            os.environ['OPENAI_API_KEY'] = config.api.openai_api_key
        if config.api.anthropic_api_key:
            os.environ['ANTHROPIC_API_KEY'] = config.api.anthropic_api_key
        if config.api.openai_base_url:
            os.environ['OPENAI_BASE_URL'] = config.api.openai_base_url
        if config.api.ollama_base_url:
            os.environ['OLLAMA_BASE_URL'] = config.api.ollama_base_url
        
        # Apply Debate MoE settings
        os.environ['ENABLE_DEBATE_MOE'] = str(config.debate_moe.enabled).lower()
        os.environ['DEBATE_MAX_ROUNDS'] = str(config.debate_moe.max_rounds)
        os.environ['DEBATE_CONSENSUS_THRESHOLD'] = str(config.debate_moe.consensus_threshold)
        os.environ['ADVOCATE_MODEL'] = config.debate_moe.advocate_model
        os.environ['CRITIC_MODEL'] = config.debate_moe.critic_model
        os.environ['SYNTHESIZER_MODEL'] = config.debate_moe.synthesizer_model
        os.environ['JUDGE_MODEL'] = config.debate_moe.judge_model
        os.environ['ADVOCATE_TEMPERATURE'] = str(config.debate_moe.advocate_temperature)
        os.environ['CRITIC_TEMPERATURE'] = str(config.debate_moe.critic_temperature)
        os.environ['SYNTHESIZER_TEMPERATURE'] = str(config.debate_moe.synthesizer_temperature)
        os.environ['JUDGE_TEMPERATURE'] = str(config.debate_moe.judge_temperature)
        
        # Apply pipeline settings
        os.environ['ROUTER_MODEL'] = config.pipeline.router_model
        os.environ['TASK_AGENT_MODEL'] = config.pipeline.task_agent_model
        os.environ['CONTROLLER_MODEL'] = config.pipeline.controller_model
        os.environ['KB_TOP_K'] = str(config.pipeline.kb_top_k)
        
        # Apply DA settings
        os.environ['PLANNER_MODEL'] = config.da.planner_model
        os.environ['PLANNER_TEMPERATURE'] = str(config.da.planner_temperature)
        os.environ['CODE_GENERATOR_MODEL'] = config.da.code_generator_model
        os.environ['CODE_GENERATOR_TEMPERATURE'] = str(config.da.code_generator_temperature)
        os.environ['CONTROLLER_DIAGNOSIS_MODEL'] = config.da.controller_diagnosis_model
        os.environ['CONTROLLER_DIAGNOSIS_TEMPERATURE'] = str(config.da.controller_diagnosis_temperature)
        os.environ['CONTROLLER_PROMPT_REFINE_MODEL'] = config.da.controller_prompt_refine_model
        os.environ['CONTROLLER_PROMPT_REFINE_TEMPERATURE'] = str(config.da.controller_prompt_refine_temperature)
        os.environ['ANALYSIS_MODEL'] = config.da.analysis_model
        os.environ['ANALYSIS_TEMPERATURE'] = str(config.da.analysis_temperature)
        os.environ['DA_MAX_STEPS'] = str(config.da.max_steps)
        os.environ['DA_SEED'] = str(config.da.seed)
        # Controller settings
        os.environ['CONTROLLER_MODE'] = config.da.controller_mode
        os.environ['FORCE_IMPROVEMENT_COUNT'] = str(config.da.controller_force_count)
        os.environ['AUTO_MAX_IMPROVEMENT_COUNT'] = str(config.da.controller_auto_max_count)

        self._current_config = config
    
    def get_current(self) -> NeuroCopilotConfig:
        """Get current configuration (loads if not already loaded)"""
        if self._current_config is None:
            return self.load()
        return self._current_config
    
    def validate_api_key(self, provider: str, api_key: str) -> tuple[bool, str]:
        """
        Validate an API key by making a test request.
        
        Returns:
            Tuple of (is_valid, message)
        """
        # Clean the API key - remove invisible Unicode characters
        # Common problematic characters: \u2028 (line separator), \u2029 (paragraph separator), 
        # \ufeff (BOM), various whitespace
        import re
        api_key = api_key.strip()
        api_key = re.sub(r'[\u2028\u2029\ufeff\u200b\u200c\u200d\xa0]', '', api_key)
        api_key = ''.join(char for char in api_key if ord(char) < 128 or char.isalnum() or char in '-_')
        
        if not api_key or len(api_key) < 10:
            return False, "API key is empty or too short"
        
        try:
            if provider == "openai":
                from openai import OpenAI
                client = OpenAI(api_key=api_key, timeout=10)
                # Make a minimal test request
                client.models.list()
                return True, "OpenAI API key is valid"
                
            elif provider == "anthropic":
                import anthropic
                client = anthropic.Anthropic(api_key=api_key, timeout=10)
                # Make a minimal test request
                client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=5,
                    messages=[{"role": "user", "content": "Hi"}]
                )
                return True, "Anthropic API key is valid"
                
            else:
                return False, f"Unknown provider: {provider}"
                
        except Exception as e:
            return False, f"API key validation failed: {str(e)[:100]}"


# Global instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get global config manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def load_and_apply_config():
    """Load configuration and apply it to runtime"""
    manager = get_config_manager()
    config = manager.load()
    manager.apply(config)
    return config
