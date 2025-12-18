"""Dynamic Configuration Manager for R-DeepONet HPO System

This module handles dynamic path resolution and configuration management
to eliminate hardcoded paths and make the system portable across different environments.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union
import yaml
import json
from datetime import datetime


class ConfigManager:
    """Manages dynamic configuration for R-DeepONet HPO system."""
    
    def __init__(self, base_config_path: Optional[Union[str, Path]] = None):
        self.project_root = Path.cwd()
        self.base_config_path = Path(base_config_path) if base_config_path else None
        self._config_cache = {}
        
        # Auto-detect data directory
        self.data_root = self._find_data_directory()
        
        # Setup dynamic paths
        self.paths = self._setup_dynamic_paths()
        
    def _find_data_directory(self) -> Path:
        """Auto-detect R-DeepONet data directory."""
        candidates = [
            self.project_root / "R-DeepONet_Data",
            self.project_root / "R-DeepONet_Data_1200", 
            self.project_root / "data",
        ]
        
        # Look for existing directories with H5 files
        for candidate in candidates:
            h5_dir = candidate / "data" / "h5"
            if h5_dir.exists() and list(h5_dir.glob("*.h5")):
                print(f"Found data directory: {candidate}")
                return candidate
                
        # If no existing data found, use default
        default_data_dir = self.project_root / "R-DeepONet_Data"
        print(f"Using default data directory: {default_data_dir}")
        return default_data_dir
    
    def _setup_dynamic_paths(self) -> Dict[str, Path]:
        """Setup all dynamic paths based on current environment."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        paths = {
            # Project structure
            'project_root': self.project_root,
            'data_root': self.data_root,
            
            # Data paths
            'h5_data': self.data_root / "data" / "h5",
            'h5_mini': self.data_root / "data" / "h5_mini",
            'images': self.data_root / "data" / "images",
            'check': self.data_root / "data" / "check",
            
            # Experiment paths
            'experiments': self.project_root / "experiments",
            'optuna_root': self.project_root / "experiments" / "optuna",
            'mlruns': self.project_root / "mlruns",
            
            # HPO specific paths
            'hpo_results': self.project_root / "experiments" / "hpo_results" / timestamp,
            'models': self.project_root / "experiments" / "models",
            'logs': self.project_root / "experiments" / "logs",
            'plots': self.project_root / "experiments" / "plots",
            
            # Documents
            'documents': self.project_root / ".trae" / "documents",
        }
        
        # Ensure critical directories exist
        for key in ['experiments', 'optuna_root', 'mlruns', 'hpo_results', 'models', 'logs', 'plots']:
            paths[key].mkdir(parents=True, exist_ok=True)
            
        return paths
    
    def get_path(self, key: str) -> Path:
        """Get a dynamic path by key."""
        if key not in self.paths:
            raise KeyError(f"Path key '{key}' not found. Available keys: {list(self.paths.keys())}")
        return self.paths[key]
    
    def get_storage_url(self, study_name: str) -> str:
        """Get SQLite storage URL for Optuna study."""
        db_path = self.get_path('optuna_root') / f"{study_name}.db"
        return f"sqlite:///{db_path}"
    
    def get_mlflow_uri(self) -> str:
        """Get MLflow tracking URI."""
        mlruns_path = self.get_path('mlruns')
        # Convert Windows path to file URI format
        path_str = str(mlruns_path.absolute()).replace('\\', '/')
        return f"file:///{path_str}"
    
    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Load and process configuration file with dynamic path resolution."""
        if config_path is None:
            config_path = self.base_config_path or (self.project_root / "config_train.yaml")
        
        config_path = Path(config_path)
        
        # Check cache
        cache_key = str(config_path)
        if cache_key in self._config_cache:
            return self._config_cache[cache_key]
        
        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Process dynamic paths
        config = self._process_config_paths(config)
        
        # Cache and return
        self._config_cache[cache_key] = config
        return config
    
    def _process_config_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process configuration to replace hardcoded paths with dynamic ones."""
        # Deep copy to avoid modifying original
        import copy
        config = copy.deepcopy(config)
        
        # Update data paths
        if 'data' in config and 'path' in config['data']:
            # Replace hardcoded data paths
            old_path = config['data']['path']
            if 'h5_mini' in old_path:
                config['data']['path'] = str(self.get_path('h5_mini'))
            else:
                config['data']['path'] = str(self.get_path('h5_data'))
            print(f"Updated data path: {old_path} -> {config['data']['path']}")
        
        # Update output directory if present
        if 'output_dir' in config:
            config['output_dir'] = str(self.get_path('hpo_results'))
        
        return config
    
    def create_hpo_config(self, 
                           study_name: str = "rdeeponet_v2_global_80h",
                           timeout_hours: float = 80.0,
                           base_config: str = "config_train.yaml") -> Dict[str, Any]:
        """Create HPO configuration with dynamic paths and global optimization settings."""

        # Load base config
        base_cfg = self.load_config(base_config)
        hpo_section = base_cfg.get('hpo', {})

        # Sampler configuration
        sampler_raw = hpo_section.get('sampler', {})
        sampler_name = sampler_raw.get('name', 'TPESampler')
        sampler_params = {}
        if 'params' in sampler_raw and isinstance(sampler_raw['params'], dict):
            sampler_params.update(sampler_raw['params'])
        for key, value in sampler_raw.items():
            if key not in {'name', 'params'}:
                sampler_params[key] = value
        sampler_params.setdefault('seed', 42)
        sampler_params.setdefault('multivariate', True)
        sampler_params.setdefault('n_startup_trials', hpo_section.get('n_startup_trials', 50))

        # Pruner configuration (support composite pruner)
        pruner_raw = hpo_section.get('pruner', {})
        raw_primary = pruner_raw.get('primary') or pruner_raw.get('name', 'SuccessiveHalving')
        raw_secondary = pruner_raw.get('secondary')
        shared_params = {}
        if 'params' in pruner_raw and isinstance(pruner_raw['params'], dict):
            shared_params.update(pruner_raw['params'])
        for key, value in pruner_raw.items():
            if key not in {'primary', 'secondary', 'name', 'params'}:
                shared_params[key] = value
        shared_params.setdefault('min_resource', 30)
        shared_params.setdefault('reduction_factor', 3)
        shared_params.setdefault('min_early_stopping_rate', 0)

        def _normalize_pruner(entry, fallback_name):
            if isinstance(entry, dict):
                name = entry.get('name', fallback_name)
                params = {k: v for k, v in entry.items() if k != 'name'}
            else:
                name = entry or fallback_name
                params = {}
            merged = shared_params.copy()
            merged.update(params)
            return {'name': name, 'params': merged}

        primary_entry = _normalize_pruner(raw_primary, 'SuccessiveHalving')
        secondary_entry = None
        if raw_secondary:
            secondary_entry = _normalize_pruner(raw_secondary, 'SuccessiveHalving')

        timeout_hours = float(hpo_section.get('timeout_hours', timeout_hours))
        n_trials = int(hpo_section.get('n_trials', 0) or 0)
        smoke_n_trials = int(hpo_section.get('smoke_n_trials', 0) or 0)

        hpo_config = {
            'study_name': study_name,
            'storage_url': self.get_storage_url(study_name),
            'mlflow_uri': self.get_mlflow_uri(),
            'timeout_hours': timeout_hours,
            'n_trials': n_trials,
            'smoke_n_trials': smoke_n_trials,

            # Enhanced sampling for global optimization
            'sampler': {
                'name': sampler_name,
                'params': sampler_params,
            },

            # ASHA pruner with conservative settings for global search
            'pruner': {
                'primary': primary_entry,
                'secondary': secondary_entry,
            },
            'evaluation': base_cfg.get('evaluation', {}),

            # Paths
            'paths': {
                'study_dir': str(self.get_path('optuna_root') / study_name),
                'results_dir': str(self.get_path('hpo_results')),
                'models_dir': str(self.get_path('models')),
                'logs_dir': str(self.get_path('logs')),
                'plots_dir': str(self.get_path('plots')),
            },
            
            # Base training config
            'base_config': base_cfg,
            
            # Dashboard settings
            'dashboard': {
                'host': '127.0.0.1',
                'port': 8080,
                'auto_start': True
            },
            
            # MLflow settings
            'mlflow': {
                'experiment_name': study_name,
                'auto_log': True,
                'log_models': True
            }
        }
        
        return hpo_config
    
    def save_config(self, config: Dict[str, Any], output_path: Union[str, Path]):
        """Save configuration to file."""
        output_path = Path(output_path)
        
        if output_path.suffix.lower() == '.json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    def get_expanded_search_space(self) -> Dict[str, Any]:
        """Get expanded hyperparameter search space for global optimization."""
        return {
            # Training parameters - expanded ranges
            "batch_size": [2, 4, 6, 8, 12, 16, 20, 24],
            "pts_per_map": [512, 1024, 2048, 4096, 6144, 8192],
            "lr": (1e-5, 1e-2),  # Log uniform
            "weight_decay": (1e-7, 1e-1),  # Log uniform
            "optimizer": ["AdamW"],
            "scheduler": ["WarmupCosine", "CosineAnnealingLR", "OneCycleLR"],

            # Model architecture - expanded
            "final_projection_dim": [64, 96, 128, 192, 256, 320, 384, 512, 640, 768],
            "trunk_hidden": [128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024],
            "trunk_depth": (3, 12),  # Integer range
            "positional_L": (3, 12),  # Integer range
            "dropout": (0.0, 0.5),  # Float range

            # CNN specific - expanded
            "pretrained": [True, False],
            "freeze_layers": ["none", "layer1", "layer1-2", "layer1-3"],

            # Training stability - expanded
            "gradient_clip_val": (0.1, 5.0),  # Float range
            "accumulate_steps": [1, 2, 4, 8],
            "num_workers": [0, 2, 4, 6, 8],

            # Additional parameters for global search
            "epochs": [80, 100, 120, 150, 200],
            "patience": [20, 30, 40, 50],
            "min_delta": (1e-6, 1e-3),  # Log uniform
            
            # Advanced scheduler parameters
            "scheduler_params": {
                "OneCycleLR": {
                    "pct_start": (0.1, 0.5),
                    "div_factor": (10.0, 100.0),
                    "final_div_factor": (100.0, 10000.0)
                },
                "CosineAnnealingLR": {
                    "eta_min": (1e-6, 1e-3)
                },
                "WarmupCosine": {
                    "warmup_epochs": (1, 6),
                    "eta_min": (1e-6, 1e-3)
                }
            }
        }
    
    def validate_environment(self) -> Dict[str, Any]:
        """Validate the current environment and return status."""
        status = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'info': {}
        }
        
        # Check data directory
        if not self.get_path('h5_data').exists():
            status['issues'].append(f"H5 data directory not found: {self.get_path('h5_data')}")
            status['valid'] = False
        else:
            h5_files = list(self.get_path('h5_data').glob("*.h5"))
            status['info']['h5_files_count'] = len(h5_files)
            if len(h5_files) == 0:
                status['warnings'].append("No H5 files found in data directory")
        
        # Check CUDA availability
        try:
            import torch
            status['info']['cuda_available'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                status['info']['cuda_device_count'] = torch.cuda.device_count()
                status['info']['cuda_device_name'] = torch.cuda.get_device_name(0)
            else:
                status['warnings'].append("CUDA not available - training will be slow")
        except ImportError:
            status['issues'].append("PyTorch not installed")
            status['valid'] = False
        
        # Check required packages
        required_packages = ['optuna', 'mlflow', 'yaml', 'numpy', 'matplotlib', 'seaborn']
        missing_packages = []
        for pkg in required_packages:
            try:
                __import__(pkg)
            except ImportError:
                missing_packages.append(pkg)
        
        if missing_packages:
            status['issues'].append(f"Missing required packages: {missing_packages}")
            status['valid'] = False
        
        # Check disk space (basic check)
        import shutil
        free_space_gb = shutil.disk_usage(self.project_root)[2] / (1024**3)
        status['info']['free_disk_space_gb'] = round(free_space_gb, 2)
        if free_space_gb < 10:
            status['warnings'].append(f"Low disk space: {free_space_gb:.1f} GB")
        
        return status


# Global instance for easy access
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Get global ConfigManager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


if __name__ == "__main__":
    # Test the configuration manager
    cm = ConfigManager()
    
    print("=== R-DeepONet Configuration Manager ===")
    print(f"Project root: {cm.project_root}")
    print(f"Data root: {cm.data_root}")
    print(f"H5 data path: {cm.get_path('h5_data')}")
    print(f"Optuna storage: {cm.get_storage_url('test_study')}")
    print(f"MLflow URI: {cm.get_mlflow_uri()}")
    
    # Validate environment
    status = cm.validate_environment()
    print(f"\nEnvironment validation: {'✓ VALID' if status['valid'] else '✗ INVALID'}")
    if status['issues']:
        print("Issues:")
        for issue in status['issues']:
            print(f"  - {issue}")
    if status['warnings']:
        print("Warnings:")
        for warning in status['warnings']:
            print(f"  - {warning}")
    
    print("\nEnvironment info:")
    for key, value in status['info'].items():
        print(f"  {key}: {value}")
