"""Integrated Logging System for R-DeepONet HPO

This module provides comprehensive logging integration between MLflow and Optuna,
including real-time metrics tracking, model artifacts management, and visualization.

Features:
- Real-time metrics logging to both MLflow and Optuna
- Automatic model artifact management
- Learning curve visualization
- Hyperparameter importance analysis
- Performance dashboard generation
- Experiment comparison tools
"""

from __future__ import annotations
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice,
    plot_contour
)

import torch
import torch.nn as nn
from config_manager import get_config_manager


class IntegratedLogger:
    """Integrated logging system for MLflow and Optuna."""
    
    def __init__(self, study_name: str, experiment_name: Optional[str] = None):
        self.study_name = study_name
        self.experiment_name = experiment_name or study_name
        self.config_manager = get_config_manager()
        
        # Setup MLflow
        self.mlflow_uri = self.config_manager.get_mlflow_uri()
        mlflow.set_tracking_uri(self.mlflow_uri)
        
        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(self.experiment_name)
        except mlflow.exceptions.MlflowException:
            self.experiment_id = mlflow.get_experiment_by_name(self.experiment_name).experiment_id
        
        mlflow.set_experiment(self.experiment_name)
        
        # Setup paths
        self.logs_dir = self.config_manager.get_path('logs')
        self.plots_dir = self.config_manager.get_path('plots')
        self.models_dir = self.config_manager.get_path('models')
        
        # Create subdirectories
        self.study_logs_dir = self.logs_dir / study_name
        self.study_plots_dir = self.plots_dir / study_name
        self.study_models_dir = self.models_dir / study_name
        
        for dir_path in [self.study_logs_dir, self.study_plots_dir, self.study_models_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking
        self.current_run_id = None
        self.metrics_history = {}
        
        print(f"Integrated Logger initialized for study: {study_name}")
        print(f"MLflow URI: {self.mlflow_uri}")
        print(f"Experiment: {self.experiment_name} (ID: {self.experiment_id})")
    
    def start_trial_logging(self, trial: optuna.Trial, params: Dict[str, Any]) -> str:
        """Start logging for a new trial."""
        run_name = f"trial_{trial.number:05d}"
        
        # Start MLflow run
        run = mlflow.start_run(run_name=run_name, nested=False)
        self.current_run_id = run.info.run_id
        
        # Log basic info
        mlflow.log_param("trial_number", trial.number)
        mlflow.log_param("study_name", self.study_name)
        mlflow.log_param("start_time", datetime.now().isoformat())
        
        # Log hyperparameters
        for key, value in params.items():
            mlflow.log_param(key, value)
        
        # Log system info
        self._log_system_info()
        
        # Initialize metrics history for this trial
        self.metrics_history[trial.number] = {
            'epochs': [],
            'train_loss': [],
            'val_mae': [],
            'learning_rate': [],
            'gpu_memory': [],
            'timestamps': []
        }
        
        return self.current_run_id
    
    def log_epoch_metrics(self, trial_number: int, epoch: int, metrics: Dict[str, float]):
        """Log metrics for a specific epoch."""
        timestamp = datetime.now()
        
        # Log to MLflow
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=epoch)
        
        # Store in history
        if trial_number in self.metrics_history:
            history = self.metrics_history[trial_number]
            history['epochs'].append(epoch)
            history['timestamps'].append(timestamp)
            
            # Store specific metrics
            for key in ['train_loss', 'val_mae', 'learning_rate', 'gpu_memory']:
                if key in metrics:
                    history[key].append(metrics[key])
                else:
                    history[key].append(None)
    
    def log_model_artifact(self, model: nn.Module, trial_number: int, 
                          epoch: int, metrics: Dict[str, float]):
        """Log model as artifact with metadata."""
        # Save model locally
        model_dir = self.study_models_dir / f"trial_{trial_number:05d}"
        model_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / f"model_epoch_{epoch:03d}.pt"
        
        # Save model state
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'metrics': metrics,
            'trial_number': trial_number,
            'timestamp': datetime.now().isoformat()
        }, model_path)
        
        # Log to MLflow
        mlflow.log_artifact(str(model_path), "models")
        
        # Save model metadata
        metadata = {
            'trial_number': trial_number,
            'epoch': epoch,
            'metrics': metrics,
            'model_path': str(model_path),
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_path = model_dir / f"metadata_epoch_{epoch:03d}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        mlflow.log_artifact(str(metadata_path), "metadata")
    
    def log_trial_completion(self, trial: optuna.Trial, result: Dict[str, Any]):
        """Log trial completion with final results."""
        # Log final metrics
        mlflow.log_metric("final_mae_db", result.get("best_mae_db", float('inf')))
        mlflow.log_metric("final_epoch", result.get("final_epoch", 0))
        mlflow.log_metric("total_training_time", result.get("training_time", 0))
        
        # Log trial outcome
        mlflow.log_param("trial_state", trial.state.name)
        if trial.state == optuna.trial.TrialState.COMPLETE:
            mlflow.log_metric("trial_success", 1)
        else:
            mlflow.log_metric("trial_success", 0)
        
        # Generate and log learning curves
        if trial.number in self.metrics_history:
            self._generate_learning_curves(trial.number)
        
        # End MLflow run
        mlflow.end_run()
        self.current_run_id = None
    
    def log_trial_failure(self, trial: optuna.Trial, error: Exception):
        """Log trial failure with error information."""
        mlflow.log_param("trial_state", "FAILED")
        mlflow.log_param("error_type", type(error).__name__)
        mlflow.log_param("error_message", str(error))
        mlflow.log_metric("trial_success", 0)
        
        # End MLflow run
        mlflow.end_run()
        self.current_run_id = None
    
    def _log_system_info(self):
        """Log system information."""
        import psutil
        import platform
        
        # System info
        mlflow.log_param("python_version", platform.python_version())
        mlflow.log_param("platform", platform.platform())
        mlflow.log_param("cpu_count", psutil.cpu_count())
        mlflow.log_param("memory_gb", round(psutil.virtual_memory().total / 1e9, 2))
        
        # GPU info
        if torch.cuda.is_available():
            mlflow.log_param("cuda_available", True)
            mlflow.log_param("cuda_device_count", torch.cuda.device_count())
            mlflow.log_param("cuda_device_name", torch.cuda.get_device_name(0))
            mlflow.log_param("cuda_memory_gb", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))
        else:
            mlflow.log_param("cuda_available", False)
    
    def _generate_learning_curves(self, trial_number: int):
        """Generate and log learning curves for a trial."""
        if trial_number not in self.metrics_history:
            return
        
        history = self.metrics_history[trial_number]
        if not history['epochs']:
            return
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Learning Curves - Trial {trial_number}', fontsize=16)
        
        # Training loss
        if any(x is not None for x in history['train_loss']):
            axes[0, 0].plot(history['epochs'], history['train_loss'], 'b-', label='Train Loss')
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True)
            axes[0, 0].legend()
        
        # Validation MAE
        if any(x is not None for x in history['val_mae']):
            axes[0, 1].plot(history['epochs'], history['val_mae'], 'r-', label='Val MAE')
            axes[0, 1].set_title('Validation MAE (dB)')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('MAE (dB)')
            axes[0, 1].grid(True)
            axes[0, 1].legend()
        
        # Learning rate
        if any(x is not None for x in history['learning_rate']):
            axes[1, 0].plot(history['epochs'], history['learning_rate'], 'g-', label='Learning Rate')
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True)
            axes[1, 0].legend()
        
        # GPU memory usage
        if any(x is not None for x in history['gpu_memory']):
            axes[1, 1].plot(history['epochs'], history['gpu_memory'], 'm-', label='GPU Memory')
            axes[1, 1].set_title('GPU Memory Usage (GB)')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Memory (GB)')
            axes[1, 1].grid(True)
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Save and log
        curves_path = self.study_plots_dir / f"learning_curves_trial_{trial_number:05d}.png"
        plt.savefig(curves_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        mlflow.log_artifact(str(curves_path), "plots")
    
    def generate_study_analysis(self, study: optuna.Study) -> Dict[str, str]:
        """Generate comprehensive study analysis and visualizations."""
        analysis_dir = self.study_plots_dir / "analysis"
        analysis_dir.mkdir(exist_ok=True)
        
        artifact_paths = {}
        
        try:
            # 1. Optimization history
            fig = plot_optimization_history(study)
            history_path = analysis_dir / "optimization_history.html"
            fig.write_html(str(history_path))
            artifact_paths['optimization_history'] = str(history_path)
            
            # 2. Parameter importances
            if len(study.trials) > 10:  # Need enough trials for importance
                fig = plot_param_importances(study)
                importance_path = analysis_dir / "parameter_importances.html"
                fig.write_html(str(importance_path))
                artifact_paths['parameter_importances'] = str(importance_path)
            
            # 3. Parallel coordinate plot
            if len(study.trials) > 5:
                fig = plot_parallel_coordinate(study)
                parallel_path = analysis_dir / "parallel_coordinate.html"
                fig.write_html(str(parallel_path))
                artifact_paths['parallel_coordinate'] = str(parallel_path)
            
            # 4. Parameter relationships (slice plot)
            if len(study.trials) > 10:
                fig = plot_slice(study)
                slice_path = analysis_dir / "parameter_slice.html"
                fig.write_html(str(slice_path))
                artifact_paths['parameter_slice'] = str(slice_path)
            
            # 5. Custom analysis plots
            self._generate_custom_analysis(study, analysis_dir, artifact_paths)
            
        except Exception as e:
            print(f"Warning: Some visualizations failed: {e}")
        
        return artifact_paths
    
    def _generate_custom_analysis(self, study: optuna.Study, analysis_dir: Path, 
                                 artifact_paths: Dict[str, str]):
        """Generate custom analysis plots."""
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if len(completed_trials) < 5:
            return
        
        # Extract data
        trial_data = []
        for trial in completed_trials:
            data = {
                'trial_number': trial.number,
                'value': trial.value,
                'duration': trial.duration.total_seconds() if trial.duration else 0,
                **trial.params
            }
            trial_data.append(data)
        
        df = pd.DataFrame(trial_data)
        
        # 1. Performance over time
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['trial_number'], df['value'], 'o-', alpha=0.7)
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('MAE (dB)')
        ax.set_title('Performance Over Time')
        ax.grid(True)
        
        # Add best value line
        best_value = df['value'].min()
        ax.axhline(y=best_value, color='r', linestyle='--', alpha=0.7, label=f'Best: {best_value:.4f}')
        ax.legend()
        
        performance_path = analysis_dir / "performance_over_time.png"
        plt.savefig(performance_path, dpi=300, bbox_inches='tight')
        plt.close()
        artifact_paths['performance_over_time'] = str(performance_path)
        
        # 2. Parameter correlation heatmap
        numeric_params = df.select_dtypes(include=[np.number]).columns
        if len(numeric_params) > 2:
            corr_matrix = df[numeric_params].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Parameter Correlation Matrix')
            
            correlation_path = analysis_dir / "parameter_correlation.png"
            plt.savefig(correlation_path, dpi=300, bbox_inches='tight')
            plt.close()
            artifact_paths['parameter_correlation'] = str(correlation_path)
        
        # 3. Top trials comparison
        top_trials = df.nsmallest(min(10, len(df)), 'value')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(len(top_trials)), top_trials['value'])
        ax.set_xlabel('Rank')
        ax.set_ylabel('MAE (dB)')
        ax.set_title('Top 10 Trials Performance')
        ax.set_xticks(range(len(top_trials)))
        ax.set_xticklabels([f"T{t}" for t in top_trials['trial_number']])
        
        # Color bars by performance
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.xticks(rotation=45)
        
        top_trials_path = analysis_dir / "top_trials.png"
        plt.savefig(top_trials_path, dpi=300, bbox_inches='tight')
        plt.close()
        artifact_paths['top_trials'] = str(top_trials_path)
        
        # 4. Training duration analysis
        if 'duration' in df.columns and df['duration'].sum() > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Duration vs performance
            ax1.scatter(df['duration'] / 3600, df['value'], alpha=0.6)  # Convert to hours
            ax1.set_xlabel('Training Duration (hours)')
            ax1.set_ylabel('MAE (dB)')
            ax1.set_title('Training Duration vs Performance')
            ax1.grid(True)
            
            # Duration histogram
            ax2.hist(df['duration'] / 3600, bins=20, alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Training Duration (hours)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Training Duration Distribution')
            ax2.grid(True)
            
            duration_path = analysis_dir / "duration_analysis.png"
            plt.savefig(duration_path, dpi=300, bbox_inches='tight')
            plt.close()
            artifact_paths['duration_analysis'] = str(duration_path)
    
    def generate_comparison_report(self, studies: List[optuna.Study]) -> str:
        """Generate comparison report between multiple studies."""
        report_dir = self.study_plots_dir / "comparison"
        report_dir.mkdir(exist_ok=True)
        
        # Collect data from all studies
        study_data = []
        for study in studies:
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if completed_trials:
                best_trial = min(completed_trials, key=lambda t: t.value)
                study_data.append({
                    'study_name': study.study_name,
                    'n_trials': len(study.trials),
                    'n_completed': len(completed_trials),
                    'best_value': best_trial.value,
                    'best_trial': best_trial.number,
                    'success_rate': len(completed_trials) / len(study.trials) if study.trials else 0
                })
        
        if not study_data:
            return ""
        
        df = pd.DataFrame(study_data)
        
        # Generate comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Study Comparison Report', fontsize=16)
        
        # Best performance comparison
        axes[0, 0].bar(df['study_name'], df['best_value'])
        axes[0, 0].set_title('Best Performance by Study')
        axes[0, 0].set_ylabel('Best MAE (dB)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Number of trials comparison
        axes[0, 1].bar(df['study_name'], df['n_completed'], alpha=0.7, label='Completed')
        axes[0, 1].bar(df['study_name'], df['n_trials'], alpha=0.3, label='Total')
        axes[0, 1].set_title('Trials by Study')
        axes[0, 1].set_ylabel('Number of Trials')
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Success rate comparison
        axes[1, 0].bar(df['study_name'], df['success_rate'])
        axes[1, 0].set_title('Success Rate by Study')
        axes[1, 0].set_ylabel('Success Rate')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Performance vs trials scatter
        axes[1, 1].scatter(df['n_completed'], df['best_value'])
        for i, study_name in enumerate(df['study_name']):
            axes[1, 1].annotate(study_name, (df['n_completed'].iloc[i], df['best_value'].iloc[i]))
        axes[1, 1].set_xlabel('Completed Trials')
        axes[1, 1].set_ylabel('Best MAE (dB)')
        axes[1, 1].set_title('Performance vs Effort')
        
        plt.tight_layout()
        
        comparison_path = report_dir / "study_comparison.png"
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(comparison_path)
    
    def export_study_data(self, study: optuna.Study) -> Dict[str, str]:
        """Export study data in various formats."""
        export_dir = self.study_logs_dir / "exports"
        export_dir.mkdir(exist_ok=True)
        
        export_paths = {}
        
        # 1. Trials data as CSV
        trials_data = []
        for trial in study.trials:
            data = {
                'trial_number': trial.number,
                'state': trial.state.name,
                'value': trial.value if trial.value is not None else np.nan,
                'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
                'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None,
                'duration_seconds': trial.duration.total_seconds() if trial.duration else np.nan,
                **trial.params,
                **{f'user_attr_{k}': v for k, v in trial.user_attrs.items()}
            }
            trials_data.append(data)
        
        df = pd.DataFrame(trials_data)
        csv_path = export_dir / f"{self.study_name}_trials.csv"
        df.to_csv(csv_path, index=False)
        export_paths['trials_csv'] = str(csv_path)
        
        # 2. Study summary as JSON
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        summary = {
            'study_name': study.study_name,
            'direction': study.direction.name,
            'n_trials': len(study.trials),
            'n_completed_trials': len(completed_trials),
            'n_failed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
            'n_pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            'best_trial': {
                'number': study.best_trial.number,
                'value': study.best_value,
                'params': study.best_params,
                'datetime_complete': study.best_trial.datetime_complete.isoformat() if study.best_trial.datetime_complete else None
            } if completed_trials else None,
            'export_timestamp': datetime.now().isoformat()
        }
        
        json_path = export_dir / f"{self.study_name}_summary.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        export_paths['summary_json'] = str(json_path)
        
        # 3. Best parameters as YAML (for easy reuse)
        if completed_trials:
            import yaml
            yaml_path = export_dir / f"{self.study_name}_best_params.yaml"
            with open(yaml_path, 'w') as f:
                yaml.dump(study.best_params, f, default_flow_style=False)
            export_paths['best_params_yaml'] = str(yaml_path)
        
        return export_paths
    
    def cleanup(self):
        """Cleanup resources."""
        if self.current_run_id:
            try:
                mlflow.end_run()
            except:
                pass
        
        # Clear metrics history to free memory
        self.metrics_history.clear()


class MetricsTracker:
    """Real-time metrics tracking utility."""
    
    def __init__(self, logger: IntegratedLogger):
        self.logger = logger
        self.current_trial = None
        self.epoch_metrics = {}
    
    def start_trial(self, trial: optuna.Trial, params: Dict[str, Any]):
        """Start tracking for a new trial."""
        self.current_trial = trial
        self.logger.start_trial_logging(trial, params)
    
    def log_epoch(self, epoch: int, **metrics):
        """Log metrics for current epoch."""
        if self.current_trial is None:
            return
        
        # Add GPU memory usage if available
        if torch.cuda.is_available():
            metrics['gpu_memory_gb'] = torch.cuda.memory_allocated() / 1e9
        
        self.logger.log_epoch_metrics(self.current_trial.number, epoch, metrics)
        self.epoch_metrics[epoch] = metrics
    
    def log_model(self, model: nn.Module, epoch: int, **metrics):
        """Log model artifact."""
        if self.current_trial is None:
            return
        
        self.logger.log_model_artifact(model, self.current_trial.number, epoch, metrics)
    
    def finish_trial(self, result: Dict[str, Any]):
        """Finish current trial logging."""
        if self.current_trial is None:
            return
        
        self.logger.log_trial_completion(self.current_trial, result)
        self.current_trial = None
        self.epoch_metrics.clear()
    
    def fail_trial(self, error: Exception):
        """Log trial failure."""
        if self.current_trial is None:
            return
        
        self.logger.log_trial_failure(self.current_trial, error)
        self.current_trial = None
        self.epoch_metrics.clear()


# Convenience functions
def create_logger(study_name: str, experiment_name: Optional[str] = None) -> IntegratedLogger:
    """Create an integrated logger instance."""
    return IntegratedLogger(study_name, experiment_name)


def create_metrics_tracker(study_name: str, experiment_name: Optional[str] = None) -> MetricsTracker:
    """Create a metrics tracker instance."""
    logger = create_logger(study_name, experiment_name)
    return MetricsTracker(logger)


if __name__ == "__main__":
    # Test the logging system
    print("Testing Integrated Logging System...")
    
    # Create logger
    logger = create_logger("test_study")
    
    # Test basic functionality
    print(f"Logger created for study: test_study")
    print(f"Logs directory: {logger.study_logs_dir}")
    print(f"Plots directory: {logger.study_plots_dir}")
    print(f"Models directory: {logger.study_models_dir}")
    
    print("Logging system test completed successfully!")