"""
Model asset manager for yapocis.

Provides access to pre-trained neural network models and their associated files
(weights, class labels, config files, etc.).

Example:
    from yapocis.models import find_model, get_model_path

    # Get path to model directory
    vgg16_dir = find_model('vgg16')

    # Get specific model file
    weights_path = get_model_path('vgg16', 'weights.npz')
    classes_path = get_model_path('vgg16', 'classes.txt')
"""
#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2025.

import os
from pathlib import Path
from typing import Optional, Dict, List

class ModelAssetManager:
    """
    Asset manager for neural network models.

    Each model is stored in its own directory containing all necessary files:
    - weights.npz: Model weights
    - classes.txt: Class labels (for classification models)
    - config.json: Model configuration (optional)
    - README.md: Model documentation (optional)
    """

    def __init__(self, base_dir: str, project_dir: Optional[str] = None):
        """
        Initialize model asset manager.

        Args:
            base_dir: Base directory containing model subdirectories
            project_dir: Optional project directory for additional models
        """
        self.base_dir = base_dir
        self.project_dir = project_dir
        self._model_cache: Dict[str, Path] = {}
        self._scan_models()

    def _scan_models(self) -> None:
        """Scan for available models."""
        base_path = Path(self.base_dir)
        if base_path.exists():
            for item in base_path.iterdir():
                if item.is_dir() and not item.name.startswith('.') and not item.name.startswith('__'):
                    self._model_cache[item.name] = item

        # Also check project directory if specified
        if self.project_dir:
            project_path = Path(self.project_dir)
            if project_path.exists():
                for item in project_path.iterdir():
                    if item.is_dir() and not item.name.startswith('.') and not item.name.startswith('__'):
                        # Project models override base models
                        self._model_cache[item.name] = item

    def list_models(self) -> List[str]:
        """
        List all available models.

        Returns:
            List of model names
        """
        return sorted(self._model_cache.keys())

    def find_model(self, name: str) -> Optional[Path]:
        """
        Find a model directory by name.

        Args:
            name: Model name

        Returns:
            Path to model directory, or None if not found
        """
        return self._model_cache.get(name)

    def get_model_path(self, model_name: str, filename: str) -> Optional[Path]:
        """
        Get path to a specific file within a model directory.

        Args:
            model_name: Model name
            filename: Filename within model directory (e.g., 'weights.npz', 'classes.txt')

        Returns:
            Path to file, or None if model or file not found
        """
        model_dir = self.find_model(model_name)
        if model_dir is None:
            return None

        file_path = model_dir / filename
        if file_path.exists():
            return file_path

        return None

    def get_model_files(self, model_name: str) -> Optional[Dict[str, Path]]:
        """
        Get all files in a model directory.

        Args:
            model_name: Model name

        Returns:
            Dictionary mapping filename to path, or None if model not found
        """
        model_dir = self.find_model(model_name)
        if model_dir is None:
            return None

        files = {}
        for file_path in model_dir.iterdir():
            if file_path.is_file() and not file_path.name.startswith('.'):
                files[file_path.name] = file_path

        return files

    def model_exists(self, model_name: str) -> bool:
        """
        Check if a model exists.

        Args:
            model_name: Model name

        Returns:
            True if model exists, False otherwise
        """
        return model_name in self._model_cache

    def get_weights_path(self, model_name: str) -> Optional[Path]:
        """
        Convenience method to get model weights file.

        Args:
            model_name: Model name

        Returns:
            Path to weights.npz file, or None if not found
        """
        return self.get_model_path(model_name, 'weights.npz')

    def get_classes_path(self, model_name: str) -> Optional[Path]:
        """
        Convenience method to get model class labels file.

        Args:
            model_name: Model name

        Returns:
            Path to classes.txt file, or None if not found
        """
        return self.get_model_path(model_name, 'classes.txt')


# Create global model asset manager instance
_model_manager = ModelAssetManager(
    os.path.join(os.path.dirname(__file__), ''),
    os.environ.get('YAPOCIS_MODELS')
)

# Public API
models = _model_manager.list_models
find_model = _model_manager.find_model
get_model_path = _model_manager.get_model_path
get_model_files = _model_manager.get_model_files
model_exists = _model_manager.model_exists
get_weights_path = _model_manager.get_weights_path
get_classes_path = _model_manager.get_classes_path
