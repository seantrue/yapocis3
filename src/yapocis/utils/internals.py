#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

"""
Path discovery utilities for yapocis package resources.

These functions dynamically discover the yapocis package location, which works
correctly in both development (src layout) and installed environments.

Path Resolution:
    get_yapocis_dir() returns the yapocis package directory by inspecting yapocis.__file__
    This automatically resolves to:
    - Development: /path/to/project/src/yapocis
    - Installed: /path/to/.venv/lib/python3.X/site-packages/yapocis

    All resource lookups (kernels, interfaces, fonts, etc.) are relative to this directory.
"""

import os
import tempfile

import yapocis


def get_yapocis_dir():
    """
    Get the yapocis package directory.

    Returns the absolute path to the yapocis package, which is used as the base
    for all resource discovery (kernels, interfaces, fonts, icons, images).

    This function works correctly in both development and installed environments
    by using yapocis.__file__ to dynamically locate the package.

    Returns:
        str: Absolute path to yapocis package directory

    Example:
        >>> yapocis_dir = get_yapocis_dir()
        >>> interfaces_dir = os.path.join(yapocis_dir, 'interfaces')
    """
    yapocis_dir = os.path.dirname(yapocis.__file__)
    return yapocis_dir


def get_performance_dir() -> str | None:
    performance_dir = os.environ.get("YAPOCIS_PERFORMANCEDIR", None)
    if performance_dir and os.path.isdir(performance_dir):
        return performance_dir
    if os.path.isdir("performance"):
        return os.path.abspath("performance")
    return tempfile.TemporaryDirectory("-performance", prefix="yapocis-").name


def get_viewing_queue() -> str | None:
    viewing_dir = os.environ.get("YAPOCIS_VIEWING", None)
    if viewing_dir and os.path.isdir(viewing_dir):
        return viewing_dir
    else:
        performance_dir = get_performance_dir()
        viewing_dir = os.path.join(performance_dir, "queue")
        os.makedirs(viewing_dir, exist_ok=True)
        return viewing_dir


def get_project_dir() -> str | None:
    yapocis_project = os.environ.get("YAPOCIS_PROJECT", None)
    if yapocis_project and os.path.isdir(yapocis_project):
        return yapocis_project
    else:
        return None


def get_yapocis_dirs() -> list[str]:
    """Directories searched for kernels, interfaces and assets, first match wins:
    $YAPOCIS_PROJECT (if set), the current directory, then the yapocis package."""
    directories = []
    for d in (get_project_dir(), os.getcwd(), get_yapocis_dir()):
        if d and d not in directories:
            directories.append(d)
    return directories


def find_file(dirs: list[str], subdirs: list[str], filename: str) -> str | None:
    for d in dirs:
        for sd in subdirs:
            pth = os.path.join(d, sd, filename)
            if os.path.exists(pth):
                return pth
    return None


def find_yapocis_file(subdirs: list[str], filename: str) -> str | None:
    dirs = get_yapocis_dirs()
    tfilename = find_file(dirs, subdirs, filename)
    return tfilename


class AssetManager:
    def __init__(
        self, dirname, ext, extensions=None, match=None, recursive=False, default=None
    ):
        self.dirname = dirname
        self.extensions = [] if extensions is None else extensions
        if ext:
            self.extensions.append(ext)
        assert match in (None, "exact", "name", "lower", "partial")
        self.match = match
        self.recursive = recursive
        self.default = default
        self.cached_asset_dirs = None
        self.cached_assets = None

    def get_dirs(self) -> list[str]:
        if self.cached_asset_dirs:
            return self.cached_asset_dirs
        base_dir_name = os.path.basename(self.dirname)
        yds = get_yapocis_dirs()
        yds = [os.path.join(yd, base_dir_name) for yd in yds]
        yds.append(self.dirname)
        yds = list(set([yd for yd in yds if os.path.isdir(yd)]))
        self.cached_asset_dirs = yds
        return yds

    def assets(self) -> list[str]:
        """
        list available assets in the accessible directories
        Returns:

        """

        def has_ext(s):
            if not self.extensions:
                return s
            _, ext = os.path.splitext(s)
            return ext in self.extensions

        if self.cached_assets:
            return list(self.cached_assets.keys())
        asset_dirs = self.get_dirs()
        assets = {}
        for asset_dir in asset_dirs:
            if self.recursive:
                for subdir_name, dir_names, file_names in os.walk(asset_dir):
                    assets.update(
                        {
                            asset_name: os.path.join(asset_dir, subdir_name, asset_name)
                            for asset_name in file_names
                            if has_ext(asset_name)
                        }
                    )
            else:
                for asset_dir in asset_dirs:
                    assets.update(
                        {
                            asset_name: os.path.join(asset_dir, asset_name)
                            for asset_name in os.listdir(asset_dir)
                            if has_ext(asset_name)
                        }
                    )
        self.cached_assets = assets
        return list(self.cached_assets.keys())

    def asset_path(self, asset_name: str) -> str:
        """
        Return absolute path to an asset
        Args:
            asset_name: as returned by assets

        Returns:

        """
        try:
            return self.cached_assets[asset_name]
        except:
            raise ValueError(
                f"{asset_name} not found, use one of: {' '.join(self.assets())}"
            )

    def asset(self, asset_suggestion: str) -> str:
        """
        Return absolute path to asset that's a full or partial match
        Args:
            asset_suggestion: first try exact match, then with .ext added, then lowercase, then substring

        Returns: path to a asset, with a default

        """
        try:
            return self.asset_path(asset_suggestion)
        except ValueError:
            asset_names = self.assets()
            if asset_suggestion in asset_names:
                return self.asset_path(asset_suggestion)
            if self.match == "name":
                for ext in self.extensions:
                    fqn = f"{asset_suggestion}{ext}"
                    if fqn in asset_names:
                        return self.asset_path(fqn)
            if self.match == "partial":
                for asset_name in asset_names:
                    if asset_suggestion in asset_name:
                        return self.asset_path(asset_name)
            if self.match == "lower":
                for asset_name in asset_names:
                    if asset_suggestion.lower() in asset_name.lower():
                        return self.asset_path(asset_name)
            if self.default:
                return self.asset_path(self.default)
            asset_names = [an.rsplit(".", 1)[0] for an in self.assets()]
            raise ValueError(
                f"{asset_suggestion} not found, use one of: {' '.join(asset_names)}"
            )
