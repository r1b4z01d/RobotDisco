"""Helpers shared by the rd_* launch files."""
from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

from . import config


def share(package: str, *parts: str) -> str:
    return os.path.join(get_package_share_directory(package), *parts)


def include(package: str, launch_file: str, **launch_arguments) -> IncludeLaunchDescription:
    """Include `<package>/launch/<launch_file>` with string launch arguments (substitutions allowed)."""
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(share(package, "launch", launch_file)),
        launch_arguments=[(k, v) for k, v in launch_arguments.items()],
    )


def cfg(key: str, default=None) -> str:
    """A robot.yaml value as the string a launch argument default wants."""
    v = config.get(key, default)
    return str(v).lower() if isinstance(v, bool) else str(v)
