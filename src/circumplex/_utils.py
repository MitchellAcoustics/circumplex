import importlib.util


def is_package_installed(package_name: str) -> bool:
    """Check if a package is installed."""
    package_spec = importlib.util.find_spec(package_name)
    return package_spec is not None


if is_package_installed("rich"):
    from rich.console import Console

    console = Console()
