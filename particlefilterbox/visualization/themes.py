"""Theme management for particlefilterbox visualizations.

Provides predefined color palettes and matplotlib style configurations.
The 'nodesecon' theme is the institutional default.

Examples
--------
>>> from particlefilterbox.visualization.themes import set_theme, get_theme
>>> set_theme('nodesecon')
>>> theme = get_theme()
>>> print(theme['colors'])
['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
"""

from __future__ import annotations

from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt

_current_theme: str = "nodesecon"

THEMES: dict[str, dict[str, Any]] = {
    "nodesecon": {
        "colors": ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B"],
        "figure.figsize": (10, 6),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "lines.linewidth": 1.8,
        "axes.prop_cycle": None,  # set dynamically
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    },
    "minimal": {
        "colors": ["#333333", "#666666", "#999999", "#BBBBBB", "#DDDDDD"],
        "figure.figsize": (8, 5),
        "axes.grid": False,
        "grid.alpha": 0.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "lines.linewidth": 1.5,
        "axes.prop_cycle": None,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    },
    "paper": {
        "colors": ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442"],
        "figure.figsize": (6, 4),
        "axes.grid": True,
        "grid.alpha": 0.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "lines.linewidth": 1.2,
        "axes.prop_cycle": None,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    },
    "dark": {
        "colors": ["#00D4FF", "#FF6B6B", "#4ECDC4", "#FFE66D", "#A855F7"],
        "figure.figsize": (10, 6),
        "axes.grid": True,
        "grid.alpha": 0.15,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "lines.linewidth": 1.8,
        "axes.prop_cycle": None,
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor": "#16213e",
        "text.color": "#e0e0e0",
        "axes.labelcolor": "#e0e0e0",
        "xtick.color": "#e0e0e0",
        "ytick.color": "#e0e0e0",
        "grid.color": "#e0e0e0",
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    },
}


def set_theme(name: str) -> None:
    """Set the global visualization theme.

    Parameters
    ----------
    name : str
        Theme name. One of 'nodesecon', 'minimal', 'paper', 'dark'.

    Raises
    ------
    ValueError
        If theme name is not recognized.

    Examples
    --------
    >>> set_theme('nodesecon')
    >>> set_theme('paper')
    """
    global _current_theme

    if name not in THEMES:
        available = ", ".join(sorted(THEMES.keys()))
        msg = f"Unknown theme '{name}'. Available: {available}"
        raise ValueError(msg)

    _current_theme = name
    theme = THEMES[name]
    colors = theme["colors"]

    # Build rcParams dict (exclude non-mpl keys)
    rc_params: dict[str, Any] = {}
    for key, value in theme.items():
        if key == "colors" or key == "axes.prop_cycle":
            continue
        rc_params[key] = value

    # Set color cycle
    rc_params["axes.prop_cycle"] = mpl.cycler(color=colors)

    # Apply
    plt.rcParams.update(rc_params)


def get_theme() -> dict[str, Any]:
    """Get the current theme configuration.

    Returns
    -------
    dict[str, Any]
        Theme configuration dictionary including 'colors' list.

    Examples
    --------
    >>> theme = get_theme()
    >>> print(theme['colors'])
    ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
    """
    return THEMES[_current_theme].copy()


def get_colors() -> list[str]:
    """Get the current theme's color palette.

    Returns
    -------
    list[str]
        List of hex color strings.
    """
    return list(THEMES[_current_theme]["colors"])


def get_current_theme_name() -> str:
    """Get the name of the currently active theme.

    Returns
    -------
    str
        Current theme name.
    """
    return _current_theme
