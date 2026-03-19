"""Figure export utilities for particlefilterbox.

Provides convenience functions for saving matplotlib figures in
multiple formats (PNG, PDF, SVG).

Examples
--------
>>> import matplotlib.pyplot as plt
>>> fig, ax = plt.subplots()
>>> ax.plot([1, 2, 3])
>>> save_figure(fig, 'output.png', format='png')
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def save_figure(
    fig: Figure,
    path: str | Path,
    fmt: str | None = None,
    dpi: int | None = None,
    **kwargs: Any,
) -> Path:
    """Save a matplotlib figure to file.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    path : str or Path
        Output file path. Format is inferred from extension if not specified.
    format : str or None
        Output format ('png', 'pdf', 'svg', 'eps'). If None, inferred from path.
    dpi : int or None
        Resolution in dots per inch. If None, uses theme default.
    **kwargs
        Additional keyword arguments passed to `fig.savefig()`.

    Returns
    -------
    Path
        The resolved output path.

    Raises
    ------
    ValueError
        If format cannot be determined from path or arguments.

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3])
    >>> save_figure(fig, 'plot.png')
    PosixPath('plot.png')
    >>> save_figure(fig, 'plot', format='pdf')
    PosixPath('plot.pdf')
    """
    path = Path(path)

    if fmt is None:
        if path.suffix:
            fmt = path.suffix.lstrip(".")
        else:
            msg = "Cannot determine format. Provide 'format' or use a file extension."
            raise ValueError(msg)

    # Ensure correct extension
    expected_suffix = f".{fmt}"
    if path.suffix != expected_suffix:
        path = path.with_suffix(expected_suffix)

    # Create parent directories
    path.parent.mkdir(parents=True, exist_ok=True)

    save_kwargs: dict[str, Any] = {
        "format": fmt,
        "bbox_inches": "tight",
    }
    if dpi is not None:
        save_kwargs["dpi"] = dpi

    save_kwargs.update(kwargs)
    fig.savefig(str(path), **save_kwargs)
    plt.close(fig)

    return path
