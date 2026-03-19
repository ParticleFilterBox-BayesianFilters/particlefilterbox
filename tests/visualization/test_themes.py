"""Tests for visualization themes."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pytest

from particlefilterbox.visualization.themes import (
    THEMES,
    get_colors,
    get_current_theme_name,
    get_theme,
    set_theme,
)


class TestThemes:
    """Tests for theme management."""

    def test_set_theme_nodesecon(self) -> None:
        """set_theme('nodesecon') should not raise."""
        set_theme("nodesecon")
        assert get_current_theme_name() == "nodesecon"

    def test_set_theme_minimal(self) -> None:
        """set_theme('minimal') should not raise."""
        set_theme("minimal")
        assert get_current_theme_name() == "minimal"

    def test_set_theme_paper(self) -> None:
        """set_theme('paper') should not raise."""
        set_theme("paper")
        assert get_current_theme_name() == "paper"

    def test_set_theme_dark(self) -> None:
        """set_theme('dark') should not raise."""
        set_theme("dark")
        assert get_current_theme_name() == "dark"

    def test_set_theme_invalid(self) -> None:
        """set_theme with invalid name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown theme"):
            set_theme("nonexistent")

    def test_get_theme_returns_dict(self) -> None:
        """get_theme should return a dict with 'colors' key."""
        set_theme("nodesecon")
        theme = get_theme()
        assert isinstance(theme, dict)
        assert "colors" in theme

    def test_nodesecon_colors(self) -> None:
        """Nodesecon theme should have the correct institutional colors."""
        assert THEMES["nodesecon"]["colors"] == [
            "#2E86AB",
            "#A23B72",
            "#F18F01",
            "#C73E1D",
            "#3B1F2B",
        ]

    def test_get_colors(self) -> None:
        """get_colors should return the current theme's color list."""
        set_theme("nodesecon")
        colors = get_colors()
        assert len(colors) == 5
        assert colors[0] == "#2E86AB"

    def test_all_themes_have_5_colors(self) -> None:
        """All themes should have exactly 5 colors."""
        for name, theme in THEMES.items():
            assert len(theme["colors"]) == 5, f"Theme '{name}' has {len(theme['colors'])} colors"

    def test_theme_is_copy(self) -> None:
        """get_theme should return a copy, not the original."""
        set_theme("nodesecon")
        theme = get_theme()
        theme["colors"] = []
        original = THEMES["nodesecon"]
        assert len(original["colors"]) == 5
