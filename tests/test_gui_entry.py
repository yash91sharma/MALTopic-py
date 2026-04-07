"""Tests for GUI entry points – main() function and __main__.py."""

from unittest.mock import patch

import pytest


class TestMainFunction:
    def test_main_is_callable(self):
        from src.maltopic.gui import main

        assert callable(main)

    @patch("src.maltopic.gui.subprocess.run")
    def test_main_invokes_streamlit(self, mock_run):
        from src.maltopic.gui import main

        main()
        mock_run.assert_called_once()
        args = mock_run.call_args
        cmd = args[0][0]  # First positional arg is the command list
        assert "-m" in cmd
        assert "streamlit" in cmd
        assert "run" in cmd
        # Should point to app.py
        assert any("app.py" in str(c) for c in cmd)
        # Should set max upload size
        assert any("maxUploadSize" in str(c) for c in cmd)

    @patch("src.maltopic.gui.subprocess.run")
    def test_main_disables_usage_stats(self, mock_run):
        from src.maltopic.gui import main

        main()
        cmd = mock_run.call_args[0][0]
        assert any("gatherUsageStats" in str(c) for c in cmd)


class TestDunderMain:
    def test_dunder_main_exists(self):
        """The __main__.py file should exist and be importable as a concept."""
        import importlib.util

        spec = importlib.util.find_spec("maltopic.gui.__main__")
        # The module exists on disk even if we can't safely import it
        # (importing it triggers main() which we don't want here)
        assert spec is not None


class TestStylesModule:
    def test_theme_css_is_string(self):
        from src.maltopic.gui.styles import THEME_CSS

        assert isinstance(THEME_CSS, str)
        assert "<style>" in THEME_CSS

    def test_inject_css_callable(self):
        from src.maltopic.gui.styles import inject_css

        assert callable(inject_css)
