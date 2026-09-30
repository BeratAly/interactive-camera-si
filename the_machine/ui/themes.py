"""Theme palettes + global QSS (§20/§22). Dark, minimal, technical."""
from __future__ import annotations

from dataclasses import dataclass

ACCENTS = {
    "green": "#3ddc84",
    "amber": "#ffb454",
    "cyan": "#4fd8e8",
}


@dataclass(frozen=True)
class Theme:
    name: str
    accent: str
    accent_dim: str          # accent at ~45% (hex alpha suffix)
    bg: str = "#07090b"
    panel: str = "#0d1114"
    panel_border: str = "#1b2426"
    text: str = "#c9d1d4"
    text_dim: str = "#5c6a6d"
    ok: str = "#3ddc84"
    warn: str = "#ffb454"
    err: str = "#ff5d5d"

    @staticmethod
    def by_name(name: str) -> "Theme":
        accent = ACCENTS.get(name, ACCENTS["green"])
        return Theme(name=name if name in ACCENTS else "green",
                     accent=accent, accent_dim=accent + "73")


def build_qss(theme: Theme) -> str:
    t = theme
    return f"""
QMainWindow, QWidget {{
    background-color: {t.bg};
    color: {t.text};
    font-family: "Consolas", "Cascadia Mono", "DejaVu Sans Mono", monospace;
    font-size: 13px;
}}
QFrame#Panel {{
    background-color: {t.panel};
    border: 1px solid {t.panel_border};
    border-radius: 4px;
}}
QLabel#Title {{
    color: {t.accent};
    font-size: 17px;
    font-weight: bold;
    letter-spacing: 3px;
}}
QLabel#Subtle {{ color: {t.text_dim}; }}
QLabel#SectionHeader {{
    color: {t.accent};
    font-weight: bold;
    letter-spacing: 2px;
    font-size: 11px;
    padding: 2px;
}}
QListWidget#EventLog {{
    background-color: {t.panel};
    border: 1px solid {t.panel_border};
    border-radius: 4px;
    color: {t.text};
    font-size: 12px;
    padding: 4px;
}}
QListWidget#EventLog::item {{ padding: 2px 4px; }}
QLineEdit {{
    background-color: {t.panel};
    border: 1px solid {t.panel_border};
    border-radius: 4px;
    padding: 6px;
    color: {t.text};
    selection-background-color: {t.accent};
    selection-color: {t.bg};
}}
QPushButton {{
    background-color: {t.panel};
    border: 1px solid {t.accent_dim};
    border-radius: 4px;
    padding: 5px 12px;
    color: {t.accent};
    font-weight: bold;
}}
QPushButton:hover {{ background-color: #14201a; border-color: {t.accent}; }}
QPushButton:pressed {{ background-color: {t.accent}; color: {t.bg}; }}
QScrollBar:vertical {{
    background: {t.bg}; width: 8px; margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {t.panel_border}; border-radius: 4px; min-height: 24px;
}}
QStatusBar {{ background: {t.panel}; color: {t.text_dim}; }}
"""
