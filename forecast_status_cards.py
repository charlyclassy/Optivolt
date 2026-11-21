"""Presentation-only status and forecast reference cards for Optivolt."""
from __future__ import annotations

from html import escape
import streamlit as st


def _safe(value) -> str:
    return escape(str(value), quote=True)


def render_card(title: str, message: str, *, tone: str = 'blue', reference: str | None = None) -> None:
    """Display a legible forecast information card without changing app state."""
    themes = {
        'blue': ('#173C70', '#FFFFFF', '#C6DAFA', '#3768A2', '#D5E5FA'),
        'green': ('#E5F7EF', '#153F37', '#33685B', '#39B897', '#234A41'),
        'amber': ('#FFF4DC', '#69420C', '#8B5B12', '#E8A33A', '#69420C'),
        'red': ('#FCE9E9', '#761E29', '#8D3841', '#D65A67', '#761E29'),
    }
    bg, fg, muted, accent, body = themes.get(tone, themes['blue'])
    ref_block = (
        f'<div style="margin-top:12px;font-size:0.82rem;font-weight:700;letter-spacing:.035em;'
        f'color:{muted};overflow-wrap:anywhere">FORECAST REF · {_safe(reference)}</div>'
        if reference else ''
    )
    st.markdown(
        f'<div style="box-sizing:border-box;background:{bg};border:1px solid {accent};'
        f'border-left:6px solid {accent};border-radius:16px;padding:19px 22px;'
        f'margin:12px 0 18px;box-shadow:0 5px 16px rgba(17,41,81,.07)">'
        f'<div style="font-size:1.02rem;font-weight:800;line-height:1.3;color:{fg} !important;'
        f'-webkit-text-fill-color:{fg} !important;margin-bottom:7px">{_safe(title)}</div>'
        f'<div style="font-size:0.98rem;line-height:1.6;color:{body} !important;'
        f'-webkit-text-fill-color:{body} !important">{_safe(message)}</div>'
        f'{ref_block}</div>',
        unsafe_allow_html=True,
    )
