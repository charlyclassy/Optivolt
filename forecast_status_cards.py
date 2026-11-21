"""Optivolt forecast-status cards.

Only presentation; no forecast storage, calculations, or application state changed.
Blue cards are rendered in a self-contained Streamlit component to prevent global
app CSS from overriding light text on their dark background.
"""
from __future__ import annotations

from html import escape
import streamlit as st
import streamlit.components.v1 as components


def _safe(value) -> str:
    return escape(str(value), quote=True)


def _render_isolated_blue_card(title: str, message: str, reference: str | None) -> None:
    """Isolate the reference card from surrounding app-wide CSS."""
    reference_html = (
        '<div class="reference"><span class="ref-label">FORECAST REF</span>'
        f'<span class="ref-value"> · {_safe(reference)}</span></div>'
        if reference else ''
    )
    markup = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
* {{box-sizing:border-box;}}
html,body {{margin:0;padding:0;background:transparent;color:#fff;}}
body {{font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;}}
.card {{background:#173C70;border:1px solid #477FC3;border-left:7px solid #477FC3;
  border-radius:18px;padding:20px 24px;min-height:148px;overflow-wrap:anywhere;}}
.title, .title * {{color:#FFFFFF !important;-webkit-text-fill-color:#FFFFFF !important;
  font-weight:800;font-size:1.06rem;line-height:1.4;}}
.title {{margin:0 0 10px;}}
.body, .body * {{color:#EAF3FF !important;-webkit-text-fill-color:#EAF3FF !important;
  font-size:0.99rem;line-height:1.55;}}
.reference,.reference * {{color:#C9DFFF !important;-webkit-text-fill-color:#C9DFFF !important;
  font-size:.84rem;font-weight:750;letter-spacing:.025em;}}
.reference {{margin-top:14px;}}
.version {{display:block;margin-top:8px;color:#A8CBFB;font-size:.66rem;letter-spacing:.02em;}}
@media(max-width:600px) {{.card {{padding:16px 17px;}} .body {{font-size:.94rem;}}}}
</style></head><body>
<section class="card" aria-label="{_safe(title)}">
 <div class="title">{_safe(title)}</div>
 <div class="body">{_safe(message)}</div>
 {reference_html}
 <span class="version">Optivolt status card · contrast v3</span>
</section>
</body></html>'''
    # A generous fixed component height prevents clipping at narrow viewport sizes.
    # The component is iframe-isolated, so site-wide CSS cannot turn its text dark.
    components.html(markup, height=232, scrolling=False)


def render_card(title: str, message: str, *, tone: str = 'blue', reference: str | None = None) -> None:
    """Render a forecast/status card, with consistent semantics and contrast."""
    if tone == 'blue' or tone not in ('green', 'amber', 'red'):
        _render_isolated_blue_card(title, message, reference)
        return
    themes = {
        'green': ('#E5F7EF', '#153F37', '#234A41', '#39B897'),
        'amber': ('#FFF4DC', '#69420C', '#69420C', '#E8A33A'),
        'red': ('#FCE9E9', '#761E29', '#761E29', '#D65A67'),
    }
    bg, title_color, text_color, accent = themes[tone]
    ref_markup = (
        f'<p style="margin:12px 0 0;color:{text_color} !important;'
        f'-webkit-text-fill-color:{text_color} !important;font-weight:700">'
        f'FORECAST REF · {_safe(reference)}</p>' if reference else ''
    )
    st.markdown(
        f'<div style="background:{bg};border:1px solid {accent};border-left:6px solid {accent};'
        f'border-radius:16px;padding:19px 22px;margin:12px 0 18px">'
        f'<div style="font-size:1.02rem;font-weight:800;line-height:1.35;'
        f'color:{title_color} !important;-webkit-text-fill-color:{title_color} !important">'
        f'<span style="color:{title_color} !important;-webkit-text-fill-color:{title_color} !important">'
        f'{_safe(title)}</span></div>'
        f'<div style="font-size:.98rem;line-height:1.6;margin-top:9px;'
        f'color:{text_color} !important;-webkit-text-fill-color:{text_color} !important">'
        f'<span style="color:{text_color} !important;-webkit-text-fill-color:{text_color} !important">'
        f'{_safe(message)}</span></div>{ref_markup}</div>',
        unsafe_allow_html=True,
    )
