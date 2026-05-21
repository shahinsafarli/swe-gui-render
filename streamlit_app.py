"""
ExperimentMice — Streamlit GUI for the Async Research Assistant.

"""
from __future__ import annotations

import asyncio
import base64
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]

import streamlit as st
import streamlit.components.v1 as components

from src.config import settings
from src.core.researcher import run_research
from src.logging_config import setup_logging
from src.models import ResearchRequest, SourceFilter
from src.storage.repository import init_db, repository

_LOGO_PATH = _ROOT / "assets" / "logo.jpeg"
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mGKHF]")


def _clean(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _logo_b64() -> str | None:
    if _LOGO_PATH.is_file():
        return base64.standard_b64encode(_LOGO_PATH.read_bytes()).decode("ascii")
    return None


def _make_logo_svg(b64: str | None, size: int = 128, ring_width: int = 4) -> str:
    half = size // 2
    r_img = half - ring_width - 2
    r_ring = half - ring_width // 2 - 1

    if b64:
        img_tag = (
            f'<image href="data:image/jpeg;base64,{b64}" '
            f'x="{half - r_img}" y="{half - r_img}" '
            f'width="{r_img * 2}" height="{r_img * 2}" '
            f'clip-path="url(#clip{size})" '
            f'preserveAspectRatio="xMidYMid slice"/>'
        )
    else:
        img_tag = (
            f'<circle cx="{half}" cy="{half}" r="{r_img}" fill="#e8f4ec"/>'
            f'<text x="{half}" y="{half + 10}" text-anchor="middle" font-size="{size // 3}">🐭</text>'
        )

    return f"""<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
  <defs>
    <clipPath id="clip{size}">
      <circle cx="{half}" cy="{half}" r="{r_img}"/>
    </clipPath>
    <linearGradient id="ring{size}" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%"   stop-color="#e8a4b8"/>
      <stop offset="50%"  stop-color="#b8a4d4"/>
      <stop offset="100%" stop-color="#56a379"/>
    </linearGradient>
  </defs>
  <circle cx="{half}" cy="{half}" r="{half - 1}" fill="white"/>
  {img_tag}
  <circle cx="{half}" cy="{half}" r="{r_ring}"
    fill="none"
    stroke="url(#ring{size})"
    stroke-width="{ring_width}"
    stroke-linecap="round"/>
</svg>"""


def _inject_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700;9..144,800&display=swap');

        :root {
            --em-green:        #3d7a5a;
            --em-green-light:  #56a379;
            --em-green-dark:   #1f4a35;
            --em-green-glow:   rgba(61,122,90,0.18);
            --em-blush:        #e8a4b8;
            --em-lavender:     #b8a4d4;
            --em-amber:        #d4a96a;
            --em-cream:        #faf8f5;
            --em-cream2:       #f4f1ec;
            --em-paper:        rgba(255,255,255,0.92);
            --em-ink:          #1e2d27;
            --em-muted:        #5a6b62;
            --em-border:       rgba(61,122,90,0.14);
            --em-radius-sm:    10px;
            --em-radius:       18px;
            --em-radius-lg:    24px;
            --em-shadow-sm:    0 2px 8px rgba(30,45,39,0.06);
            --em-shadow:       0 4px 24px rgba(30,45,39,0.09);
            --em-shadow-lg:    0 8px 40px rgba(30,45,39,0.13);
        }

        /* Base App Overrides */
        .stApp {
            background:
                radial-gradient(ellipse 80% 60% at 10% 0%, rgba(86,163,121,0.08) 0%, transparent 60%),
                radial-gradient(ellipse 60% 50% at 90% 100%, rgba(184,164,212,0.08) 0%, transparent 55%),
                linear-gradient(165deg, #faf8f5 0%, #f3f8f4 40%, #f8f3fa 75%, #faf8f5 100%) !important;
            min-height: 100vh;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(170deg, #f5f0e8 0%, #e8f3ec 60%, #ede8f3 100%) !important;
            border-right: 1px solid var(--em-border);
        }

        /* Typography fixes */
        h1, h2, h3 {
            font-family: 'Fraunces', Georgia, serif !important;
            color: var(--em-green-dark) !important;
        }
        p, label, .stMarkdown p, [data-testid="stWidgetLabel"] p {
            font-family: 'DM Sans', system-ui, sans-serif !important;
            color: var(--em-ink) !important;
        }

        /* --- STREAMLIT COMPONENT FIXES (Fixes black fields & broken text) --- */
        
        /* Selectboxes / Dropdowns */
        [data-baseweb="select"] > div {
            background-color: var(--em-cream) !important;
            border: 1.5px solid var(--em-border) !important;
            border-radius: var(--em-radius-sm) !important;
        }
        [data-baseweb="select"] * {
            color: var(--em-ink) !important;
            font-family: 'DM Sans', sans-serif !important;
        }
        
        /* Text Area / Text Input Box */
        .stTextArea textarea {
            background-color: var(--em-cream) !important;
            color: var(--em-ink) !important;
            border: 1.5px solid var(--em-border) !important;
            border-radius: var(--em-radius-sm) !important;
            font-family: 'DM Sans', sans-serif !important;
        }
        .stTextArea textarea:focus {
            border-color: var(--em-green-light) !important;
            box-shadow: 0 0 0 3px var(--em-green-glow) !important;
        }

        /* MultiSelect Pills */
        [data-baseweb="tag"] {
            background-color: rgba(61,122,90,0.12) !important;
            border: 1px solid rgba(61,122,90,0.2) !important;
            color: var(--em-green-dark) !important;
            border-radius: 999px !important;
        }
        [data-baseweb="tag"] span {
            color: var(--em-green-dark) !important;
        }

        /* Fix buttons in Sidebar and Main Screen */
        .stButton > button {
            font-family: 'DM Sans', sans-serif !important;
            border-radius: 12px !important;
            transition: all 0.2s ease;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, var(--em-green-dark) 0%, var(--em-green) 60%, var(--em-green-light) 100%) !important;
            border: none !important;
            color: #ffffff !important;
            box-shadow: 0 4px 14px rgba(61,122,90,0.35) !important;
        }
        .stButton > button[kind="primary"]:hover {
            box-shadow: 0 6px 20px rgba(61,122,90,0.5) !important;
            transform: translateY(-1px);
        }
        .stButton > button[kind="secondary"] {
            background-color: rgba(255, 255, 255, 0.7) !important;
            color: var(--em-green-dark) !important;
            border: 1px solid var(--em-border) !important;
        }
        .stButton > button[kind="secondary"]:hover {
            background-color: var(--em-cream2) !important;
            border-color: var(--em-green-light) !important;
        }

        /* --- CUSTOM RENDERED BLOCKS --- */
        
        /* Hero */
        .em-hero {
            display: flex;
            align-items: center;
            gap: 2rem;
            padding: 2rem 2.5rem;
            margin-bottom: 1.5rem;
            background: var(--em-paper);
            border-radius: var(--em-radius-lg);
            border: 1px solid var(--em-border);
            box-shadow: var(--em-shadow-lg);
        }
        .em-hero-text h1 {
            margin: 0 0 0.3rem 0 !important;
            font-size: 2.6rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.03em;
            line-height: 1.1;
            background: linear-gradient(135deg, var(--em-green-dark) 20%, var(--em-green-light) 80%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .em-tagline {
            color: var(--em-muted) !important;
            font-size: 1.05rem;
            margin: 0;
            line-height: 1.55;
        }
        .em-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            margin-top: 0.8rem;
            padding: 0.3rem 0.9rem;
            border-radius: 999px;
            font-size: 0.73rem;
            font-weight: 700;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            background: linear-gradient(90deg, var(--em-blush) 0%, var(--em-lavender) 100%);
            color: #fff;
        }

        /* Section headers */
        .em-section-header {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            margin: 1.5rem 0 0.75rem;
        }
        .em-section-header .em-line {
            flex: 1;
            height: 1px;
            background: linear-gradient(90deg, var(--em-border), transparent);
        }
        .em-section-label {
            font-family: 'Fraunces', Georgia, serif !important;
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--em-muted) !important;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }

        /* Answer box */
        .em-answer-box {
            background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(243,248,244,0.95));
            border-radius: var(--em-radius);
            padding: 1.75rem 2rem;
            border: 1px solid rgba(86,163,121,0.2);
            box-shadow: var(--em-shadow);
            line-height: 1.75;
            color: var(--em-ink);
            font-size: 1.02rem;
            margin: 0.75rem 0;
        }

        /* Citations */
        .em-citation {
            display: flex;
            align-items: flex-start;
            gap: 0.75rem;
            padding: 0.75rem 1rem;
            border-radius: var(--em-radius-sm);
            border: 1px solid var(--em-border);
            background: var(--em-cream);
            margin: 0.4rem 0;
        }
        .em-citation-num {
            flex-shrink: 0;
            width: 26px; height: 26px;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--em-green), var(--em-green-light));
            color: #fff;
            font-size: 0.75rem;
            font-weight: 700;
            display: flex; align-items: center; justify-content: center;
        }
        .em-citation-body { flex: 1; min-width: 0; }
        .em-citation-origin {
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: var(--em-green);
            margin-bottom: 0.15rem;
        }
        .em-citation-title {
            font-size: 0.9rem;
            font-weight: 500;
            color: var(--em-ink);
            text-decoration: none;
        }
        .em-citation-title:hover { color: var(--em-green); }

        /* Elapsed */
        .em-elapsed {
            display: inline-flex; align-items: center; gap: 0.4rem;
            padding: 0.3rem 0.8rem;
            border-radius: 999px;
            background: rgba(86,163,121,0.1);
            border: 1px solid rgba(86,163,121,0.2);
            color: var(--em-green-dark);
            font-size: 0.82rem;
            font-weight: 600;
            margin-top: 0.75rem;
        }

        /* History meta row */
        .em-history-meta {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            font-size: 0.8rem;
            color: var(--em-muted);
            margin-bottom: 0.75rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--em-border);
            flex-wrap: wrap;
        }

        /* Empty state */
        .em-empty {
            text-align: center;
            padding: 3rem 2rem;
            color: var(--em-muted);
        }
        .em-empty .em-empty-icon { font-size: 3rem; margin-bottom: 0.75rem; }
        .em-empty p { font-size: 0.95rem; line-height: 1.6; }

        /* Sidebar info key-values */
        .em-sidebar-section {
            font-family: 'Fraunces', Georgia, serif !important;
            font-size: 0.85rem;
            font-weight: 700;
            color: var(--em-green-dark);
            letter-spacing: 0.03em;
            text-transform: uppercase;
            margin: 1.2rem 0 0.6rem;
        }
        .em-kv {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.4rem 0.7rem;
            border-radius: var(--em-radius-sm);
            background: rgba(255,255,255,0.6);
            margin: 0.25rem 0;
            border: 1px solid var(--em-border);
        }
        .em-kv-key  { font-size: 0.78rem; color: var(--em-muted); font-weight: 500; }
        .em-kv-val  { font-size: 0.78rem; color: var(--em-ink);   font-weight: 600; }
        .em-db-pill {
            display: flex; align-items: center; gap: 0.5rem;
            padding: 0.45rem 0.8rem;
            border-radius: 10px;
            font-size: 0.78rem; font-weight: 600;
            margin-top: 0.5rem;
        }
        .em-db-ok  { background: rgba(86,163,121,0.12); border: 1px solid rgba(86,163,121,0.25); color: var(--em-green-dark); }
        .em-db-off { background: rgba(212,169,106,0.12); border: 1px solid rgba(212,169,106,0.25); color: #7a5c20; }
        .em-db-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
        .em-dot-ok  { background: var(--em-green-light); }
        .em-dot-off { background: var(--em-amber); }

        /* Tabs layout tracking */
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            background: transparent;
            border-bottom: 2px solid var(--em-border);
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 12px 12px 0 0;
            padding: 0.5rem 1.3rem;
            font-weight: 600;
            font-size: 0.9rem;
            color: var(--em-muted);
            background: transparent;
            margin-bottom: -2px;
        }
        .stTabs [aria-selected="true"] {
            background: var(--em-paper) !important;
            color: var(--em-green-dark) !important;
            border-color: var(--em-border) !important;
            border-bottom: 2px solid var(--em-paper) !important;
        }

        [data-testid="stExpander"] summary p,
        .streamlit-expanderHeader p {
            font-family: 'DM Sans', system-ui, sans-serif !important;
            font-size: 0.95rem !important;
            font-weight: 500 !important;
            color: var(--em-ink) !important;
            opacity: 1 !important;
        }

        #MainMenu { visibility: hidden; }
        footer    { visibility: hidden; }

        header[data-testid="stHeader"] {
            background: transparent !important;
            height: 3.2rem !important;
        }
        [data-testid="stToolbar"] {
            display: flex !important;
            background: transparent !important;
        }
        [data-testid="stToolbar"] > *:not(:first-child) { display: none !important; }
        [data-testid="stToolbarActions"]                 { display: none !important; }

        [data-testid="collapsedControl"],
        button[kind="header"] {
            width: 38px !important;
            height: 38px !important;
            border-radius: 10px !important;
            border: 1px solid rgba(61,122,90,0.22) !important;
            background: linear-gradient(135deg, #f5f0e8 0%, #e8f3ec 100%) !important;
            box-shadow: 0 2px 10px rgba(30,45,39,0.13) !important;
            color: #1f4a35 !important;
            transition: background 0.2s, box-shadow 0.2s, transform 0.15s !important;
        }
        [data-testid="collapsedControl"]:hover,
        button[kind="header"]:hover {
            background: linear-gradient(135deg, #e8f3ec 0%, #ddeee5 100%) !important;
            box-shadow: 0 4px 16px rgba(61,122,90,0.22) !important;
            transform: scale(1.07) !important;
        }

        [data-testid="stSidebar"] {
            transition: transform 0.32s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
def _render_header() -> None:
    b64 = _logo_b64()
    svg = _make_logo_svg(b64, size=128, ring_width=5)
    svg_data = base64.b64encode(svg.encode()).decode()

    st.markdown(
        f"""
        <div class="em-hero">
            <img src="data:image/svg+xml;base64,{svg_data}"
                 width="128" height="128"
                 style="flex-shrink:0;border-radius:50%;display:block;"
                 alt="ExperimentMice logo"/>
            <div class="em-hero-text">
                <h1>ExperimentMice</h1>
                <p class="em-tagline">
                    Parallel research from Wikipedia, arXiv &amp; the web —
                    woven into one elegant, cited answer.
                </p>
                <span class="em-badge">Async Research Assistant</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def _render_sidebar(b64: str | None) -> None:
    with st.sidebar:
        svg_sm = _make_logo_svg(b64, size=90, ring_width=4)
        svg_sm_data = base64.b64encode(svg_sm.encode()).decode()
        st.markdown(
            f"""<div style="text-align:center;margin-bottom:0.75rem;">
            <img src="data:image/svg+xml;base64,{svg_sm_data}"
                 width="90" height="90"
                 style="border-radius:50%;display:inline-block;"
                 alt="logo"/>
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="em-sidebar-section">Lab Settings</div>', unsafe_allow_html=True)
        log_level = st.selectbox(
            "Log level",
            ["WARNING", "INFO", "DEBUG"],
            index=0,
            help="Lower verbosity keeps the UI cleaner.",
        )
        if st.button("Apply log level", use_container_width=True):
            import logging
            logging.getLogger().setLevel(log_level)
            st.success(f"Logging set to {log_level}")

        st.markdown("<hr style='border:none;border-top:1px solid rgba(61,122,90,0.14);margin:1rem 0;'/>", unsafe_allow_html=True)

        st.markdown('<div class="em-sidebar-section">LLM Config</div>', unsafe_allow_html=True)
        st.markdown(
            f"""<div class="em-kv">
                <span class="em-kv-key">Provider</span>
                <span class="em-kv-val">{settings.llm_provider}</span>
            </div>
            <div class="em-kv">
                <span class="em-kv-key">Model</span>
                <span class="em-kv-val">{settings.llm_model}</span>
            </div>""",
            unsafe_allow_html=True,
        )

        if settings.db_url_is_placeholder:
            st.markdown(
                '<div class="em-db-pill em-db-off"><div class="em-db-dot em-dot-off"></div>History DB not configured</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="em-db-pill em-db-ok"><div class="em-db-dot em-dot-ok"></div>PostgreSQL connected</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<hr style='border:none;border-top:1px solid rgba(61,122,90,0.14);margin:1rem 0;'/>", unsafe_allow_html=True)

        st.markdown('<div class="em-sidebar-section">Quick Prompts</div>', unsafe_allow_html=True)
        examples = [
            "What is photosynthesis?",
            "How does CRISPR gene editing work?",
            "Latest developments in quantum computing",
        ]
        for i, q in enumerate(examples):
            if st.button(q, key=f"example_{i}", use_container_width=True):
                st.session_state["question_input"] = q
                st.rerun()


# ---------------------------------------------------------------------------
# Async helpers
# ---------------------------------------------------------------------------
async def _run_research(request: ResearchRequest):
    await init_db()
    return await run_research(request)


async def _load_history(limit: int) -> list[dict]:
    await init_db()
    return await repository.get_sessions(limit=limit)


def _run_async(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Ask tab
# ---------------------------------------------------------------------------
def _render_ask_tab() -> None:
    st.markdown(
        """<div class="em-section-header">
            <span class="em-section-label">New Experiment</span>
            <div class="em-line"></div>
        </div>""",
        unsafe_allow_html=True,
    )

    question = st.text_area(
        "Research question",
        value=st.session_state.get("question_input", ""),
        height=110,
        placeholder="What is artificial intelligence?",
        max_chars=500,
        key="question_input",
        label_visibility="collapsed",
    )

    col1, col2, col3 = st.columns([2.2, 1, 1.1])
    with col1:
        source_labels = st.multiselect(
            "Sources (empty = all)",
            options=["wiki", "arxiv", "web"],
            default=["wiki", "arxiv", "web"],
        )
    with col2:
        no_cache = st.checkbox("Bypass cache", value=False)
    with col3:
        run_clicked = st.button("Run research", type="primary", use_container_width=True)

    if run_clicked:
        if not question or not question.strip():
            st.error("Please enter a research question.")
        else:
            try:
                if not source_labels or len(source_labels) == 3:
                    sources = None
                else:
                    sources = [SourceFilter(s) for s in source_labels]
                request = ResearchRequest(
                    question=question.strip(),
                    sources=sources,
                    no_cache=no_cache,
                )
            except Exception as exc:
                st.error(f"Invalid request: {exc}")
            else:
                with st.spinner("ExperimentMice are gathering sources..."):
                    try:
                        result = _run_async(_run_research(request))
                        st.session_state["last_result"] = result
                    except Exception as exc:
                        st.error(f"Research failed: {exc}")
                        st.session_state.pop("last_result", None)

    result = st.session_state.get("last_result")
    if result is None:
        st.markdown(
            """<div class="em-empty">
                <div class="em-empty-icon">🔬</div>
                <p>Enter a question above and click <strong>Run research</strong> to begin.<br/>
                   All three sources are queried in parallel.</p>
            </div>""",
            unsafe_allow_html=True,
        )
        return

    if result.cached:
        st.info("Answer served from cache.")

    st.markdown(
        """<div class="em-section-header">
            <span class="em-section-label">Answer</span>
            <div class="em-line"></div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="em-answer-box">{_clean(result.answer)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="em-elapsed">Completed in {result.elapsed_seconds:.2f}s</div>',
        unsafe_allow_html=True,
    )

    if result.sources_failed:
        st.warning("Some sources failed: " + ", ".join(result.sources_failed))

    if result.citations:
        st.markdown(
            """<div class="em-section-header">
                <span class="em-section-label">References</span>
                <div class="em-line"></div>
            </div>""",
            unsafe_allow_html=True,
        )
        citations_html = ""
        for c in result.citations:
            origin = _clean(c.origin)
            title  = _clean(c.title)
            url    = _clean(c.url)
            citations_html += f"""
            <div class="em-citation">
                <div class="em-citation-num">{c.index}</div>
                <div class="em-citation-body">
                    <div class="em-citation-origin">{origin}</div>
                    <a href="{url}" target="_blank" class="em-citation-title">{title}</a>
                </div>
            </div>"""
        st.markdown(citations_html, unsafe_allow_html=True)

    with st.expander("Sources used", expanded=False):
        for s in result.sources_used:
            st.markdown(
                f"**{_clean(s.origin)}** — [{_clean(s.title)}]({_clean(s.url)})  \n"
                f"{_clean(s.snippet[:300])}{'...' if len(s.snippet) > 300 else ''}"
            )

    with st.expander("JSON export", expanded=False):
        st.code(
            json.dumps(
                {
                    "question":        result.question,
                    "answer":          result.answer,
                    "citations":       [c.model_dump() for c in result.citations],
                    "sources_failed":  result.sources_failed,
                    "cached":          result.cached,
                    "elapsed_seconds": result.elapsed_seconds,
                },
                indent=2,
            ),
            language="json",
        )


# ---------------------------------------------------------------------------
# History tab
# ---------------------------------------------------------------------------
def _render_history_tab() -> None:
    col_l, col_r = st.columns([3, 1])
    with col_l:
        st.markdown(
            """<div class="em-section-header">
                <span class="em-section-label">Past Experiments</span>
                <div class="em-line"></div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col_r:
        limit = st.number_input("Sessions", min_value=5, max_value=50, value=10, step=5)

    if st.button("Refresh history", use_container_width=True):
        st.session_state.pop("history_rows", None)

    if "history_rows" not in st.session_state:
        with st.spinner("Loading history..."):
            try:
                st.session_state["history_rows"] = _run_async(_load_history(limit))
            except Exception as exc:
                st.error(f"Could not load history: {exc}")
                st.session_state["history_rows"] = []

    rows = st.session_state.get("history_rows", [])

    if not rows:
        st.markdown(
            """<div class="em-empty">
                <div class="em-empty-icon">📚</div>
                <p>No sessions yet. Run a question on the Ask tab, or configure
                   DATABASE_URL in .env and start PostgreSQL.</p>
            </div>""",
            unsafe_allow_html=True,
        )
        return

    for row in rows:
        sid     = row.get("id", "?")
        ts      = str(row.get("created_at", ""))
        elapsed = row.get("elapsed_seconds", 0)
        cached  = row.get("cached", False)
        q       = row.get("question", "")
        ans     = row.get("answer", "")

        cache_flag = " [cached]" if cached else ""
        q_short    = q[:75] + ("..." if len(q) > 75 else "")
        label      = f"{sid}.  {q_short}{cache_flag}"

        with st.expander(label, expanded=False):
            ts_short = ts[:19].replace("T", " ") if ts else ""
            st.markdown(
                f"""<div class="em-history-meta">
                    <span>{ts_short}</span>
                    <span>|</span>
                    <span>{elapsed:.2f}s</span>
                    {"<span>| cached</span>" if cached else ""}
                </div>""",
                unsafe_allow_html=True,
            )
            st.markdown(_clean(ans))
            failed = row.get("sources_failed")
            if failed:
                if isinstance(failed, str):
                    try:
                        failed = json.loads(failed)
                    except json.JSONDecodeError:
                        pass
                if failed:
                    st.warning(f"Failed sources: {failed}")


# ---------------------------------------------------------------------------
# Swipe-to-toggle sidebar
# ---------------------------------------------------------------------------
def _inject_swipe_js() -> None:
    components.html(
        """
        <script>
        (function() {
            var doc = window.parent.document;

            function getNativeBtn() {
                return doc.querySelector('[data-testid="collapsedControl"]')
                    || doc.querySelector('button[kind="header"]')
                    || doc.querySelector('header button');
            }

            if (!getNativeBtn()) return;

            function isOpen() {
                var sb = doc.querySelector('[data-testid="stSidebar"]');
                if (!sb) return true;
                var style = window.parent.getComputedStyle(sb);
                var tx = new DOMMatrix(style.transform).m41;
                return tx > -100;
            }

            function toggle() {
                var btn = getNativeBtn();
                if (btn) { btn.click(); }
            }

            var touchStartX = 0, touchStartY = 0;
            var SWIPE_MIN   = 55;   
            var ANGLE_MAX   = 40;   

            doc.addEventListener('touchstart', function(e) {
                touchStartX = e.touches[0].clientX;
                touchStartY = e.touches[0].clientY;
            }, { passive: true });

            doc.addEventListener('touchend', function(e) {
                var dx    = e.changedTouches[0].clientX - touchStartX;
                var dy    = e.changedTouches[0].clientY - touchStartY;
                var angle = Math.abs(Math.atan2(dy, dx) * 180 / Math.PI);
                if (angle > ANGLE_MAX && angle < 180 - ANGLE_MAX) return; 
                if (Math.abs(dx) < SWIPE_MIN) return;
                if (dx > 0 && !isOpen()) toggle();  
                if (dx < 0 &&  isOpen()) toggle();  
            }, { passive: true });
        })();
        </script>
        """,
        height=0,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="ExperimentMice",
        page_icon="🐭",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _inject_theme()
    _inject_swipe_js()

    if "log_configured" not in st.session_state:
        setup_logging(level="WARNING")
        st.session_state["log_configured"] = True

    b64 = _logo_b64()
    _render_sidebar(b64)
    _render_header()

    tab_ask, tab_history = st.tabs(["Ask", "History"])

    with tab_ask:
        _render_ask_tab()

    with tab_history:
        _render_history_tab()


if __name__ == "__main__":
    main()
