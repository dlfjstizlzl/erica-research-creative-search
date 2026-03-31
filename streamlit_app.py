"""Streamlit frontend for the creative-search pipeline."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from html import escape
import json
from pathlib import Path
import threading
from typing import Any

import streamlit as st

from config import RESULTS_DIR
from core.utils import load_json, save_json
from pipeline.runner import load_problem_from_file, run_pipeline


st.set_page_config(
    page_title="creative-search",
    page_icon="cs",
    layout="wide",
)


def main() -> None:
    _bootstrap_state()
    _sync_pipeline_status()
    _render_global_style()

    if st.session_state["screen"] == "results":
        _render_results_screen()
    else:
        _render_search_screen()


def _bootstrap_state() -> None:
    defaults = {
        "screen": "search",
        "result": None,
        "selected_collection": "filtered_ideas",
        "selected_idea_id": None,
        "ui_language": "English",
        "pipeline_state": None,
        "pipeline_thread": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    if st.session_state["pipeline_state"] is None:
        st.session_state["pipeline_state"] = _load_pipeline_state_file() or _new_pipeline_state()


def _render_global_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --app-bg: #f7f7f8;
            --surface: #ffffff;
            --surface-2: #fbfbfc;
            --surface-3: #f3f4f6;
            --border: rgba(15, 23, 42, 0.10);
            --border-strong: rgba(15, 23, 42, 0.16);
            --text-primary: #111827;
            --text-secondary: rgba(17, 24, 39, 0.68);
            --text-tertiary: rgba(17, 24, 39, 0.50);
            --accent: #10a37f;
            --accent-soft: rgba(16, 163, 127, 0.10);
            --shadow-soft: 0 1px 2px rgba(15, 23, 42, 0.05);
        }
        html[data-theme="dark"] {
            --app-bg: #0d0d0d;
            --surface: #171717;
            --surface-2: #212121;
            --surface-3: #2a2a2a;
            --border: rgba(255, 255, 255, 0.10);
            --border-strong: rgba(255, 255, 255, 0.16);
            --text-primary: #ececec;
            --text-secondary: rgba(236, 236, 236, 0.72);
            --text-tertiary: rgba(236, 236, 236, 0.56);
            --accent: #19c37d;
            --accent-soft: rgba(25, 195, 125, 0.12);
            --shadow-soft: none;
        }
        @media (prefers-color-scheme: dark) {
            :root {
                --app-bg: #0d0d0d;
                --surface: #171717;
                --surface-2: #212121;
                --surface-3: #2a2a2a;
                --border: rgba(255, 255, 255, 0.10);
                --border-strong: rgba(255, 255, 255, 0.16);
                --text-primary: #ececec;
                --text-secondary: rgba(236, 236, 236, 0.72);
                --text-tertiary: rgba(236, 236, 236, 0.56);
                --accent: #19c37d;
                --accent-soft: rgba(25, 195, 125, 0.12);
                --shadow-soft: none;
            }
        }
        .stApp,
        [data-testid="stAppViewContainer"] {
            background: var(--app-bg);
            color: var(--text-primary);
        }
        [data-testid="stHeader"] {
            background: color-mix(in srgb, var(--app-bg) 88%, transparent);
            backdrop-filter: blur(8px);
            border-bottom: 1px solid var(--border);
        }
        .block-container {
            max-width: 1400px;
            padding-top: 4.5rem;
            padding-bottom: 2rem;
        }
        .app-shell {
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1rem 1.25rem;
            background: var(--surface);
        }
        .subtle {
            color: var(--text-secondary);
            font-size: 0.95rem;
        }
        .idea-card {
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1rem 1.05rem;
            margin-bottom: 0.75rem;
            background: var(--surface);
            box-shadow: var(--shadow-soft);
        }
        .metric-card {
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            background: var(--surface);
            min-height: 110px;
            box-shadow: var(--shadow-soft);
        }
        .list-item-summary {
            color: var(--text-secondary);
            font-size: 0.88rem;
            line-height: 1.45;
            margin-top: -0.15rem;
        }
        .recent-run-meta {
            color: var(--text-tertiary);
            font-size: 0.84rem;
            margin-bottom: 0.2rem;
        }
        .problem-hero {
            border: 1px solid var(--border);
            border-radius: 20px;
            background: linear-gradient(180deg, var(--surface-2) 0%, var(--surface) 100%);
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
            box-shadow: var(--shadow-soft);
        }
        .problem-hero-label {
            color: var(--text-tertiary);
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.4rem;
        }
        .problem-hero-text {
            color: var(--text-primary);
            font-size: 1.15rem;
            line-height: 1.55;
            font-weight: 600;
        }
        .recent-run-problem {
            color: var(--text-primary);
            font-size: 0.98rem;
            line-height: 1.45;
            font-weight: 600;
            margin: 0.15rem 0 0.45rem;
        }
        .detail-shell {
            display: flex;
            flex-direction: column;
            gap: 0.9rem;
        }
        .detail-card {
            border: 1px solid var(--border);
            border-radius: 20px;
            background: var(--surface);
            padding: 1rem 1.05rem;
            box-shadow: var(--shadow-soft);
        }
        .detail-title {
            color: var(--text-primary);
            font-size: 1.45rem;
            line-height: 1.3;
            font-weight: 700;
            margin: 0;
        }
        .detail-meta {
            color: var(--text-secondary);
            font-size: 0.9rem;
            line-height: 1.45;
            margin-top: 0.45rem;
        }
        .detail-description {
            color: var(--text-primary);
            font-size: 1rem;
            line-height: 1.65;
            margin: 0;
        }
        .detail-section-title {
            color: var(--text-primary);
            font-size: 0.9rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            margin-bottom: 0.7rem;
        }
        .detail-kv {
            border: 1px solid var(--border);
            border-radius: 16px;
            background: var(--surface-2);
            padding: 0.8rem 0.9rem;
            min-height: 100%;
        }
        .detail-kv-label {
            color: var(--text-tertiary);
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.4rem;
        }
        .detail-kv-value {
            color: var(--text-primary);
            font-size: 0.95rem;
            line-height: 1.55;
        }
        .score-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 0.7rem;
        }
        .score-card {
            border: 1px solid var(--border);
            border-radius: 16px;
            background: var(--surface-2);
            padding: 0.8rem 0.9rem;
        }
        .score-label {
            color: var(--text-tertiary);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            margin-bottom: 0.35rem;
        }
        .score-value {
            color: var(--text-primary);
            font-size: 1.1rem;
            font-weight: 700;
        }
        .parent-chip-wrap {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
        }
        .parent-chip {
            border: 1px solid var(--border);
            background: var(--surface-2);
            color: var(--text-primary);
            border-radius: 999px;
            padding: 0.35rem 0.7rem;
            font-size: 0.86rem;
        }
        .log-panel {
            border: 1px solid var(--border);
            border-radius: 16px;
            background: var(--surface-2);
            color: var(--text-primary);
            height: 260px;
            overflow-y: auto;
            padding: 0.9rem 1rem;
            font-family: ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, monospace;
            font-size: 0.84rem;
            line-height: 1.5;
            white-space: pre-wrap;
        }
        h1, h2, h3, h4, h5, h6, p, label, span, div {
            color: var(--text-primary);
        }
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stCaptionContainer"],
        .stCaption {
            color: var(--text-secondary);
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 18px;
            box-shadow: var(--shadow-soft);
        }
        div.stButton > button {
            width: 100%;
            border-radius: 14px;
            border: 1px solid var(--border);
            background: var(--surface-2);
            color: var(--text-primary);
            text-align: left;
            padding: 0.8rem 0.95rem;
            font-weight: 500;
            transition: border-color 0.15s ease, background 0.15s ease, transform 0.15s ease;
        }
        div.stButton > button:hover {
            border-color: var(--border-strong);
            background: var(--surface-3);
        }
        div.stButton > button:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 1px var(--accent-soft);
            color: var(--text-primary);
        }
        div.stButton > button[kind="primary"] {
            background: var(--accent);
            border-color: var(--accent);
            color: #ffffff;
            text-align: center;
        }
        div.stButton > button[kind="primary"]:hover {
            filter: brightness(1.03);
        }
        .stTextArea textarea,
        .stTextInput input,
        .stNumberInput input,
        .stSelectbox div[data-baseweb="select"] > div,
        .stMultiSelect div[data-baseweb="select"] > div {
            background: var(--surface) !important;
            color: var(--text-primary) !important;
            border-color: var(--border) !important;
        }
        .stRadio label,
        .stSelectbox label,
        .stTextArea label,
        .stNumberInput label {
            color: var(--text-secondary) !important;
        }
        [data-baseweb="tag"] {
            background: var(--surface-3) !important;
            color: var(--text-primary) !important;
        }
        .stCode, code {
            color: var(--text-primary);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_search_screen() -> None:
    _render_language_toggle()
    st.title("creative-search")
    st.caption(_t("search_caption"))

    hero_left, hero_right = st.columns([1.7, 1.0], gap="large")

    with hero_left:
        st.markdown(f"### {_t('start_new_search')}")
        problem_mode = st.radio(
            _t("problem_source"),
            options=[_t("write_custom_problem"), _t("use_saved_example")],
            index=0,
            horizontal=True,
        )

        default_problem = ""
        if problem_mode == _t("use_saved_example"):
            example_index = st.number_input(
                _t("example_index"),
                min_value=0,
                value=0,
                step=1,
            )
            default_problem = load_problem_from_file(int(example_index))

        problem = st.text_area(
            _t("problem"),
            value=default_problem,
            height=180,
            placeholder=_t("problem_placeholder"),
        )

        run_clicked = st.button(_t("run_pipeline"), type="primary", use_container_width=True)
        if run_clicked:
            if not problem.strip():
                st.error(_t("enter_problem"))
            elif _pipeline_state().get("running"):
                st.warning(_t("pipeline_already_running"))
            else:
                _start_pipeline_run(problem.strip())
                st.rerun()

        _render_pipeline_run_panel_fragment()

    with hero_right:
        st.markdown(f"### {_t('open_saved_result')}")
        run_files = _list_result_files()
        selected_run = st.selectbox(
            _t("saved_result"),
            options=run_files,
            index=0 if run_files else None,
            format_func=lambda path: path.name,
        )
        load_clicked = st.button(_t("open_result"), use_container_width=True)
        if load_clicked and selected_run is not None:
            _load_result_into_state(load_json(selected_run))
            st.rerun()

        if run_files:
            st.markdown(f"### {_t('recent_runs')}")
            for index, summary in enumerate(_load_recent_run_summaries(run_files[:8])):
                with st.container(border=True):
                    st.markdown(
                        f"<div class='recent-run-meta'>{summary['time_text']}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div class='recent-run-problem'>{summary['problem']}</div>",
                        unsafe_allow_html=True,
                    )
                    if st.button(
                        summary["file_name"],
                        key=f"recent-run-{index}-{summary['file_name']}",
                        use_container_width=True,
                    ):
                        _load_result_into_state(summary["result"])
                        st.rerun()
        else:
            st.info(_t("no_saved_runs"))

    st.divider()
    st.markdown(f"### {_t('what_you_will_see')}")
    info_cols = st.columns(4)
    info_cols[0].markdown(
        f"""
        <div class="metric-card">
        <strong>{_t("best_ideas")}</strong><br/>
        <span class="subtle">{_t("best_ideas_caption")}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    info_cols[1].markdown(
        f"""
        <div class="metric-card">
        <strong>{_t("idea_lists")}</strong><br/>
        <span class="subtle">{_t("idea_lists_caption")}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    info_cols[2].markdown(
        f"""
        <div class="metric-card">
        <strong>{_t("idea_detail")}</strong><br/>
        <span class="subtle">{_t("idea_detail_caption")}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    info_cols[3].markdown(
        f"""
        <div class="metric-card">
        <strong>{_t("archive")}</strong><br/>
        <span class="subtle">{_t("archive_caption")}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_results_screen() -> None:
    result = st.session_state.get("result")
    if not result:
        st.session_state["screen"] = "search"
        st.rerun()
        return

    _render_language_toggle()
    top_left, top_right = st.columns([1.0, 0.35])
    with top_left:
        st.title(_t("result_browser"))
        st.markdown(
            f"""
            <div class="problem-hero">
              <div class="problem-hero-label">{_t("problem")}</div>
              <div class="problem-hero-text">{result.get("problem") or ""}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top_right:
        if st.button(_t("new_search"), use_container_width=True):
            st.session_state["screen"] = "search"
            st.rerun()

    _render_pipeline_banner_fragment()
    _render_best_row(result)
    st.divider()

    list_col, detail_col = st.columns([0.95, 1.35], gap="large")

    with list_col:
        _render_result_lists(result)

    with detail_col:
        _render_selected_detail(result)


def _render_best_row(result: dict[str, Any]) -> None:
    st.markdown(f"### {_t('final_bests')}")
    cols = st.columns(3)
    _render_best_card(cols[0], _t("best_practical"), result.get("best_practical"))
    _render_best_card(cols[1], _t("best_balanced"), result.get("best_balanced"))
    _render_best_card(cols[2], _t("best_wild"), result.get("best_wild"))


@st.fragment(run_every=2)
def _render_pipeline_run_panel_fragment() -> None:
    _sync_pipeline_status()
    _render_pipeline_run_panel()


@st.fragment(run_every=2)
def _render_pipeline_banner_fragment() -> None:
    _sync_pipeline_status()
    _render_pipeline_banner()


def _render_pipeline_banner() -> None:
    state = _pipeline_state()
    running = bool(state.get("running"))
    pending_result = state.get("pending_result")
    pipeline_error = state.get("error")

    if not running and not pending_result and not pipeline_error:
        return

    if running:
        started = state.get("started_at")
        started_text = started.strftime("%Y-%m-%d %H:%M:%S") if isinstance(started, datetime) else "-"
        st.info(
            f"{_t('pipeline_running_banner')}  \n"
            f"{_t('problem')}: {state.get('problem') or '-'}  \n"
            f"{_t('started_at')}: {started_text}"
        )
    elif pipeline_error:
        st.error(f"{_t('pipeline_failed')}: {pipeline_error}")
    elif pending_result:
        st.success(_t("pipeline_finished_ready"))

    if pending_result and st.button(
        _t("open_latest_result"),
        key="open-latest-pipeline-result",
        use_container_width=False,
    ):
        _load_result_into_state(pending_result)
        state["pending_result"] = None
        state["error"] = None
        _save_pipeline_state_file(state)
        st.rerun()


def _render_pipeline_run_panel() -> None:
    state = _pipeline_state()
    running = bool(state.get("running"))
    pending_result = state.get("pending_result")
    pipeline_error = state.get("error")
    logs = state.get("logs") or []

    if running:
        started = state.get("started_at")
        started_text = started.strftime("%Y-%m-%d %H:%M:%S") if isinstance(started, datetime) else "-"
        st.info(
            f"{_t('pipeline_running_banner')}  \n"
            f"{_t('problem')}: {state.get('problem') or '-'}  \n"
            f"{_t('started_at')}: {started_text}"
        )
    elif pipeline_error:
        st.error(f"{_t('pipeline_failed')}: {pipeline_error}")
    elif pending_result:
        st.success(_t("pipeline_finished_ready"))

    action_cols = st.columns([0.24, 0.76])
    with action_cols[0]:
        if pending_result and st.button(
            _t("open_latest_result"),
            key="open-latest-pipeline-result-search",
            use_container_width=True,
        ):
            _load_result_into_state(pending_result)
            state["pending_result"] = None
            state["error"] = None
            _save_pipeline_state_file(state)
            st.rerun()

    st.caption(_t("pipeline_logs"))
    log_text = "\n".join(logs[-200:]) if logs else _t("no_logs_yet")
    st.markdown(
        f"<div class='log-panel'>{escape(log_text)}</div>",
        unsafe_allow_html=True,
    )


def _start_pipeline_run(problem: str) -> None:
    state = _pipeline_state()
    state["running"] = True
    state["problem"] = problem
    state["started_at"] = datetime.now()
    state["logs"] = []
    state["error"] = None
    state["pending_result"] = None
    _save_pipeline_state_file(state)

    writer = _StreamlitLogWriter(state)

    def _target() -> None:
        try:
            with redirect_stdout(writer), redirect_stderr(writer):
                result = run_pipeline(problem)
            state["pending_result"] = result
            writer.log("Pipeline run completed.")
            _save_pipeline_state_file(state)
        except Exception as exc:  # noqa: BLE001
            state["error"] = str(exc)
            writer.log(f"ERROR: {exc}")
            _save_pipeline_state_file(state)
        finally:
            state["running"] = False
            writer.flush()
            _save_pipeline_state_file(state)

    thread = threading.Thread(target=_target, daemon=True)
    st.session_state["pipeline_thread"] = thread
    thread.start()


def _sync_pipeline_status() -> None:
    file_state = _load_pipeline_state_file()
    if file_state is not None:
        st.session_state["pipeline_state"] = file_state

    thread = st.session_state.get("pipeline_thread")
    state = _pipeline_state()
    if thread is not None and not thread.is_alive() and state.get("running"):
        state["running"] = False
        _save_pipeline_state_file(state)


class _StreamlitLogWriter:
    def __init__(self, state: Any) -> None:
        self.state = state
        self._buffer = ""

    def write(self, text: str) -> int:
        if not text:
            return 0
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self.log(line)
        return len(text)

    def flush(self) -> None:
        if self._buffer.strip():
            self.log(self._buffer)
        self._buffer = ""

    def log(self, line: str) -> None:
        text = str(line).strip()
        if not text:
            return
        logs = self.state.get("logs") or []
        logs.append(text)
        self.state["logs"] = logs[-400:]
        _save_pipeline_state_file(self.state)


def _new_pipeline_state() -> dict[str, Any]:
    return {
        "running": False,
        "problem": "",
        "started_at": None,
        "logs": [],
        "error": None,
        "pending_result": None,
    }


def _pipeline_state() -> dict[str, Any]:
    state = st.session_state.get("pipeline_state")
    if not isinstance(state, dict):
        state = _new_pipeline_state()
        st.session_state["pipeline_state"] = state
    return state


def _pipeline_state_file() -> Path:
    return RESULTS_DIR / ".streamlit_pipeline_state.json"


def _save_pipeline_state_file(state: dict[str, Any]) -> None:
    serializable = {
        "running": bool(state.get("running")),
        "problem": str(state.get("problem") or ""),
        "started_at": _serialize_datetime(state.get("started_at")),
        "logs": list(state.get("logs") or [])[-400:],
        "error": state.get("error"),
        "pending_result": state.get("pending_result"),
    }
    save_json(_pipeline_state_file(), serializable)


def _load_pipeline_state_file() -> dict[str, Any] | None:
    path = _pipeline_state_file()
    if not path.exists():
        return None
    try:
        payload = load_json(path)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return {
        "running": bool(payload.get("running")),
        "problem": str(payload.get("problem") or ""),
        "started_at": _deserialize_datetime(payload.get("started_at")),
        "logs": list(payload.get("logs") or [])[-400:],
        "error": payload.get("error"),
        "pending_result": payload.get("pending_result"),
    }


def _serialize_datetime(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat()
    return None


def _deserialize_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _render_result_lists(result: dict[str, Any]) -> None:
    collections = [
        ("filtered_ideas", _t("filtered_pool")),
        ("base_ideas", _t("base_ideas")),
        ("mutated_ideas", _t("mutated_ideas")),
        ("combined_ideas", _t("combined_ideas")),
        ("archive", _t("archive")),
    ]

    st.markdown(f"### {_t('result_items')}")
    for key, label in collections:
        items = result.get(key) or []
        count = len(items)
        header = f"{label} ({count})"
        with st.expander(header, expanded=(key == st.session_state["selected_collection"])):
            if not items:
                st.caption(_t("no_items"))
                continue

            if key == "archive":
                for index, item in enumerate(items):
                    idea_id = str(item.get("idea_id") or "-")
                    selected = (
                        st.session_state["selected_collection"] == "archive"
                        and st.session_state["selected_idea_id"] == idea_id
                    )
                    button_label = f"{'●' if selected else '○'} {idea_id}"
                    with st.container(border=True):
                        if st.button(
                            button_label,
                            key=f"archive-{index}-{idea_id}",
                            use_container_width=True,
                        ):
                            st.session_state["selected_collection"] = "archive"
                            st.session_state["selected_idea_id"] = idea_id
                            st.rerun()
                        st.markdown(
                            f"<div class='list-item-summary'>{_archive_summary(item)}</div>",
                            unsafe_allow_html=True,
                        )
                continue

            for index, idea in enumerate(items):
                idea_id = str(idea.get("id") or "-")
                title = str(idea.get("title") or idea_id)
                short = title if len(title) <= 48 else f"{title[:48]}..."
                selected = (
                    st.session_state["selected_collection"] == key
                    and st.session_state["selected_idea_id"] == idea_id
                )
                prefix = "●" if selected else "○"
                meta = []
                if idea.get("origin_type"):
                    meta.append(str(idea.get("origin_type")))
                if idea.get("generation") is not None:
                    meta.append(f"g{idea.get('generation')}")
                if idea.get("strategy_type"):
                    meta.append(str(idea.get("strategy_type")))
                label_text = f"{prefix} {short}"
                if meta:
                    label_text += f"  ·  {' / '.join(meta[:2])}"
                with st.container(border=True):
                    if st.button(
                        label_text,
                        key=f"{key}-{index}-{idea_id}",
                        use_container_width=True,
                    ):
                        st.session_state["selected_collection"] = key
                        st.session_state["selected_idea_id"] = idea_id
                        st.rerun()
                    st.markdown(
                        f"<div class='list-item-summary'>{_idea_summary(idea)}</div>",
                        unsafe_allow_html=True,
                    )


def _render_selected_detail(result: dict[str, Any]) -> None:
    collection_key = st.session_state["selected_collection"]
    selected_id = st.session_state["selected_idea_id"]

    if collection_key == "archive":
        _render_archive_detail(result.get("archive") or [], selected_id)
        return

    ideas = result.get(collection_key) or []
    selected = None
    if selected_id is not None:
        selected = next((idea for idea in ideas if str(idea.get("id") or "") == selected_id), None)
    if selected is None and ideas:
        selected = ideas[0]
        st.session_state["selected_idea_id"] = str(selected.get("id") or "")

    st.markdown(f"### {_t('idea_detail')}")
    if not selected:
        st.info(_t("select_item"))
        return

    title = selected.get("title") or selected.get("id") or "Untitled idea"
    meta = " | ".join(
        part
        for part in [
            f"id={selected.get('id')}" if selected.get("id") else "",
            f"origin={selected.get('origin_type')}" if selected.get("origin_type") else "",
            f"generation={selected.get('generation')}" if selected.get("generation") is not None else "",
            f"model={selected.get('source_model')}" if selected.get("source_model") else "",
            f"persona={selected.get('source_persona')}" if selected.get("source_persona") else "",
        ]
        if part
    )
    st.markdown("<div class='detail-shell'>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="detail-card">
          <div class="detail-title">{title}</div>
          <div class="detail-meta">{meta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="detail-card">
          <div class="detail-section-title">{_t("description")}</div>
          <p class="detail-description">{selected.get("description") or ""}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div class='detail-card'><div class='detail-section-title'>{_t('idea_profile')}</div>",
        unsafe_allow_html=True,
    )
    info_cols = st.columns(2)
    with info_cols[0]:
        _kv(_t("strategy"), selected.get("strategy_type"))
        _kv(_t("persona"), selected.get("persona"))
        _kv(_t("mechanism"), selected.get("mechanism"))
        _kv(_t("target_user"), selected.get("target_user"))
    with info_cols[1]:
        _kv(_t("execution_context"), selected.get("execution_context"))
        _kv(_t("expected_advantage"), selected.get("expected_advantage"))
        _kv(_t("mutation_type"), selected.get("mutation_type"))
        _kv(_t("combination_type"), selected.get("combination_type"))
    st.markdown("</div>", unsafe_allow_html=True)

    parent_ids = selected.get("parent_ids") or []
    if parent_ids:
        parent_markup = "".join(
            f"<span class='parent-chip'>{parent_id}</span>" for parent_id in parent_ids
        )
        st.markdown(
            f"""
            <div class="detail-card">
              <div class="detail-section-title">{_t("parents")}</div>
              <div class="parent-chip-wrap">{parent_markup}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    scores = selected.get("scores") or {}
    if scores:
        ordered = [
            "creativity",
            "problem_fit",
            "novelty",
            "mechanism_clarity",
            "mutation_quality",
            "combination_quality",
            "feasibility",
            "risk",
        ]
        score_markup = []
        for key in [key for key in ordered if key in scores]:
            value = scores[key]
            if isinstance(value, (int, float)):
                score_markup.append(
                    f"""
                    <div class="score-card">
                      <div class="score-label">{key}</div>
                      <div class="score-value">{float(value):.3f}</div>
                    </div>
                    """
                )
        st.markdown(
            f"""
            <div class="detail-card">
              <div class="detail-section-title">{_t("scores")}</div>
              <div class="score-grid">
                {''.join(score_markup)}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander(_t("raw_item_json")):
        st.json(selected)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_archive_detail(records: list[dict[str, Any]], selected_id: str | None) -> None:
    st.markdown(f"### {_t('archive_detail')}")
    selected = None
    if selected_id is not None:
        selected = next((record for record in records if str(record.get("idea_id") or "") == selected_id), None)
    if selected is None and records:
        selected = records[0]
        st.session_state["selected_idea_id"] = str(selected.get("idea_id") or "")
    if not selected:
        st.info(_t("select_archive"))
        return

    st.markdown(
        f"""
        <div class="detail-card">
          <div class="detail-title">{selected.get('idea_id') or '-'}</div>
          <div class="detail-meta">
            origin={selected.get('origin_type', '-')} | generation={selected.get('generation', '-')} | survived={selected.get('survived', False)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.json(selected)


def _render_best_card(container: Any, label: str, idea: dict[str, Any] | None) -> None:
    with container:
        st.markdown(
            "<div class='idea-card'>",
            unsafe_allow_html=True,
        )
        st.markdown(f"**{label}**")
        if not idea:
            st.caption(_t("no_idea_selected"))
            st.markdown("</div>", unsafe_allow_html=True)
            return
        st.markdown(f"### {idea.get('title') or idea.get('id')}")
        st.caption(
            " | ".join(
                part
                for part in [
                    str(idea.get("origin_type") or "").strip(),
                    str(idea.get("strategy_type") or "").strip(),
                    f"id={idea.get('id')}" if idea.get("id") else "",
                ]
                if part
            )
        )
        st.write(idea.get("description") or "")
        scores = idea.get("scores") or {}
        if scores:
            st.caption(_format_scores(scores))
        st.markdown("</div>", unsafe_allow_html=True)


def _kv(label: str, value: Any) -> None:
    if not value:
        return
    st.markdown(
        f"""
        <div class="detail-kv">
          <div class="detail-kv-label">{label}</div>
          <div class="detail-kv-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _format_scores(scores: dict[str, Any]) -> str:
    ordered_keys = [
        "creativity",
        "problem_fit",
        "novelty",
        "mechanism_clarity",
        "mutation_quality",
        "combination_quality",
        "feasibility",
        "risk",
    ]
    parts: list[str] = []
    for key in ordered_keys:
        if key not in scores:
            continue
        value = scores.get(key)
        if isinstance(value, (int, float)):
            parts.append(f"{key}={value:.3f}")
    return " | ".join(parts)


def _idea_summary(idea: dict[str, Any], max_len: int = 120) -> str:
    summary = (
        idea.get("description")
        or idea.get("mechanism")
        or idea.get("expected_advantage")
        or ""
    )
    text = " ".join(str(summary).split())
    if not text:
        return _t("no_summary")
    return _truncate_text(text, max_len)


def _archive_summary(record: dict[str, Any]) -> str:
    parts = [
        f"origin={record.get('origin_type', '-')}",
        f"generation={record.get('generation', '-')}",
        f"survived={record.get('survived', False)}",
    ]
    selected_labels = record.get("selected_labels") or []
    if selected_labels:
        parts.append(f"selected={', '.join(str(label) for label in selected_labels)}")
    return " | ".join(parts)


def _truncate_text(text: str, max_len: int) -> str:
    normalized = " ".join(str(text).split())
    if len(normalized) <= max_len:
        return normalized
    return f"{normalized[:max_len].rstrip()}..."


def _render_language_toggle() -> None:
    left, right = st.columns([0.82, 0.18])
    with right:
        st.selectbox(
            "UI Language",
            options=["English", "한국어"],
            key="ui_language",
            label_visibility="collapsed",
        )


def _t(key: str) -> str:
    language = st.session_state.get("ui_language", "English")
    translations = {
        "English": {
            "search_caption": "Search and explore idea trajectories",
            "start_new_search": "Start a new search",
            "problem_source": "Problem source",
            "write_custom_problem": "Write custom problem",
            "use_saved_example": "Use saved example",
            "example_index": "Example index",
            "problem": "Problem",
            "problem_placeholder": "Describe the problem to explore...",
            "run_pipeline": "Run pipeline",
            "enter_problem": "Enter a problem before running the pipeline.",
            "running_pipeline": "Running creative-search pipeline...",
            "pipeline_completed": "Pipeline run completed.",
            "pipeline_already_running": "A pipeline run is already in progress.",
            "pipeline_running_banner": "Pipeline is currently running.",
            "started_at": "Started at",
            "pipeline_failed": "Pipeline failed",
            "pipeline_finished_ready": "Pipeline finished. The latest result is ready.",
            "refresh_status": "Refresh status",
            "open_latest_result": "Open latest result",
            "pipeline_logs": "Pipeline logs",
            "no_logs_yet": "No logs yet.",
            "open_saved_result": "Open a saved result",
            "saved_result": "Saved result",
            "open_result": "Open result",
            "recent_runs": "Recent runs",
            "no_saved_runs": "No saved runs found in results/.",
            "what_you_will_see": "What you will see",
            "best_ideas": "Best ideas",
            "best_ideas_caption": "Best practical, balanced, and wild candidates.",
            "idea_lists": "Idea lists",
            "idea_lists_caption": "Base, mutated, combined, and filtered ideas as item lists.",
            "idea_detail": "Idea detail",
            "description": "Description",
            "idea_profile": "Idea profile",
            "idea_detail_caption": "Click an item to inspect mechanism, parents, and scores.",
            "archive": "Archive",
            "archive_caption": "Inspect lineage, survival, and final selection markers.",
            "result_browser": "Result browser",
            "new_search": "New search",
            "final_bests": "Final bests",
            "best_practical": "Best Practical",
            "best_balanced": "Best Balanced",
            "best_wild": "Best Wild",
            "filtered_pool": "Filtered Pool",
            "base_ideas": "Base Ideas",
            "mutated_ideas": "Mutated Ideas",
            "combined_ideas": "Combined Ideas",
            "result_items": "Result items",
            "no_items": "No items",
            "select_item": "Select an item from the left list.",
            "strategy": "Strategy",
            "persona": "Persona",
            "mechanism": "Mechanism",
            "target_user": "Target user",
            "execution_context": "Execution context",
            "expected_advantage": "Expected advantage",
            "mutation_type": "Mutation type",
            "combination_type": "Combination type",
            "parents": "Parents",
            "scores": "Scores",
            "raw_item_json": "Raw item JSON",
            "archive_detail": "Archive detail",
            "select_archive": "Select an archive record from the left list.",
            "no_idea_selected": "No idea selected",
            "no_summary": "No summary available.",
        },
        "한국어": {
            "search_caption": "아이디어 탐색 경로를 검색하고 살펴봅니다",
            "start_new_search": "새 탐색 시작",
            "problem_source": "문제 입력 방식",
            "write_custom_problem": "직접 문제 입력",
            "use_saved_example": "저장된 예제 사용",
            "example_index": "예제 인덱스",
            "problem": "문제",
            "problem_placeholder": "탐색할 문제를 입력하세요...",
            "run_pipeline": "파이프라인 실행",
            "enter_problem": "실행 전에 문제를 입력하세요.",
            "running_pipeline": "creative-search 파이프라인 실행 중...",
            "pipeline_completed": "파이프라인 실행이 완료됐습니다.",
            "pipeline_already_running": "이미 실행 중인 파이프라인이 있습니다.",
            "pipeline_running_banner": "파이프라인이 현재 실행 중입니다.",
            "started_at": "시작 시간",
            "pipeline_failed": "파이프라인 실행 실패",
            "pipeline_finished_ready": "파이프라인 실행이 끝났고 최신 결과를 열 수 있습니다.",
            "refresh_status": "상태 새로고침",
            "open_latest_result": "최신 결과 열기",
            "pipeline_logs": "파이프라인 로그",
            "no_logs_yet": "아직 로그가 없습니다.",
            "open_saved_result": "저장된 결과 열기",
            "saved_result": "저장된 결과",
            "open_result": "결과 열기",
            "recent_runs": "최근 실행",
            "no_saved_runs": "results/에 저장된 실행 결과가 없습니다.",
            "what_you_will_see": "볼 수 있는 내용",
            "best_ideas": "최종 베스트",
            "best_ideas_caption": "실용형, 균형형, 파격형 최종 후보를 봅니다.",
            "idea_lists": "아이디어 리스트",
            "idea_lists_caption": "base, mutation, combination, filtered 아이디어를 리스트로 봅니다.",
            "idea_detail": "아이디어 상세",
            "description": "설명",
            "idea_profile": "아이디어 프로필",
            "idea_detail_caption": "아이템을 눌러 메커니즘, 부모, 점수를 확인합니다.",
            "archive": "아카이브",
            "archive_caption": "계보, 생존 여부, 최종 선택 표시를 확인합니다.",
            "result_browser": "결과 브라우저",
            "new_search": "새 탐색",
            "final_bests": "최종 베스트",
            "best_practical": "실용형 베스트",
            "best_balanced": "균형형 베스트",
            "best_wild": "파격형 베스트",
            "filtered_pool": "중복 제거 풀",
            "base_ideas": "기본 아이디어",
            "mutated_ideas": "변형 아이디어",
            "combined_ideas": "조합 아이디어",
            "result_items": "결과 아이템",
            "no_items": "아이템이 없습니다",
            "select_item": "왼쪽 리스트에서 아이템을 선택하세요.",
            "strategy": "전략",
            "persona": "페르소나",
            "mechanism": "메커니즘",
            "target_user": "대상 사용자",
            "execution_context": "실행 맥락",
            "expected_advantage": "기대 장점",
            "mutation_type": "변형 유형",
            "combination_type": "조합 유형",
            "parents": "부모 아이디어",
            "scores": "점수",
            "raw_item_json": "원본 JSON",
            "archive_detail": "아카이브 상세",
            "select_archive": "왼쪽 리스트에서 아카이브 레코드를 선택하세요.",
            "no_idea_selected": "선택된 아이디어가 없습니다",
            "no_summary": "요약이 없습니다.",
        },
    }
    return translations.get(language, translations["English"]).get(key, key)


def _list_result_files() -> list[Path]:
    if not RESULTS_DIR.exists():
        return []
    return sorted(RESULTS_DIR.glob("run_*.json"), reverse=True)


def _load_recent_run_summaries(paths: list[Path]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for path in paths:
        result = load_json(path)
        summaries.append(
            {
                "path": path,
                "file_name": path.name,
                "time_text": _format_run_time(path),
                "problem": _truncate_text(str(result.get("problem") or path.name), 100),
                "result": result,
            }
        )
    return summaries


def _format_run_time(path: Path) -> str:
    stem = path.stem
    if stem.startswith("run_"):
        raw = stem.removeprefix("run_")
        try:
            parsed = datetime.strptime(raw, "%Y%m%d_%H%M%S")
            return parsed.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def _load_result_into_state(result: dict[str, Any]) -> None:
    st.session_state["result"] = result
    st.session_state["screen"] = "results"

    filtered = result.get("filtered_ideas") or []
    if filtered:
        st.session_state["selected_collection"] = "filtered_ideas"
        st.session_state["selected_idea_id"] = str(filtered[0].get("id") or "")
        return

    for key in ("base_ideas", "mutated_ideas", "combined_ideas"):
        ideas = result.get(key) or []
        if ideas:
            st.session_state["selected_collection"] = key
            st.session_state["selected_idea_id"] = str(ideas[0].get("id") or "")
            return

    archive = result.get("archive") or []
    if archive:
        st.session_state["selected_collection"] = "archive"
        st.session_state["selected_idea_id"] = str(archive[0].get("idea_id") or "")


if __name__ == "__main__":
    main()
