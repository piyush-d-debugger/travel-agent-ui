"""Streamlit chat frontend for the travel agent deployed on Agent Runtime.

Run locally:
    gcloud auth application-default login
    streamlit run app.py
"""

import asyncio
import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

USER_ID = "streamlit_user"

# Text from these authors is meant for the user. Everything else (router JSON,
# specialist replies) is internal. Same split as your Runner.py.
USER_FACING = {"travel_planning_workflow", "itinerary_builder"}

# A state key appearing in an event's state_delta means a stage finished.
STAGES = {
    "router_decision": "Understood your request",
    "train_options": "Searched trains",
    "hotel_options": "Searched hotels",
    "activity_options": "Searched activities",
    "final_itinerary": "Built the itinerary",
}


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
def setting(name: str, default: str | None = None) -> str | None:
    """Read from Streamlit secrets first, then environment variables."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name, default)


@st.cache_resource
def setup_credentials() -> str:
    """On Streamlit Cloud there is no gcloud login, so read the service account
    key from the [gcp_service_account] table in secrets. Returns a short status
    line for the sidebar. Runs once per app start."""
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        return "key file from environment"

    try:
        table = st.secrets.get("gcp_service_account")
    except Exception:
        table = None
    if not table:
        return "no [gcp_service_account] table in secrets (using default login)"

    info = dict(table)
    if info.get("type") != "service_account":
        return 'the table is missing type = "service_account"'
    missing = [k for k in ("private_key", "client_email", "token_uri") if not info.get(k)]
    if missing:
        return f"the table is missing: {', '.join(missing)}"

    key_file = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(info, key_file)
    key_file.close()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_file.name
    return "service account key loaded"


# --------------------------------------------------------------------------
# Async bridge: the SDK is async, Streamlit is sync. One background event loop
# is kept alive so the cached client is never used from a closed loop.
# --------------------------------------------------------------------------
@st.cache_resource
def get_loop() -> asyncio.AbstractEventLoop:
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    return loop


def run(coro):
    return asyncio.run_coroutine_threadsafe(coro, get_loop()).result()


_DONE = object()


async def _next(agen):
    try:
        return await agen.__anext__()
    except StopAsyncIteration:
        return _DONE


def iter_async(agen):
    """Turn an async generator into a normal generator, one event at a time."""
    while True:
        item = run(_next(agen))
        if item is _DONE:
            return
        yield item


@st.cache_resource(show_spinner="Connecting to the agent...")
def get_agent(resource_name: str, project: str, location: str):
    try:
        import agentplatform  # newer standalone SDK

        client = agentplatform.Client(project=project, location=location)
        return client.runtimes.get(name=resource_name)
    except ImportError:
        import vertexai  # older SDK, deprecated but still works

        client = vertexai.Client(project=project, location=location)
        return client.agent_engines.get(name=resource_name)


# --------------------------------------------------------------------------
# Event helpers. Events arrive as dicts; accept snake_case and camelCase.
# --------------------------------------------------------------------------
def pick(data, *keys):
    if isinstance(data, dict):
        for key in keys:
            if data.get(key) is not None:
                return data[key]
    return None


def event_text(event: dict) -> str:
    parts = pick(pick(event, "content") or {}, "parts") or []
    return "".join(
        p["text"]
        for p in parts
        if isinstance(p, dict) and p.get("text") and not p.get("thought")
    )


def state_keys(event: dict) -> list[str]:
    actions = pick(event, "actions") or {}
    delta = pick(actions, "state_delta", "stateDelta") or {}
    return list(delta.keys())


def typed(text: str):
    """Yield text in word-sized pieces for a typewriter effect."""
    for chunk in re.split(r"(?<=\s)", text):
        if chunk:
            yield chunk
            time.sleep(0.012)


# --------------------------------------------------------------------------
# Progress display
# --------------------------------------------------------------------------
class Progress:
    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.done: list[str] = []
        self.status = st.status("Planning your trip...", expanded=True)
        with self.status:
            self.slot = st.empty()
        self._render()

    def _render(self) -> None:
        lines = [
            f"{'✓' if key in self.done else '·'} {label}"
            for key, label in STAGES.items()
        ]
        self.slot.markdown("\n\n".join(lines))
        elapsed = time.perf_counter() - self.started
        self.status.update(label=f"Planning your trip... {elapsed:.0f}s")

    def mark(self, key: str) -> None:
        if key in STAGES and key not in self.done:
            self.done.append(key)
            self._render()

    def finish(self, ok: bool) -> None:
        elapsed = time.perf_counter() - self.started
        if ok:
            self.status.update(
                label=f"Done in {elapsed:.0f}s", state="complete", expanded=False
            )
        else:
            self.status.update(label="Stopped with an error", state="error")


# --------------------------------------------------------------------------
# Streaming
# --------------------------------------------------------------------------
def stream_reply(agent, question: str, progress: Progress, typewriter: bool, log: list):
    # One new session per request: the workflow is single-turn, so nothing
    # carries over between messages.
    session = run(agent.async_create_session(user_id=USER_ID))
    session_id = pick(session, "id")

    partial_authors: set[str] = set()
    last_author = None
    emitted = False

    events = agent.async_stream_query(
        user_id=USER_ID, session_id=session_id, message=question
    )
    for event in iter_async(events):
        log.append(event)

        for key in state_keys(event):
            progress.mark(key)

        author = pick(event, "author")
        text = event_text(event)
        if not text or author not in USER_FACING:
            continue

        is_partial = bool(pick(event, "partial"))
        if not is_partial and author in partial_authors:
            continue  # already streamed token by token, skip the final repeat

        if emitted and author != last_author:
            yield "\n\n"
        last_author = author
        emitted = True

        if is_partial:
            partial_authors.add(author)
            yield text
        elif typewriter:
            yield from typed(text)
        else:
            yield text


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.set_page_config(page_title="Travel planner", layout="centered")

resource_name = setting("AGENT_RESOURCE_NAME")
if not resource_name:
    st.error(
        "AGENT_RESOURCE_NAME is not set. Copy .env.example to .env and paste "
        "your agent's resource name."
    )
    st.stop()

parts = resource_name.split("/")
if len(parts) != 6 or parts[0] != "projects" or parts[4] != "reasoningEngines":
    st.error(
        "AGENT_RESOURCE_NAME should look like "
        "projects/PROJECT/locations/REGION/reasoningEngines/ENGINE_ID"
    )
    st.stop()

project = setting("GOOGLE_CLOUD_PROJECT") or parts[1]
location = parts[3]

cred_status = setup_credentials()

with st.sidebar:
    st.header("Settings")
    typewriter = st.toggle("Typewriter effect", value=True)
    show_raw = st.toggle("Show raw events", value=False)
    if st.button("New chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption(f"Region: {location}")
    st.caption(f"Agent: {parts[5]}")
    st.caption(f"Credentials: {cred_status}")

st.title("Travel planner")
st.caption("Trains, hotels and activities, planned in one go.")

try:
    agent = get_agent(resource_name, project, location)
except Exception as exc:
    st.error(
        f"Could not connect to the agent: {type(exc).__name__}: {exc}\n\n"
        "If this is a credentials error, run `gcloud auth application-default login`."
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Where do you want to go? e.g. 3 days in Goa from Mumbai")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        progress = Progress()
        log: list = []
        try:
            reply = st.write_stream(
                stream_reply(agent, question, progress, typewriter, log)
            )
            progress.finish(True)
        except Exception as exc:
            progress.finish(False)
            reply = f"Something went wrong: {type(exc).__name__}: {exc}"
            st.error(reply)

        if not reply:
            reply = (
                "The agent finished without a reply. Turn on raw events in the "
                "sidebar to see what came back."
            )
            st.warning(reply)

        if show_raw:
            with st.expander("Raw events"):
                st.json(log)

    st.session_state.messages.append({"role": "assistant", "content": reply})