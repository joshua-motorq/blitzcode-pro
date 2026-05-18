import asyncio
import contextlib
import json
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

# Load provider config EARLY before SDK imports (SDK reads env at import time)
_agent_webkit_root = Path.home() / ".agent-webkit"
_blitz_pro_root = Path(
    os.environ.get("BLITZCODE_PRO_ROOT", str(_agent_webkit_root / "blitzcode-pro"))
)
_provider_config_path = _blitz_pro_root / "provider-config.json"
if _provider_config_path.exists() and not os.environ.get("ANTHROPIC_API_KEY"):
    try:
        with open(_provider_config_path) as f:
            config = json.load(f)
            if config.get("api_key"):
                os.environ["ANTHROPIC_API_KEY"] = config["api_key"]
            if config.get("base_url"):
                os.environ["ANTHROPIC_API_BASE"] = config["base_url"]
    except (json.JSONDecodeError, OSError, KeyError):
        pass

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from agent_webkit_server.adapters.fastapi import create_app
from agent_webkit_server.auth import AuthConfig
from agent_webkit_server.session import SessionConfig
from agent_webkit_server.session_metadata import FileSessionMetadataStore

from acks import AckStore
from discovery import discover_for_cwd
from builtin_agents import BUILTIN_AGENTS
from initiatives import Initiative, InitiativeStore, is_valid_key as is_valid_initiative_key
from workspaces import (
    DocRef,
    RepoSpec,
    Workspace,
    WorkspaceRepo,
    WorkspaceStore,
    create_workspace,
    delete_workspace,
    is_valid_ticket_key,
)
from workflow_prompt import render_workflow_prompt
from atlassian_creds import CredsStore
from atlassian.jira import JiraClient, JiraError
from atlassian.confluence import ConfluenceClient, ConfluenceError
from settings import SettingsStore
from workflow_mcp.workflow import (
    ALLOWED_TOOL_PATTERNS as WORKFLOW_ALLOWED_TOOLS,
    WorkflowDeps,
    build_workflow_mcp_server,
)
import git_worktree as gw

# ────────────────────────────────────────────────────────────────────────────
# Paths — all blitzcode-pro state under one root so uninstall is `rm -rf`.
# ────────────────────────────────────────────────────────────────────────────

_AGENT_WEBKIT_ROOT = Path.home() / ".agent-webkit"
_BLITZ_PRO_ROOT = Path(
    os.environ.get("BLITZCODE_PRO_ROOT", str(_AGENT_WEBKIT_ROOT / "blitzcode-pro"))
)
_SESSIONS_DIR = Path(
    os.environ.get("AGENT_WEBKIT_SESSIONS_DIR", str(_AGENT_WEBKIT_ROOT / "sessions"))
)
_WORKSPACES_ROOT = Path(
    os.environ.get("BLITZCODE_PRO_WORKSPACES_ROOT", str(_BLITZ_PRO_ROOT / "workspaces"))
)

# Bundled mode: the Tauri shell sets this to "1" so we know to route logs
# to disk and trust the embedded port. Defaults to dev behavior (stderr
# logs, dev port).
_BUNDLED = os.environ.get("BLITZCODE_PRO_BUNDLED") == "1"


def _install_file_logging() -> None:
    """Route Python + uvicorn logs to ~/.blitzcode-pro/logs/server.log
    with rotation. Called only when running as a packaged app. In dev,
    uvicorn keeps its colorful stderr output."""
    log_dir = _BLITZ_PRO_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_dir / "server.log",
        maxBytes=2_000_000,         # 2 MB per file
        backupCount=4,              # keep .1 through .4
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    ))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Replace handlers so we don't double-log to stderr inside the .app.
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    # Uvicorn / FastAPI loggers — make them propagate to root.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True


if _BUNDLED:
    _install_file_logging()

metadata_store = FileSessionMetadataStore(_SESSIONS_DIR)
ack_store = AckStore(
    Path(os.environ.get("BLITZ_ACKS_PATH", str(_AGENT_WEBKIT_ROOT / "app-acks.json")))
)
workspace_store = WorkspaceStore(_BLITZ_PRO_ROOT / "workspaces.json")
initiative_store = InitiativeStore(_BLITZ_PRO_ROOT / "initiatives.json")
creds_store = CredsStore(_BLITZ_PRO_ROOT / "atlassian-creds.json")
settings_store = SettingsStore(_BLITZ_PRO_ROOT / "settings.json")

# ────────────────────────────────────────────────────────────────────────────
# Workflow prompt injection — called per session-creation by the SDK factory.
# Sessions outside a workspace get the stock Claude Code preset; sessions
# inside a workspace get an addendum with ticket + repo context.
# ────────────────────────────────────────────────────────────────────────────


def _workflow_system_prompt_append(config: SessionConfig) -> Optional[str]:
    if not config.workspace_id:
        return None
    ws = workspace_store.get(config.workspace_id)
    if ws is None:
        return None
    initiative_name: Optional[str] = None
    if ws.initiative_key:
        initiative = initiative_store.get(ws.initiative_key)
        if initiative is not None:
            initiative_name = initiative.display_name
    return render_workflow_prompt(ws, initiative_display_name=initiative_name)


# Build the workflow MCP server lazily on first session that needs it. The
# build closes over the real SDK import (claude_agent_sdk) which we don't
# want to load at module import — keeps test fixtures + non-SDK
# environments importing this module cleanly.
_workflow_server_cache: dict = {}


def _workflow_extra_mcp_servers(config: SessionConfig) -> Optional[dict[str, Any]]:
    if not config.workspace_id:
        return None
    if "server" not in _workflow_server_cache:
        try:
            _workflow_server_cache["server"] = build_workflow_mcp_server(
                WorkflowDeps(
                    creds_store=creds_store,
                    workspace_store=workspace_store,
                    initiative_store=initiative_store,
                )
            )
        except ImportError:  # pragma: no cover — claude_agent_sdk missing
            return None
    return {"workflow": _workflow_server_cache["server"]}


def _workflow_extra_allowed_tools(config: SessionConfig) -> Optional[list[str]]:
    if not config.workspace_id:
        return None
    return list(WORKFLOW_ALLOWED_TOOLS)


app = create_app(
    auth=AuthConfig(disabled=True),
    metadata_store=metadata_store,
    system_prompt_append=_workflow_system_prompt_append,
    extra_mcp_servers=_workflow_extra_mcp_servers,
    extra_allowed_tools=_workflow_extra_allowed_tools,
)

# CORS — only needed in dev where the Next dev server sits on a different
# port. The bundled .app serves frontend + API from the same origin so
# the middleware becomes a no-op there. Tauri dev also runs on an ephemeral
# port, so allow all localhost origins.
_DEV_WEB_ORIGINS = ["http://localhost:*", "http://127.0.0.1:*"]
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:[0-9]+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ────────────────────────────────────────────────────────────────────────────
# "Needs input" tracking — listens to result events on the global event log.
# ────────────────────────────────────────────────────────────────────────────


async def _watch_completions() -> None:
    registry = app.state.registry
    async for ev in registry.event_log.subscribe(0):
        if ev.event == "result":
            try:
                await ack_store.mark_completion(ev.session_id)
            except Exception:
                pass


# `create_app` already installs its own lifespan (starts the reaper). Chain
# ours around it instead of using `@app.on_event`, which FastAPI silently
# ignores once a lifespan context manager is registered.
_inner_lifespan = app.router.lifespan_context


@contextlib.asynccontextmanager
async def _chained_lifespan(app_: FastAPI):
    async with _inner_lifespan(app_):
        task = asyncio.create_task(_watch_completions())
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass


app.router.lifespan_context = _chained_lifespan


# ────────────────────────────────────────────────────────────────────────────
# Existing endpoints — acks, filesystem listing, completions
# ────────────────────────────────────────────────────────────────────────────


@app.get("/app/state")
async def app_state() -> dict:
    snap = ack_store.snapshot()
    return {
        "acks": {
            sid: {
                "last_completion_at": e.last_completion_at,
                "last_ack_at": e.last_ack_at,
            }
            for sid, e in snap.items()
        }
    }


@app.get("/app/fs/list")
async def fs_list(path: str | None = None) -> dict:
    target = Path(path).expanduser() if path else Path.home()
    try:
        target = target.resolve(strict=True)
    except (FileNotFoundError, RuntimeError):
        raise HTTPException(status_code=404, detail="Path not found")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")
    entries: list[dict] = []
    try:
        for child in sorted(target.iterdir(), key=lambda p: p.name.lower()):
            if child.name.startswith("."):
                continue
            try:
                is_dir = child.is_dir()
            except OSError:
                continue
            if not is_dir:
                continue
            entries.append({"name": child.name})
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")
    parent = str(target.parent) if target.parent != target else None
    return {
        "path": str(target),
        "parent": parent,
        "home": str(Path.home()),
        "entries": entries,
    }


@app.get("/app/sessions/{session_id}/completions")
async def session_completions(session_id: str) -> dict:
    md = await metadata_store.load(session_id) if metadata_store is not None else None
    cwd = md.cwd if md is not None else None
    data = discover_for_cwd(cwd)
    seen_agent_names = {a["name"] for a in data["agents"]}
    for ba in BUILTIN_AGENTS:
        if ba["name"] not in seen_agent_names:
            data["agents"].append(ba)
    return data


@app.post("/app/sessions/{session_id}/acknowledge")
async def acknowledge(session_id: str) -> dict:
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")
    entry = await ack_store.acknowledge(session_id)
    return {
        "session_id": session_id,
        "last_completion_at": entry.last_completion_at,
        "last_ack_at": entry.last_ack_at,
    }


# ────────────────────────────────────────────────────────────────────────────
# Initiatives — friendly umbrellas (e.g. "meowtorq" → display "Meowtorq",
# epic LLM-608, Confluence root page, list of repos). The agent uses these
# to correlate "is this part of <initiative>?" questions at workspace-create.
# ────────────────────────────────────────────────────────────────────────────


class InitiativeIn(BaseModel):
    key: str
    display_name: str
    epic_jira_key: Optional[str] = None
    confluence_root_page_id: Optional[str] = None
    repo_paths: list[str] = Field(default_factory=list)


def _initiative_dict(it: Initiative) -> dict:
    return {
        "key": it.key,
        "display_name": it.display_name,
        "epic_jira_key": it.epic_jira_key,
        "confluence_root_page_id": it.confluence_root_page_id,
        "repo_paths": list(it.repo_paths),
    }


@app.get("/app/initiatives")
async def list_initiatives() -> dict:
    return {"initiatives": [_initiative_dict(it) for it in initiative_store.list()]}


def _normalize_initiative_key(raw: str) -> str:
    """Coerce a user-typed key into the canonical slug form.

    Lowercases, swaps whitespace/underscores for dashes, drops anything
    outside [a-z0-9-]. Reduces runs of dashes and trims leading/trailing.
    """
    import re as _re
    s = (raw or "").strip().lower().replace("_", "-")
    s = _re.sub(r"\s+", "-", s)
    s = _re.sub(r"[^a-z0-9-]", "", s)
    s = _re.sub(r"-+", "-", s).strip("-")
    return s


@app.post("/app/initiatives")
async def create_initiative(body: InitiativeIn) -> dict:
    key = _normalize_initiative_key(body.key)
    if not is_valid_initiative_key(key):
        raise HTTPException(
            status_code=400,
            detail="Provide at least one alphanumeric character for the key.",
        )
    it = Initiative(
        key=key,
        display_name=body.display_name or key,
        epic_jira_key=body.epic_jira_key,
        confluence_root_page_id=body.confluence_root_page_id,
        repo_paths=list(body.repo_paths),
    )
    saved = await initiative_store.upsert(it)
    return _initiative_dict(saved)


class InitiativePatch(BaseModel):
    display_name: Optional[str] = None
    epic_jira_key: Optional[str] = None
    confluence_root_page_id: Optional[str] = None
    repo_paths: Optional[list[str]] = None


@app.patch("/app/initiatives/{key}")
async def patch_initiative(key: str, body: InitiativePatch) -> dict:
    fields = {k: v for k, v in body.model_dump().items() if v is not None}
    updated = await initiative_store.patch(key, **fields)
    if updated is None:
        raise HTTPException(status_code=404, detail="Initiative not found")
    return _initiative_dict(updated)


@app.delete("/app/initiatives/{key}")
async def delete_initiative(key: str) -> dict:
    ok = await initiative_store.remove(key)
    if not ok:
        raise HTTPException(status_code=404, detail="Initiative not found")
    return {"ok": True}


# ────────────────────────────────────────────────────────────────────────────
# Workspaces — tickets + worktrees + sessions
# ────────────────────────────────────────────────────────────────────────────


class RepoSpecIn(BaseModel):
    source_path: str
    branch: Optional[str] = None  # defaults to ticket_key


class CreateWorkspaceIn(BaseModel):
    ticket_key: str
    ticket_title: Optional[str] = None
    initiative_key: Optional[str] = None
    repos: list[RepoSpecIn] = Field(default_factory=list)
    # If true, spawn the first session immediately after worktrees are up.
    spawn_initial_session: bool = True
    permission_mode: Optional[str] = "acceptEdits"
    model: Optional[str] = None


def _workspace_dict(ws: Workspace) -> dict:
    return {
        "id": ws.id,
        "ticket_key": ws.ticket_key,
        "ticket_title": ws.ticket_title,
        "initiative_key": ws.initiative_key,
        "dir": ws.dir,
        "repos": [r.to_dict() for r in ws.repos],
        "session_ids": list(ws.session_ids),
        "session_names": dict(ws.session_names),
        "docs": {k: v.to_dict() for k, v in ws.docs.items()},
        "created_at": ws.created_at,
        "archived_at": ws.archived_at,
    }


@app.get("/app/workspaces")
async def list_workspaces(include_archived: bool = False) -> dict:
    return {
        "workspaces": [
            _workspace_dict(ws)
            for ws in workspace_store.list(include_archived=include_archived)
        ]
    }


@app.get("/app/workspaces/{workspace_id}")
async def get_workspace(workspace_id: str) -> dict:
    ws = workspace_store.get(workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return _workspace_dict(ws)


@app.post("/app/workspaces")
async def post_workspace(body: CreateWorkspaceIn) -> dict:
    if body.initiative_key and initiative_store.get(body.initiative_key) is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown initiative_key: {body.initiative_key}. POST /app/initiatives first.",
        )

    try:
        ws = await create_workspace(
            workspace_store,
            workspaces_root=_WORKSPACES_ROOT,
            ticket_key=body.ticket_key,
            ticket_title=body.ticket_title,
            initiative_key=body.initiative_key,
            repos=[RepoSpec(source_path=r.source_path, branch=r.branch) for r in body.repos],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except gw.WorktreeError as e:
        raise HTTPException(
            status_code=400,
            detail={"code": e.code, "message": e.message, "detail": e.detail},
        )

    first_session_id: Optional[str] = None
    if body.spawn_initial_session:
        first_session_id = await _spawn_session_into_workspace(
            ws,
            permission_mode=body.permission_mode,
            model=body.model,
        )
    return {
        "workspace": _workspace_dict(workspace_store.get(ws.id) or ws),
        "first_session_id": first_session_id,
    }


class PatchWorkspaceIn(BaseModel):
    ticket_title: Optional[str] = None
    initiative_key: Optional[str] = None
    archived: Optional[bool] = None  # True → archive, False → unarchive


@app.patch("/app/workspaces/{workspace_id}")
async def patch_workspace(workspace_id: str, body: PatchWorkspaceIn) -> dict:
    ws = workspace_store.get(workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail="Workspace not found")
    fields: dict = {}
    if body.ticket_title is not None:
        fields["ticket_title"] = body.ticket_title
    if body.initiative_key is not None:
        if body.initiative_key and initiative_store.get(body.initiative_key) is None:
            raise HTTPException(status_code=400, detail=f"Unknown initiative_key: {body.initiative_key}")
        fields["initiative_key"] = body.initiative_key or None
    if fields:
        ws = await workspace_store.patch(workspace_id, **fields)  # type: ignore[assignment]
    if body.archived is True:
        ws = await workspace_store.archive(workspace_id)
    elif body.archived is False:
        ws = await workspace_store.unarchive(workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail="Workspace disappeared")
    return _workspace_dict(ws)


@app.delete("/app/workspaces/{workspace_id}")
async def remove_workspace(workspace_id: str, force: bool = True) -> dict:
    ws = workspace_store.get(workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail="Workspace not found")
    ok = await delete_workspace(workspace_store, workspace_id, force=force)
    return {"ok": ok}


class AddRepoIn(BaseModel):
    source_path: str
    branch: Optional[str] = None


@app.post("/app/workspaces/{workspace_id}/repos")
async def add_repo(workspace_id: str, body: AddRepoIn) -> dict:
    ws = workspace_store.get(workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail="Workspace not found")
    source = str(Path(body.source_path).expanduser().resolve())
    branch = body.branch or ws.ticket_key
    repo_name = Path(source).name or "repo"
    worktree_path = str(Path(ws.dir) / repo_name)
    try:
        await gw.add_worktree(source, worktree_path, branch)
    except gw.WorktreeError as e:
        raise HTTPException(status_code=400, detail={"code": e.code, "message": e.message, "detail": e.detail})
    repo = WorkspaceRepo(source_path=source, worktree_path=worktree_path, branch=branch)
    updated = await workspace_store.add_repo(workspace_id, repo)
    if updated is None:
        raise HTTPException(status_code=404, detail="Workspace disappeared")
    return _workspace_dict(updated)


class SpawnSessionIn(BaseModel):
    permission_mode: Optional[str] = "acceptEdits"
    model: Optional[str] = None
    include_partial_messages: bool = True


@app.post("/app/workspaces/{workspace_id}/sessions")
async def spawn_session(workspace_id: str, body: SpawnSessionIn) -> dict:
    ws = workspace_store.get(workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail="Workspace not found")
    sid = await _spawn_session_into_workspace(
        ws,
        permission_mode=body.permission_mode,
        model=body.model,
        include_partial_messages=body.include_partial_messages,
    )
    return {"session_id": sid, "workspace": _workspace_dict(workspace_store.get(workspace_id) or ws)}


class RenameSessionIn(BaseModel):
    name: Optional[str] = None  # None or "" clears back to default ("S1", "S2", …)


@app.patch("/app/workspaces/{workspace_id}/sessions/{session_id}")
async def rename_session(workspace_id: str, session_id: str, body: RenameSessionIn) -> dict:
    updated = await workspace_store.set_session_name(workspace_id, session_id, body.name)
    if updated is None:
        raise HTTPException(status_code=404, detail="Workspace or session not found")
    return _workspace_dict(updated)


@app.delete("/app/workspaces/{workspace_id}/sessions/{session_id}")
async def delete_session(workspace_id: str, session_id: str) -> dict:
    ws = workspace_store.get(workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail="Workspace not found")
    if session_id not in ws.session_ids:
        raise HTTPException(status_code=404, detail="Session not in workspace")
    # Tear down the live session first (best-effort), then drop from the
    # workspace record. Order matters: if removal fails the user can retry.
    registry = app.state.registry
    try:
        await registry.remove(session_id, purge_metadata=True)
    except Exception:
        pass
    updated = await workspace_store.remove_session(workspace_id, session_id)
    if updated is None:
        raise HTTPException(status_code=404, detail="Workspace disappeared")
    return _workspace_dict(updated)


async def _spawn_session_into_workspace(
    ws: Workspace,
    *,
    permission_mode: Optional[str] = "acceptEdits",
    model: Optional[str] = None,
    include_partial_messages: bool = True,
) -> str:
    """Create a session with cwd=workspace dir + add_dirs=per-repo worktrees."""
    registry = app.state.registry
    config = SessionConfig(
        model=model,
        permission_mode=permission_mode,
        cwd=ws.dir,
        include_partial_messages=include_partial_messages,
        add_dirs=[r.worktree_path for r in ws.repos],
        workspace_id=ws.id,
    )
    session = await registry.create(config)
    await workspace_store.add_session(ws.id, session.id)
    return session.id


# ────────────────────────────────────────────────────────────────────────────
# Atlassian credentials — JIRA + Confluence. UI talks to these via two
# endpoints; the token is set once and lives in a chmod-0600 JSON file. The
# GET never returns the token, only "has_creds" + the non-secret site/email.
# ────────────────────────────────────────────────────────────────────────────


class AtlassianCredsIn(BaseModel):
    site_url: str
    email: str
    api_token: str


@app.get("/app/atlassian/has-creds")
async def atlassian_has_creds() -> dict:
    return creds_store.public_meta()


@app.post("/app/atlassian/creds")
async def atlassian_set_creds(body: AtlassianCredsIn) -> dict:
    try:
        await creds_store.set(body.site_url, body.email, body.api_token)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # Echo back only public metadata; never the token.
    return creds_store.public_meta()


@app.delete("/app/atlassian/creds")
async def atlassian_clear_creds() -> dict:
    await creds_store.clear()
    return creds_store.public_meta()


def _jira_client_or_400() -> JiraClient:
    creds = creds_store.get()
    if creds is None:
        raise HTTPException(
            status_code=400,
            detail={"code": "requires_credentials", "message": "Atlassian credentials not set"},
        )
    return JiraClient(creds.site_url, creds.email, creds.api_token)


@app.get("/app/workflow/search-tickets")
async def workflow_search_tickets(q: str, max_results: int = 10) -> dict:
    """UI-facing typeahead proxy. Live calls the JIRA API on every keystroke;
    the JIRA endpoint itself is cheap and we cap at 10 results."""
    client = _jira_client_or_400()
    try:
        results = await client.typeahead(q, max_results=max_results)
    except JiraError as e:
        raise HTTPException(status_code=400, detail={"code": "jira_error", "status": e.status, "message": e.message})
    return {
        "results": [
            {"key": r.key, "title": r.title, "status": r.status, "issuetype": r.issuetype}
            for r in results
        ]
    }


@app.get("/app/workflow/ticket/{key}")
async def workflow_get_ticket_raw(key: str) -> dict:
    """Single-ticket fetch. Used by the Ticket tile in the Docs panel until
    the full ADF→markdown round-trip lands (Phase 4)."""
    client = _jira_client_or_400()
    try:
        return await client.get_issue(key)
    except JiraError as e:
        raise HTTPException(status_code=400, detail={"code": "jira_error", "status": e.status, "message": e.message})


@app.get("/app/workflow/ticket-meta/{key}")
async def workflow_ticket_meta(key: str) -> dict:
    """Compact ticket metadata for inline validation/preview in the UI.

    Returns just enough to confirm "yes this ticket exists" and give the
    user a sanity-check title — no description body. Used by the Initiative
    Manager when validating an epic key.
    """
    client = _jira_client_or_400()
    try:
        raw = await client.get_issue(key, fields=["summary", "status", "issuetype"])
    except JiraError as e:
        raise HTTPException(status_code=400, detail={"code": "jira_error", "status": e.status, "message": e.message})
    f = raw.get("fields") or {}
    def name(o):
        return o.get("name") if isinstance(o, dict) else None
    creds = creds_store.get()
    url = f"{creds.site_url}/browse/{key}" if creds else None
    return {
        "key": str(raw.get("key") or key),
        "title": f.get("summary") or "",
        "status": name(f.get("status")),
        "issuetype": name(f.get("issuetype")),
        "url": url,
    }


@app.get("/app/workflow/search-pages")
async def workflow_search_pages(q: str, limit: int = 10) -> dict:
    """Fuzzy Confluence page search by title (CQL backed). Used by the
    Initiative Manager's root-page typeahead."""
    creds = creds_store.get()
    if creds is None:
        raise HTTPException(
            status_code=400,
            detail={"code": "requires_credentials", "message": "Atlassian credentials not set"},
        )
    client = ConfluenceClient(creds.site_url, creds.email, creds.api_token)
    try:
        results = await client.search_pages(q, limit=limit)
    except ConfluenceError as e:
        raise HTTPException(status_code=400, detail={"code": "confluence_error", "status": e.status, "message": e.message})
    return {"results": results}


@app.get("/app/workflow/page/{page_id}")
async def workflow_get_confluence_page(page_id: str, include_body: bool = True) -> dict:
    """Resolve a Confluence page id → title + url + ADF body.

    The body is included by default so the in-app viewer can render the
    page inline. Set ?include_body=0 for cheap metadata fetches (e.g. the
    Initiative Manager's root-page validation tooltip)."""
    creds = creds_store.get()
    if creds is None:
        raise HTTPException(
            status_code=400,
            detail={"code": "requires_credentials", "message": "Atlassian credentials not set"},
        )
    client = ConfluenceClient(creds.site_url, creds.email, creds.api_token)
    try:
        page = await client.get_page(page_id)
    except ConfluenceError as e:
        raise HTTPException(status_code=400, detail={"code": "confluence_error", "status": e.status, "message": e.message})
    out: dict = {
        "id": page.id,
        "title": page.title,
        "space_id": page.space_id,
        "parent_id": page.parent_id,
        "version": page.version,
        "url": page.url,
    }
    if include_body:
        out["body_adf"] = page.body_adf
    return out


# ────────────────────────────────────────────────────────────────────────────
# Settings — app-level user preferences (theme, etc).
# ────────────────────────────────────────────────────────────────────────────


@app.get("/app/settings")
async def get_settings() -> dict:
    return {"settings": settings_store.snapshot()}


@app.patch("/app/settings")
async def patch_settings(updates: dict) -> dict:
    """Shallow-merge per category. Pass `null` for a key to delete it."""
    merged = await settings_store.patch(updates or {})
    return {"settings": merged}


# ────────────────────────────────────────────────────────────────────────────
# Static frontend mount — bundled .app sets BLITZCODE_PRO_WEB_DIST to the
# Next.js static export. In dev we leave this unmounted and the Next dev
# server serves the UI on its own port. Mounted LAST so /app/* and
# /sessions/* routes always win.
# ────────────────────────────────────────────────────────────────────────────

_web_dist = os.environ.get("BLITZCODE_PRO_WEB_DIST")
if _web_dist:
    _web_path = Path(_web_dist).expanduser().resolve()
    if _web_path.is_dir():
        # `html=True` makes /foo serve /foo.html or /foo/index.html — matches
        # what Next's static export emits and gives us "real" routing.
        app.mount("/", StaticFiles(directory=str(_web_path), html=True), name="web")


if __name__ == "__main__":
    # Port priority: explicit env > dev default. Bundled mode sets the
    # env var to an ephemeral port chosen by the Rust shell.
    port = int(os.environ.get("BLITZCODE_PRO_PORT", "51820"))
    host = os.environ.get("BLITZCODE_PRO_HOST", "127.0.0.1")
    # When bundled, suppress uvicorn's own log_config so our root file
    # handler captures everything (otherwise dictConfig stomps it).
    log_config = None if _BUNDLED else uvicorn.config.LOGGING_CONFIG
    uvicorn.run(app, host=host, port=port, log_config=log_config)
