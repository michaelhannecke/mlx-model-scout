"""Orchestrator (Plan-and-Execute with human-in-the-loop) for the recommend workflow.

Server-side, fully local implementation of the `recommend_models` flow, exposed as the
tools `run_recommendation` and `continue_recommendation` (two-call HITL protocol: the
run stops at a checkpoint, returns `needs_user_input`, and resumes on the next call).

    start():
      plan      = LLM -> ExecutionPlan (search terms)                 # plan-and-execute
      scan      = gather(scan per search term)                        # parallel
      scout     = scout.select_candidates(...)                        # sequential agent step
      fits      = gather(estimate_model_fit per candidate)            # parallel, deterministic
      if nothing fits -> no-fit branch: shorter context, smaller quants   # conditional
      >>> CHECKPOINT 1 (select_candidates)
    resume(select_candidates):
      apply keep/add/context -> recompute fits where needed
      if nothing selected fits -> no-fit branch again, stay at checkpoint 1
      cards     = analyst per usable candidate (sequential: one local LLM)
      report    = advisor.draft_report(...)
      >>> CHECKPOINT 2 (approve_report)
    resume(approve_report):
      approve -> save_report(approved=True); feedback -> advisor revision; abort/reject

Every step is audited: start_run/finish_run, tool_audit inside the tools, log_hitl for
each user decision. State between calls lives in `run_states` (RunState as JSON).
"""

import asyncio
import logging
import re

from openai import AsyncOpenAI

from mlx_model_scout.agents.advisor import USABLE, draft_report
from mlx_model_scout.agents.llm import LLMError, structured_call
from mlx_model_scout.agents.scout import select_candidates
from mlx_model_scout.app.alternatives import base_name, max_fitting_context, smaller_siblings
from mlx_model_scout.app.fit_calculator import available_gb
from mlx_model_scout.app.hf_client import HFClient, HubError
from mlx_model_scout.app.profiles import load_chips
from mlx_model_scout.config.settings import get_settings
from mlx_model_scout.db.audit import finish_run, log_hitl, start_run
from mlx_model_scout.db.models import RunStateRow, utcnow
from mlx_model_scout.db.session import session_scope
from mlx_model_scout.schemas.agents import CandidateFit, ExecutionPlan, RecommendationRun, RunState
from mlx_model_scout.schemas.fit import FitEstimate
from mlx_model_scout.schemas.model import HubModel, ModelCardInsights
from mlx_model_scout.tools.fit_tools import estimate_model_fit
from mlx_model_scout.tools.hardware_tools import resolve_profile
from mlx_model_scout.tools.hub_tools import analyze_model, analyze_model_card, scan_mlx_models
from mlx_model_scout.tools.report_tools import save_report

log = logging.getLogger(__name__)

WORKFLOW = "recommend_models"
MAX_SEARCH_TERMS = 3
MAX_REVISIONS = 2
MAX_NO_FIT_PROBES = 3  # no_fit candidates to search alternatives for
MAX_SIBLINGS = 3  # smaller variants to estimate per no_fit candidate
_TERM = re.compile(r"^[a-z0-9][a-z0-9.\-]{1,23}$")

PLAN_SYSTEM = (
    "You plan a search on the Hugging Face Hub for MLX models (author mlx-community) that suit a "
    "use case. Return 1-3 short search terms that appear in repo NAMES, e.g. 'coder', 'instruct', "
    "'vl' (vision-language), 'math', 'embed'. Prefer broad terms; never full model names or "
    "sentences. needs_vision is true only if the user must process images."
)

# Deterministic fallback when the planner LLM is unavailable: keyword -> search term.
_KEYWORD_TERMS = {
    "code": "coder",
    "coding": "coder",
    "program": "coder",
    "python": "coder",
    "image": "vl",
    "vision": "vl",
    "photo": "vl",
    "math": "math",
    "reason": "thinking",
}

CHECKPOINT_1 = (
    "CHECKPOINT 1 - candidate selection. Show the user the candidate table (repo, verdict, "
    "required vs. available GiB, est. tok/s, confidence, reason){alt} and ask which candidates to "
    "keep, remove or add. Do NOT decide for the user. Then call continue_recommendation with this "
    "run_id and keep=[...] (omit to keep all usable), optional add=[...] (repo ids) and optional "
    "context_tokens, or abort=true."
)
CHECKPOINT_2 = (
    "CHECKPOINT 2 - report approval. Show the user the ranked report (rank, repo, rationale, "
    "trade-offs, verdict, tok/s, required GiB) and the summary, and ask for approval. Only after "
    "the user explicitly approves call continue_recommendation(run_id, approve=true). For changes "
    "call it with feedback='...'; to reject call it with approve=false or abort=true."
)


# --- state ------------------------------------------------------------------------


def _save_state(run_id: str, state: RunState) -> None:
    with session_scope() as s:
        row = s.get(RunStateRow, run_id) or RunStateRow(run_id=run_id, state_json="")
        row.checkpoint = state.checkpoint
        row.state_json = state.model_dump_json()
        row.updated_at = utcnow()
        s.add(row)


def _load_state(run_id: str) -> RunState:
    with session_scope() as s:
        row = s.get(RunStateRow, run_id)
        if row is None:
            raise ValueError(f"Unknown run_id '{run_id}'. Start with run_recommendation.")
        state = RunState.model_validate_json(row.state_json)
    if state.checkpoint is None:
        raise ValueError(f"Run '{run_id}' is already finished. Start a new run with run_recommendation.")
    return state


def _response(run_id: str, state: RunState, status: str, message: str, **extra) -> RecommendationRun:
    return RecommendationRun(
        run_id=run_id,
        status=status,
        checkpoint=state.checkpoint,
        message=message,
        plan=state.plan,
        candidates=state.candidates,
        alternatives=state.alternatives,
        report=state.report,
        notes=state.notes,
        **extra,
    )


# --- steps --------------------------------------------------------------------------


def fallback_terms(use_case: str) -> list[str]:
    """Deterministic search terms from use-case keywords; always ends with 'instruct'."""
    text = use_case.lower()
    terms = [term for kw, term in _KEYWORD_TERMS.items() if kw in text]
    terms = list(dict.fromkeys([*terms, "instruct"]))
    return terms[:MAX_SEARCH_TERMS]


def validate_plan(plan: ExecutionPlan, use_case: str) -> tuple[ExecutionPlan, list[str]]:
    """Normalize terms (lowercase, short, no sentences), dedupe, cap; fall back if none survive."""
    notes: list[str] = []
    terms = [t.strip().lower().replace(" ", "-") for t in plan.search_terms]
    valid = list(dict.fromkeys(t for t in terms if _TERM.match(t)))
    if dropped := [t for t in terms if not _TERM.match(t)]:
        notes.append(f"Planner: dropped invalid search terms {dropped}.")
    if not valid:
        valid = fallback_terms(use_case)
        notes.append("Planner returned no usable search terms; keyword fallback used.")
    return plan.model_copy(update={"search_terms": valid[:MAX_SEARCH_TERMS]}), notes


async def make_plan(
    use_case: str, run_id: str | None = None, client: AsyncOpenAI | None = None
) -> tuple[ExecutionPlan, list[str]]:
    """Planning step (LLM, structured) with validation and keyword fallback."""
    try:
        plan = await structured_call(
            "orchestrator", ExecutionPlan, PLAN_SYSTEM, f"Use case: {use_case}", run_id=run_id, client=client
        )
    except LLMError as exc:
        log.warning("Planner LLM call failed: %s", exc)
        plan = ExecutionPlan(
            search_terms=fallback_terms(use_case), rationale="Keyword fallback (LLM unavailable)."
        )
        return plan, ["Planner LLM unavailable; keyword fallback used."]
    return validate_plan(plan, use_case)


async def _scan(terms: list[str], run_id: str) -> list[HubModel]:
    """Parallel scans (one per search term), deduplicated by repo id."""
    results = await asyncio.gather(*(scan_mlx_models(search=t, run_id=run_id) for t in terms))
    models = {m["repo_id"]: HubModel.model_validate(m) for r in results for m in r["models"]}
    return list(models.values())


async def _estimate(
    entries: list[tuple[str, str]], profile_id: str, ctx: int, run_id: str
) -> tuple[list[CandidateFit], list[str]]:
    """Parallel fit estimates for (repo_id, reason) pairs; failures become notes."""
    results = await asyncio.gather(
        *(estimate_model_fit(rid, profile_id, ctx, run_id=run_id) for rid, _ in entries),
        return_exceptions=True,
    )
    fits: list[CandidateFit] = []
    notes: list[str] = []
    for (rid, reason), res in zip(entries, results, strict=True):
        if isinstance(res, FitEstimate):
            fits.append(CandidateFit(repo_id=rid, reason=reason, fit=res))
        else:
            notes.append(f"{rid}: could not be estimated ({type(res).__name__}: {res}).")
    return fits, notes


async def _no_fit_branch(state: RunState, run_id: str) -> tuple[list[CandidateFit], list[str]]:
    """Conditional branch: shorter context for the same model, smaller quantized siblings."""
    settings = get_settings()
    profile = resolve_profile(state.profile_id)
    chip = load_chips().get(profile.chip)
    notes: list[str] = []
    probes = [c for c in state.candidates if c.fit.verdict == "no_fit"][:MAX_NO_FIT_PROBES]
    sibling_entries: list[tuple[str, str]] = []
    async with HFClient() as hf:
        for c in probes:
            spec = await analyze_model(c.repo_id)
            ctx = max_fitting_context(spec, profile, chip, state.context_tokens, settings)
            if ctx:
                notes.append(f"{c.repo_id} fits with a shorter context: continue with context_tokens={ctx}.")
            author = c.repo_id.split("/")[0]
            try:
                same_family = await hf.list_models(author, search=base_name(c.repo_id), limit=30)
            except HubError as exc:
                notes.append(f"Sibling search for {c.repo_id} failed: {exc}")
                continue
            for sib in smaller_siblings(c.repo_id, [m.repo_id for m in same_family])[:MAX_SIBLINGS]:
                sibling_entries.append((sib, f"Smaller variant of {c.repo_id} (no-fit alternative)."))
    known = {c.repo_id for c in state.candidates}
    sibling_entries = [e for e in dict.fromkeys(sibling_entries) if e[0] not in known]
    fits, est_notes = await _estimate(sibling_entries, state.profile_id, state.context_tokens, run_id)
    alternatives = [f for f in fits if f.fit.verdict in USABLE]
    if not alternatives and not any("shorter context" in n for n in notes):
        notes.append("No alternative found: consider a machine with more memory or a smaller model family.")
    return alternatives, notes + est_notes


async def _checkpoint_1(run_id: str, state: RunState, lead: str) -> RecommendationRun:
    state.checkpoint = "select_candidates"
    state.alternatives = []
    if not any(c.fit.verdict in USABLE for c in state.candidates):
        state.alternatives, notes = await _no_fit_branch(state, run_id)
        state.notes.extend(notes)
        lead += " None of the candidates fits; alternatives and shorter-context options are listed."
    _save_state(run_id, state)
    alt = " and the no-fit alternatives" if state.alternatives else ""
    return _response(run_id, state, "needs_user_input", f"{lead} {CHECKPOINT_1.format(alt=alt)}")


async def _report(run_id: str, state: RunState, feedback: str | None = None) -> RecommendationRun:
    """Analyst per usable candidate (sequential), then advisor; stops at checkpoint 2."""
    usable = [c for c in state.candidates if c.fit.verdict in USABLE]
    cards: dict[str, ModelCardInsights] = {}
    for c in usable:  # sequential: a local LLM server processes one request at a time
        try:
            cards[c.repo_id] = await analyze_model_card(c.repo_id, run_id=run_id)
        except HubError as exc:
            state.notes.append(f"{c.repo_id}: model card unavailable ({exc}).")
    use_case = (
        state.use_case
        if not feedback
        else f"{state.use_case}\nUser feedback on the previous draft: {feedback}"
    )
    report = await draft_report(
        use_case,
        state.profile_id,
        state.context_tokens,
        [c.fit for c in state.candidates],
        cards,
        run_id=run_id,
    )
    state.report = report.model_copy(update={"use_case": state.use_case})
    state.checkpoint = "approve_report"
    _save_state(run_id, state)
    return _response(run_id, state, "needs_user_input", CHECKPOINT_2)


# --- public entry points --------------------------------------------------------------


async def start(
    use_case: str, profile_id: str = "local", context_tokens: int | None = None, max_candidates: int = 6
) -> RecommendationRun:
    """Run plan -> scan -> scout -> fits and stop at checkpoint 1."""
    settings = get_settings()
    ctx = context_tokens or settings.default_context_tokens
    run_id = start_run(WORKFLOW, {"use_case": use_case, "profile_id": profile_id, "context_tokens": ctx})
    try:
        profile = resolve_profile(profile_id)
        avail, _ = available_gb(profile, settings)
        state = RunState(use_case=use_case, profile_id=profile_id, context_tokens=ctx)

        state.plan, notes = await make_plan(use_case, run_id)
        state.notes.extend(notes)
        models = await _scan(state.plan.search_terms, run_id)
        scout_use_case = use_case + ("" if state.plan.needs_vision else " (text only, no images)")
        scout = await select_candidates(scout_use_case, models, avail, max_candidates, run_id=run_id)
        state.notes.extend(scout.notes)
        state.candidates, notes = await _estimate(
            [(c.repo_id, c.reason) for c in scout.candidates], profile_id, ctx, run_id
        )
        state.notes.extend(notes)
        lead = (
            f"Scanned {len(models)} repos for {state.plan.search_terms}; {len(state.candidates)} candidates."
        )
        return await _checkpoint_1(run_id, state, lead)
    except Exception as exc:
        finish_run(run_id, "failed", f"{type(exc).__name__}: {exc}")
        raise


def _check_arguments(
    checkpoint: str | None,
    keep: list[str] | None,
    add: list[str] | None,
    context_tokens: int | None,
    approve: bool | None,
    feedback: str | None,
) -> None:
    """Reject decisions that belong to the other checkpoint.

    Without this a client model could skip checkpoint 1 by sending approve=true (observed with a
    local model in agent mode): the selection would silently default to "keep all".
    """
    if checkpoint == "select_candidates" and (approve is not None or feedback):
        raise ValueError(
            "This run is at checkpoint 'select_candidates': show the candidates to the user and pass "
            "keep/add/context_tokens (or abort). approve/feedback are only valid at 'approve_report'."
        )
    if checkpoint == "approve_report" and (keep is not None or add or context_tokens):
        raise ValueError(
            "This run is at checkpoint 'approve_report': show the report to the user and pass "
            "approve=true/false or feedback (or abort). keep/add/context_tokens are only valid at "
            "'select_candidates'."
        )


async def resume(
    run_id: str,
    keep: list[str] | None = None,
    add: list[str] | None = None,
    context_tokens: int | None = None,
    approve: bool | None = None,
    feedback: str | None = None,
    abort: bool = False,
) -> RecommendationRun:
    """Apply the user's decision at the current checkpoint and continue the run."""
    state = _load_state(run_id)
    checkpoint = state.checkpoint
    _check_arguments(checkpoint, keep, add, context_tokens, approve, feedback)
    if abort:
        log_hitl(run_id, checkpoint, options=["continue", "abort"], decision={"abort": True})
        state.checkpoint = None
        _save_state(run_id, state)
        finish_run(run_id, "aborted")
        return _response(run_id, state, "aborted", "Run aborted by the user.")
    try:
        if checkpoint == "select_candidates":
            return await _resume_selection(run_id, state, keep, add, context_tokens)
        return await _resume_approval(run_id, state, approve, feedback)
    except Exception as exc:
        finish_run(run_id, "failed", f"{type(exc).__name__}: {exc}")
        raise


async def _resume_selection(
    run_id: str, state: RunState, keep: list[str] | None, add: list[str] | None, context_tokens: int | None
) -> RecommendationRun:
    offered = {c.repo_id: c for c in [*state.candidates, *state.alternatives]}
    log_hitl(
        run_id,
        "select_candidates",
        options=list(offered),
        decision={"keep": keep, "add": add, "context_tokens": context_tokens},
    )
    if keep is None:
        keep = [c.repo_id for c in state.candidates if c.fit.verdict in USABLE]
    if unknown := [rid for rid in keep if rid not in offered]:
        state.notes.append(f"Ignored unknown repo ids in keep: {unknown} (use add for new repos).")
    selected = [offered[rid] for rid in dict.fromkeys(keep) if rid in offered]
    added = [(rid, "Added by the user.") for rid in dict.fromkeys(add or []) if rid not in offered]

    if context_tokens and context_tokens != state.context_tokens:
        state.context_tokens = context_tokens
        entries = [(c.repo_id, c.reason) for c in selected] + added
        state.candidates, notes = await _estimate(entries, state.profile_id, context_tokens, run_id)
    else:
        new, notes = await _estimate(added, state.profile_id, state.context_tokens, run_id)
        state.candidates = selected + new
    state.notes.extend(notes)

    if not any(c.fit.verdict in USABLE for c in state.candidates):
        return await _checkpoint_1(run_id, state, "Selection re-evaluated.")
    return await _report(run_id, state)


async def _resume_approval(
    run_id: str, state: RunState, approve: bool | None, feedback: str | None
) -> RecommendationRun:
    if approve is True:
        saved = save_report(state.report.model_dump(mode="json"), approved=True, run_id=run_id)
        state.checkpoint = None
        _save_state(run_id, state)
        finish_run(run_id, "succeeded")
        return _response(
            run_id, state, "completed", "Report approved and saved.", report_id=saved["report_id"]
        )
    if feedback:
        if state.revisions >= MAX_REVISIONS:
            return _response(
                run_id,
                state,
                "needs_user_input",
                f"Maximum of {MAX_REVISIONS} revisions reached. Ask the user to approve or abort.",
            )
        log_hitl(
            run_id, "approve_report", options=["approve", "revise", "reject"], decision={"feedback": feedback}
        )
        state.revisions += 1
        return await _report(run_id, state, feedback=feedback)
    if approve is False:
        saved = save_report(state.report.model_dump(mode="json"), approved=False, run_id=run_id)
        state.checkpoint = None
        _save_state(run_id, state)
        finish_run(run_id, "aborted")
        return _response(
            run_id, state, "aborted", "Report rejected; stored as not approved.", report_id=saved["report_id"]
        )
    return _response(run_id, state, "needs_user_input", "No decision given. " + CHECKPOINT_2)
