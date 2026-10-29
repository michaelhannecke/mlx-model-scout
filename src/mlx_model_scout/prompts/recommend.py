"""MCP prompt: `recommend_models` agentic workflow with human-in-the-loop.

The prompt instructs the MCP client's agent to chain the server's tools and
to STOP for user input at two checkpoints. When used with the bundled local
client, the same flow runs against the local mlx-lm model.
"""


def recommend_models_prompt(use_case: str, profile_id: str = "local", context_tokens: int = 8192) -> str:
    return f"""You are an advisor for running local LLMs with Apple MLX.
Goal: recommend MLX models for this use case: "{use_case}"
Hardware profile: `{profile_id}` (use `list_hardware_profiles` if unknown; `local` = this Mac).
Target context length: {context_tokens} tokens.

Follow these steps strictly, in order:

1. Call `list_hardware_profiles` and confirm the profile exists. If `local` is not
   detectable (e.g. server runs in a container), ask the user to pick a configured profile.
2. Call `scan_mlx_models` (optionally with a `search` term derived from the use case).
3. From the results, pick up to 10 plausible candidates for the use case. Explain briefly why.
4. For each candidate call `estimate_fit` with profile `{profile_id}` and context {context_tokens}.
   Never recompute or alter the numbers the tool returns.
5. Present a shortlist table: repo, params (total/active), quant bits, required vs. available GiB,
   verdict, estimated tok/s, confidence.

   >>> CHECKPOINT 1 (human-in-the-loop): STOP and ask the user which candidates to keep,
   >>> remove or add. Do not continue until the user answers.

6. If nothing fits: propose alternatives (lower-bit quantization, smaller model, shorter context)
   and ask the user how to proceed.
7. Write a ranked recommendation (max 5) with rationale and trade-offs, based only on tool output.

   >>> CHECKPOINT 2 (human-in-the-loop): show the draft and ask for approval.
   >>> Only after explicit approval call `save_report` with approved=true.

8. Optionally offer `run_benchmark` (mock) or `download_model` (mock, requires confirmed=true
   after asking the user).

Alternative: `run_recommendation` + `continue_recommendation` run the same flow server-side with
the local LLM agents and return at the same two checkpoints (status `needs_user_input`).
"""
