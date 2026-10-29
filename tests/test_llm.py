from mlx_model_scout.agents.llm import _extract_json


def test_leading_think_block_and_fences_are_stripped():
    assert _extract_json('<think>\nplan {x}\n</think>\n```json\n{"a": 1}\n```') == '{"a": 1}'


def test_think_tags_inside_json_values_survive():
    # Qwen cards: "does not generate <think></think> blocks" ends up as a caveat string.
    raw = '{"caveats": ["Does not generate <think></think> blocks"]}'
    assert _extract_json(raw) == raw
