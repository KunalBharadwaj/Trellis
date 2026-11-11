"""Harness metric helpers (Phase 0)."""

from bench import metrics


def test_parse_plain_json():
    assert metrics.parse_json('{"a": 1}') == {"a": 1}


def test_parse_fenced_json():
    text = 'Sure!\n```json\n{"a": 1, "b": [2, 3]}\n```\nDone.'
    assert metrics.parse_json(text) == {"a": 1, "b": [2, 3]}


def test_parse_embedded_json_object():
    text = 'Here you go: {"name": "Ada", "age": 36} hope that helps'
    assert metrics.parse_json(text) == {"name": "Ada", "age": 36}


def test_braces_inside_strings_do_not_confuse_extractor():
    text = '{"note": "a } brace in a string"}'
    assert metrics.parse_json(text) == {"note": "a } brace in a string"}


def test_invalid_json_returns_none():
    assert metrics.parse_json("not json at all") is None
    assert metrics.parse_json('{"a": 1,}') is None  # trailing comma is invalid JSON


def test_satisfies_schema():
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
        "additionalProperties": False,
    }
    assert metrics.satisfies_schema({"name": "Ada", "age": 36}, schema)
    assert not metrics.satisfies_schema({"name": "Ada"}, schema)  # missing age
    assert not metrics.satisfies_schema({"name": "Ada", "age": "x"}, schema)  # wrong type
    assert not metrics.satisfies_schema({"name": "Ada", "age": 1, "extra": 1}, schema)


def test_aggregate_rates_and_latency():
    rs = [
        metrics.RunResult("t", "baseline", True, True, 1.0, 10),
        metrics.RunResult("t", "baseline", True, False, 3.0, 10),
    ]
    aggs = metrics.aggregate(rs)
    assert len(aggs) == 1
    a = aggs[0]
    assert a.n == 2
    assert a.schema_valid_rate == 0.5
    assert a.valid_json_rate == 1.0
    assert a.mean_latency_s == 2.0
