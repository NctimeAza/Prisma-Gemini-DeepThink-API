import unittest

from clients.openai_client import (
    _clean_json_string,
    _normalize_json_schema,
    _parse_json_lenient,
)


class TestOpenAISchemaNormalization(unittest.TestCase):
    def test_parse_json_lenient_accepts_trailing_garbage(self):
        parsed = _parse_json_lenient('{"ok": true} trailing text')

        self.assertEqual(parsed, {"ok": True})

    def test_clean_json_string_extracts_first_complete_object(self):
        raw = (
            '{"ok": true, "text": "brace } inside string"}'
            "ResponseTextConfig(format={'type':'json_schema'})"
        )

        cleaned = _clean_json_string(raw)

        self.assertEqual(cleaned, '{"ok": true, "text": "brace } inside string"}')

    def test_normalize_json_schema_forces_required_to_cover_all_properties(self):
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "string"},
            },
            "required": ["a"],
        }

        normalized = _normalize_json_schema(schema)

        self.assertEqual(normalized["required"], ["a", "b"])

    def test_normalize_json_schema_adds_additional_properties_false(self):
        schema = {
            "type": "OBJECT",
            "properties": {
                "meta": {
                    "type": "OBJECT",
                    "properties": {
                        "title": {"type": "STRING"},
                    },
                    "required": ["title"],
                },
                "items": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "name": {"type": "STRING"},
                        },
                    },
                },
            },
            "required": ["meta", "items"],
        }

        normalized = _normalize_json_schema(schema)

        self.assertEqual(normalized["type"], "object")
        self.assertFalse(normalized["additionalProperties"])
        self.assertEqual(normalized["properties"]["meta"]["type"], "object")
        self.assertFalse(
            normalized["properties"]["meta"]["additionalProperties"]
        )
        self.assertEqual(
            normalized["properties"]["items"]["items"]["type"], "object"
        )
        self.assertFalse(
            normalized["properties"]["items"]["items"]["additionalProperties"]
        )

    def test_normalize_json_schema_keeps_explicit_additional_properties(self):
        schema = {
            "type": "object",
            "properties": {
                "payload": {
                    "type": "object",
                    "additionalProperties": True,
                    "properties": {
                        "x": {"type": "string"},
                    },
                }
            },
        }

        normalized = _normalize_json_schema(schema)

        self.assertFalse(normalized["additionalProperties"])
        self.assertTrue(
            normalized["properties"]["payload"]["additionalProperties"]
        )
