import unittest
from unittest.mock import patch

from clients.openai_responses_client import (
    _create_and_consume_stream_with_retry,
    _extract_response_payload,
    _normalize_to_response_input,
)


class TestOpenAIResponsesClient(unittest.TestCase):
    def test_extract_response_payload_ignores_non_text_sdk_metadata(self):
        class FakeMetadata:
            type = "response_format"

            def __str__(self) -> str:
                return "ResponseTextConfig(format=...)"

        response = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": '{"ok": true}',
                        },
                        FakeMetadata(),
                    ],
                }
            ]
        }

        text, reasoning, grounding = _extract_response_payload(response)
        self.assertEqual(text, '{"ok": true}')
        self.assertEqual(reasoning, "")
        self.assertEqual(grounding, [])

    def test_normalize_to_response_input_keeps_images(self):
        response_input, instructions = _normalize_to_response_input(
            contents="hello",
            image_parts=[
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": "ZmFrZQ==",
                    }
                }
            ],
            system_instruction="sys",
        )

        self.assertEqual(instructions, "sys")
        self.assertEqual(len(response_input), 1)
        self.assertEqual(response_input[0]["role"], "user")
        self.assertEqual(response_input[0]["content"][0]["type"], "input_text")
        self.assertEqual(response_input[0]["content"][1]["type"], "input_image")
        self.assertTrue(
            response_input[0]["content"][1]["image_url"].startswith("data:image/png;base64,")
        )

    def test_normalize_to_response_input_maps_assistant_to_output_text(self):
        response_input, instructions = _normalize_to_response_input(
            contents=[
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "上一轮回复"},
            ],
            system_instruction="sys",
        )

        self.assertEqual(instructions, "sys")
        self.assertEqual(response_input[0]["role"], "user")
        self.assertEqual(response_input[0]["content"][0]["type"], "input_text")
        self.assertEqual(response_input[1]["role"], "assistant")
        self.assertEqual(response_input[1]["content"][0]["type"], "output_text")
        self.assertEqual(response_input[1]["content"][0]["text"], "上一轮回复")

    def test_extract_response_payload_reads_text_reasoning_and_grounding(self):
        response = {
            "output_text": "final answer",
            "output": [
                {
                    "type": "reasoning",
                    "content": [{"type": "reasoning_text", "text": "step1"}],
                    "summary": [{"type": "summary_text", "text": "sum"}],
                },
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "final answer",
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "title": "Example",
                                    "url": "https://example.com",
                                },
                                {
                                    "type": "url_citation",
                                    "title": "Example",
                                    "url": "https://example.com",
                                },
                            ],
                        }
                    ],
                },
            ],
        }

        text, reasoning, grounding = _extract_response_payload(response)
        self.assertEqual(text, "final answer")
        self.assertEqual(reasoning, "step1sum")
        self.assertEqual(
            grounding,
            [{"title": "Example", "uri": "https://example.com"}],
        )


class TestOpenAIResponsesStreamRetry(unittest.IsolatedAsyncioTestCase):
    async def test_create_and_consume_stream_with_retry_retries_chunk_timeout(self):
        class TimeoutStream:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise TimeoutError()

        class SuccessStream:
            def __init__(self):
                self._events = iter(
                    [
                        {
                            "type": "response.output_text.delta",
                            "delta": "hello",
                        }
                    ]
                )

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    return next(self._events)
                except StopIteration:
                    raise StopAsyncIteration from None

        class FakeResponses:
            def __init__(self):
                self.calls = 0

            async def create(self, **_kwargs):
                self.calls += 1
                if self.calls == 1:
                    return TimeoutStream()
                return SuccessStream()

        class FakeClient:
            def __init__(self):
                self.responses = FakeResponses()

        async def fake_with_retry(fn, **_kwargs):
            last_exc = None
            for _ in range(2):
                try:
                    return await fn()
                except TimeoutError as exc:
                    last_exc = exc
                    continue
            raise last_exc or AssertionError("expected timeout retry")

        client = FakeClient()
        with patch("clients.openai_responses_client.with_retry", fake_with_retry):
            text, reasoning, grounding = await _create_and_consume_stream_with_retry(
                client,
                {"model": "gpt-5.4", "input": []},
            )

        self.assertEqual(client.responses.calls, 2)
        self.assertEqual(text, "hello")
        self.assertEqual(reasoning, "")
        self.assertEqual(grounding, [])
