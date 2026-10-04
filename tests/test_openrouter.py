import json
import unittest

from kk_ru.openrouter import (
    ModelInfo,
    OpenRouterClient,
    OpenRouterError,
    estimate_cost,
    usage_cost,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return self.payload


class FakeOpener:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def __call__(self, request, timeout):
        self.requests.append((request, timeout))
        return FakeResponse(next(self.responses))


class OpenRouterClientTests(unittest.TestCase):
    def test_lists_models_and_parses_prices(self):
        opener = FakeOpener(
            [
                {
                    "data": [
                        {
                            "id": "qwen/test",
                            "name": "Qwen test",
                            "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                            "context_length": 4096,
                            "supported_parameters": ["reasoning", "temperature"],
                        }
                    ]
                }
            ]
        )
        client = OpenRouterClient(base_url="https://example.test/api/v1", opener=opener)

        models = client.list_models()

        self.assertEqual(models["qwen/test"].prompt_price, 0.000001)
        self.assertEqual(models["qwen/test"].completion_price, 0.000002)
        self.assertEqual(models["qwen/test"].supported_parameters, ("reasoning", "temperature"))
        self.assertEqual(opener.requests[0][0].full_url, "https://example.test/api/v1/models")

    def test_unknown_catalog_price_is_unavailable(self):
        opener = FakeOpener(
            [{"data": [{"id": "qwen/test", "pricing": {"prompt": "-1", "completion": "-1"}}]}]
        )

        model = OpenRouterClient(opener=opener).list_models()["qwen/test"]

        self.assertIsNone(model.prompt_price)
        self.assertIsNone(model.completion_price)

    def test_complete_sends_auth_and_extracts_text(self):
        opener = FakeOpener(
            [
                {
                    "id": "response-1",
                    "model": "qwen/test",
                    "provider": "provider-test",
                    "choices": [{"message": {"content": [{"type": "text", "text": " перевод "}]}}],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 2},
                }
            ]
        )
        client = OpenRouterClient(api_key="secret", opener=opener)

        result = client.complete("qwen/test", [{"role": "user", "content": "Сәлем"}])

        request = opener.requests[0][0]
        self.assertEqual(request.get_header("Authorization"), "Bearer secret")
        self.assertEqual(result.text, "перевод")
        self.assertEqual(result.provider, "provider-test")
        self.assertEqual(result.usage["completion_tokens"], 2)

    def test_missing_api_key_fails_before_network(self):
        opener = FakeOpener([])
        client = OpenRouterClient(api_key=None, opener=opener)

        with self.assertRaises(OpenRouterError):
            client.complete("qwen/test", [{"role": "user", "content": "Сәлем"}])
        self.assertEqual(opener.requests, [])


class CostTests(unittest.TestCase):
    def setUp(self):
        self.model = ModelInfo("qwen/test", "Qwen test", 0.000001, 0.000002, 4096)

    def test_usage_cost_prefers_reported_cost(self):
        self.assertEqual(usage_cost({"cost": "0.25", "prompt_tokens": 1}, self.model), 0.25)

    def test_usage_cost_calculates_when_cost_is_missing(self):
        self.assertAlmostEqual(
            usage_cost({"prompt_tokens": 10, "completion_tokens": 5}, self.model),
            0.00002,
        )

    def test_estimate_cost_uses_maximum_completion_tokens(self):
        messages = [{"role": "user", "content": "Сәлем"}]
        self.assertGreater(estimate_cost(self.model, messages, 10), 0)


if __name__ == "__main__":
    unittest.main()
