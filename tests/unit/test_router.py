"""Tests for task classification and model routing."""

from unittest.mock import MagicMock

from forge.core.router import Route, classify, ModelRouter


class TestClassify:
    """Tests for the classify() heuristic function."""

    def test_force_overrides_heuristic(self):
        assert classify("hello", force=Route.HEAVY) == Route.HEAVY
        assert classify("write a function", force=Route.FAST) == Route.FAST
        assert classify("explain this", force=Route.NPU) == Route.NPU

    def test_short_prompt_routes_to_fast(self):
        assert classify("hello") == Route.FAST
        assert classify("hi there") == Route.FAST

    def test_short_prompt_routes_to_npu_when_available(self):
        assert classify("hello", has_npu=True) == Route.NPU
        assert classify("hi there", has_npu=True) == Route.NPU

    def test_heavy_keywords_route_to_heavy(self):
        heavy_prompts = [
            "write a function to sort a list",
            "implement binary search",
            "refactor this class",
            "debug this error",
            "fix the login bug",
            "create a REST API",
            "build a web scraper",
            "design the database schema",
            "optimize this query",
            "analyze this code",
        ]
        for prompt in heavy_prompts:
            assert classify(prompt) == Route.HEAVY, f"Expected HEAVY for: {prompt}"

    def test_fast_keywords_route_to_fast(self):
        fast_prompts = [
            "what is a closure?",
            "explain decorators in Python",
            "summarize this function",
            "define polymorphism",
            "list the HTTP methods",
            "how does garbage collection work?",
            "translate this to English",
            "convert celsius to fahrenheit formula",
        ]
        for prompt in fast_prompts:
            assert classify(prompt) == Route.FAST, f"Expected FAST for: {prompt}"

    def test_long_prompts_default_to_heavy(self):
        long_prompt = "a " * 150  # > 200 chars
        assert classify(long_prompt) == Route.HEAVY

    def test_medium_prompts_default_to_fast(self):
        medium_prompt = "Tell me about the history of computing and its impact"
        assert classify(medium_prompt) == Route.FAST

    def test_short_heavy_keyword_still_routes_heavy(self):
        assert classify("fix this bug", has_npu=True) == Route.HEAVY


class TestModelRouter:
    """Tests for the ModelRouter class."""

    def setup_method(self):
        self.heavy = MagicMock()
        self.fast = MagicMock()
        self.npu = MagicMock()

    def test_route_returns_correct_backend(self):
        router = ModelRouter(heavy=self.heavy, fast=self.fast)
        route, backend = router.route("write a function")
        assert route == Route.HEAVY
        assert backend is self.heavy

    def test_route_fast(self):
        router = ModelRouter(heavy=self.heavy, fast=self.fast)
        route, backend = router.route("what is Python?")
        assert route == Route.FAST
        assert backend is self.fast

    def test_route_with_npu(self):
        router = ModelRouter(heavy=self.heavy, fast=self.fast, npu=self.npu)
        route, backend = router.route("hello")
        assert route == Route.NPU
        assert backend is self.npu

    def test_npu_fallback_to_fast(self):
        router = ModelRouter(heavy=self.heavy, fast=self.fast)
        backend = router.get_backend(Route.NPU)
        assert backend is self.fast

    def test_force_route(self):
        router = ModelRouter(heavy=self.heavy, fast=self.fast)
        route, backend = router.route("hello", force=Route.HEAVY)
        assert route == Route.HEAVY
        assert backend is self.heavy

    def test_get_backend_direct(self):
        router = ModelRouter(heavy=self.heavy, fast=self.fast, npu=self.npu)
        assert router.get_backend(Route.HEAVY) is self.heavy
        assert router.get_backend(Route.FAST) is self.fast
        assert router.get_backend(Route.NPU) is self.npu
