import json
import pytest


def _make_understanding():
    return {
        "summary": "Expedia flight search page with round-trip, from/to, date range, and search button.",
        "affordance": [
            "select trip type",
            "set origin/destination",
            "set date range",
            "search flights",
        ],
        "elements": [
            {
                "element_id": "element_19",
                "primary_action": "click",
                "description": "Round-trip option",
                "secondary_actions": ["toggle"],
                "confidence": 0.92,
            },
            {
                "element_id": "element_7",
                "primary_action": "type",
                "description": "Leaving from",
                "secondary_actions": ["click"],
                "confidence": 0.88,
            },
            {
                "element_id": "element_8",
                "primary_action": "type",
                "description": "Going to",
                "secondary_actions": ["click"],
                "confidence": 0.87,
            },
            {
                "element_id": "element_1",
                "primary_action": "click",
                "description": "Date range",
                "secondary_actions": ["type"],
                "confidence": 0.86,
            },
            {
                "element_id": "element_6",
                "primary_action": "click",
                "description": "Search",
                "secondary_actions": [],
                "confidence": 0.9,
            },
        ],
    }


class _DummyTok:
    def __call__(self, inputs, return_tensors="pt"):
        # Minimal structure; values without .to() keep device move a no-op
        return {"input_ids": [0, 1, 2, 3]}


class _DummyModel:
    def parameters(self):
        # No real parameters; _infer_runtime_device is patched in tests
        return []

    def generate(self, **kwargs):
        # Shape/content not used because _decode_new_text is patched
        return object()


@pytest.fixture()
def monkeypatch_decider(monkeypatch):
    # Avoid heavy imports; stub transformers and torch before importing decision_agent
    import sys, types

    tfm = types.ModuleType("transformers")
    class _AutoTok:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _DummyTok()
    class _AutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            class _M:
                def eval(self):
                    return self
            return _M()
    tfm.AutoTokenizer = _AutoTok
    tfm.AutoModelForCausalLM = _AutoModel
    sys.modules.setdefault("transformers", tfm)

    torch_mod = types.ModuleType("torch")
    class _Cuda:
        @staticmethod
        def is_available():
            return False
    class _NoGrad:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
    def _device(x):
        return "cpu"
    torch_mod.cuda = _Cuda()
    torch_mod.no_grad = _NoGrad
    torch_mod.device = _device
    sys.modules.setdefault("torch", torch_mod)

    # Now import and patch the agent
    from decision_agent import DecisionAgent

    def _fake_init(self, model_name: str = "fake", use_gpu: bool = False):
        self.model_name = model_name
        self.device = "cpu"
        self.tokenizer = _DummyTok()
        self.model = _DummyModel()

    monkeypatch.setattr(DecisionAgent, "__init__", _fake_init, raising=False)
    # Keep device on CPU for inputs alignment
    monkeypatch.setattr(DecisionAgent, "_infer_runtime_device", lambda self: "cpu", raising=False)
    return DecisionAgent


def test_plan_returns_structured_json_when_model_emits_embedded_json(monkeypatch_decider):
    DecisionAgent = monkeypatch_decider
    agent = DecisionAgent()

    # Model output with leading noise and an inner JSON object (no <thinking> tags)
    embedded = (
        "some leading notes... assistantthinking>\n"
        + json.dumps(
            {
                "plan": "Set round-trip, fill from/to, set dates, then search.",
                "steps": [
                    {"element_id": "element_19", "actions": ["click"], "details": "Select round-trip"},
                    {"element_id": "element_7", "actions": ["click", "type"], "details": "Type 'Boston'"},
                    {"element_id": "element_8", "actions": ["click", "type"], "details": "Type 'Los Angeles'"},
                    {"element_id": "element_1", "actions": ["click", "type"], "details": "10/5-10/8"},
                    {"element_id": "element_6", "actions": ["click"], "details": "Search"},
                ],
                "success_criteria": [
                    "Origin shows BOS",
                    "Destination shows LAX",
                    "Dates set to 10/5-10/8",
                ],
            }
        )
        + "\ntrailing noise"
    )

    from decision_agent import DecisionAgent as _DA

    # Force the decoding to return our embedded string
    def _fake_decode(self, out_ids, inputs):
        return embedded

    # patch decode only
    import decision_agent as _mod

    _mod.DecisionAgent._decode_new_text = _fake_decode  # type: ignore

    understanding = _make_understanding()
    task = "Book flight BOS to LAX on 10/5-10/8"

    obj = agent.plan(understanding, task)
    assert isinstance(obj, dict)
    assert set(["plan", "steps", "success_criteria"]).issubset(obj.keys())
    assert isinstance(obj["steps"], list) and len(obj["steps"]) >= 3
    allowed_ids = {e["element_id"] for e in understanding["elements"]}
    for st in obj["steps"]:
        assert st.get("element_id") in allowed_ids
        acts = st.get("actions")
        assert isinstance(acts, list) and len(acts) > 0
        assert all(isinstance(a, str) for a in acts)
    assert isinstance(obj["success_criteria"], list) and len(obj["success_criteria"]) >= 2

