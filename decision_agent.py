"""
decision_agent.py - Post-processing decision agent that selects a single element/action
using a large language model (default: openai/gpt-oss-20b).

Inputs: minimal screen understanding dict {summary, affordance, elements}
Task: a natural-language instruction of what to accomplish

Output JSON:
{
  "thoughts": "short reasoning (<=120 words)",
  "element_id": "element_11",
  "action": "click"  # one of: click,type,toggle,open,navigate,select,none
}

Coordinates are intentionally excluded from the model input.
"""

from typing import Dict, Any, List, Optional
import json
import logging

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

logger = logging.getLogger(__name__)


ALLOWED_ACTIONS = ["click", "type", "toggle", "open", "navigate", "select", "none"]


class DecisionAgent:
    def __init__(self, model_name: str = "openai/gpt-oss-20b", use_gpu: bool = True):
        self.model_name = model_name
        self.device = 'cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu'
        logger.info(f"Initializing DecisionAgent with model={model_name} on {self.device}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            # Prefer auto-dtype and device_map auto for large models. When using device_map='auto',
            # DO NOT manually move inputs to CUDA; let HF Accelerate handle placement.
            self._device_map = ("cuda" if self.device == 'cuda' else None)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map=self._device_map,
                trust_remote_code=True
            ).eval()
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            raise

    def _sanitize(self, understanding: Dict[str, Any]) -> Dict[str, Any]:
        """Drop coordinates and keep only model-relevant fields."""
        out: Dict[str, Any] = {
            'summary': understanding.get('summary', ''),
            'affordance': understanding.get('affordance', []),
        }
        elements: List[Dict[str, Any]] = []
        for e in understanding.get('elements', []) or []:
            elements.append({
                'element_id': e.get('element_id'),
                'primary_action': e.get('primary_action'),
                'description': e.get('description'),
                'secondary_actions': e.get('secondary_actions') or [],
                'confidence': e.get('confidence', 0.0),
            })
        out['elements'] = elements
        return out

    def _infer_runtime_device(self):
        """Pick a concrete tensor device for inputs to reduce HF warnings.

        Prefer the first non-meta parameter device; fallback to CUDA if available,
        else CPU. This helps align input_ids device with model's runtime device
        even when using device_map='auto'.
        """
        try:
            for p in self.model.parameters():
                dev = getattr(p, 'device', None)
                if dev is not None and str(dev) != 'meta':
                    return dev
        except Exception:
            pass
        return torch.device('cuda') if (self.device == 'cuda') else torch.device('cpu')

    def _build_prompt(self, task: str, ctx: Dict[str, Any]) -> str:
        # Render context as plain text (avoid embedding JSON that can confuse extraction)
        lines = []
        lines.append("You are an expert UI automation planner.")
        lines.append("Select EXACTLY ONE element and ONE action to progress the Task.")
        lines.append("Rules:")
        lines.append("- Prefer semantic match between Task and element description/secondary_actions.")
        lines.append("- Prefer higher-confidence; tie-break by earlier order.")
        lines.append(f"- Allowed actions: {', '.join(ALLOWED_ACTIONS)}.")
        lines.append("- If none apply, return action='none' and element_id=''.")
        lines.append("- Do NOT invent element ids.")
        lines.append("")
        lines.append(f"Task: {task}")
        lines.append("")
        lines.append("Summary:")
        lines.append(str(ctx.get('summary','')).strip())
        lines.append("")
        aff = ctx.get('affordance') or []
        if aff:
            lines.append("Affordances:")
            for a in aff[:60]:
                lines.append(f"- {a}")
            lines.append("")
        elems = ctx.get('elements') or []
        lines.append("Elements:")
        for e in elems[:200]:
            eid = e.get('element_id')
            pa = e.get('primary_action')
            desc = e.get('description') or ''
            sec = ", ".join(e.get('secondary_actions') or [])
            conf = e.get('confidence', 0.0)
            lines.append(f"- id={eid} | action={pa} | conf={conf:.2f} | desc={desc} | secondary=[{sec}]")
        lines.append("")
        lines.append("Return ONLY one JSON object with keys exactly: thoughts, element_id, action, wrapped inside <decision> and </decision> tags.")
        lines.append("Do not echo the above context.")
        return "\n".join(lines)

    def _extract_decision(self, text: str) -> Dict[str, Any]:
        """Extract decision JSON from model text.

        Preference order:
        1) JSON inside <decision>...</decision> tags
        2) Any JSON object that contains keys 'element_id' and 'action'
        3) Fallback: empty
        """
        import re
        # 1) decision tags
        try:
            m = re.search(r"<decision>\s*(\{.*?\})\s*</decision>", text, re.S | re.I)
            if m:
                return json.loads(m.group(1))
        except Exception:
            pass
        # 2) find all candidate objects and prefer the last valid one
        try:
            candidates = re.findall(r"\{[^\{\}]*\}", text, re.S)
            picked = {}
            for c in candidates:
                try:
                    obj = json.loads(c)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                if 'element_id' in obj and 'action' in obj:
                    act = str(obj.get('action') or '').lower()
                    # skip schema-like placeholders
                    if 'one of' in act or 'string' in str(obj.get('element_id','')).lower():
                        continue
                    picked = obj
            return picked
        except Exception:
            return {}

    def decide(self, understanding: Dict[str, Any], task: str, max_new_tokens: int = 384) -> Dict[str, Any]:
        ctx = self._sanitize(understanding)
        prompt = self._build_prompt(task, ctx)

        inputs = self.tokenizer([prompt], return_tensors='pt')
        # Align input tensors with a real model parameter device to avoid warnings.
        try:
            dev = self._infer_runtime_device()
            if str(dev) != 'cpu':
                inputs = {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in inputs.items()}
        except Exception:
            pass
        try:
            with torch.no_grad():
                out_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.2,
                    num_beams=1,
                )
        except RuntimeError as e:
            # Handle dtype/device mismatches like "expected scalar type Half but found BFloat16"
            # or meta/cuda placement issues. Reload cleanly and retry.
            msg = str(e)
            logger.warning(f"Generation failed, attempting reload fallback: {msg}")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype="auto",
                    device_map=("auto" if self.device == 'cuda' else None),
                    trust_remote_code=True
                ).eval()
                # Keep inputs on CPU for auto device map
                inputs = self.tokenizer([prompt], return_tensors='pt')
            except Exception as e2:
                logger.error(f"Fallback reload failed: {e2}")
                raise e
            with torch.no_grad():
                out_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.2,
                    num_beams=1,
                )
        text = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]
        obj = self._extract_decision(text)

        # Minimal validation
        el = str(obj.get('element_id') or '')
        act = str(obj.get('action') or '').lower().strip()
        if act not in ALLOWED_ACTIONS:
            act = 'none'
        # keep thoughts as provided (may be empty)
        return {
            'thoughts': str(obj.get('thoughts') or ''),
            'element_id': el,
            'action': act,
            'model': self.model_name,
        }
