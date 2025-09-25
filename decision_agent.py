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

    def _decode_new_text(self, out_ids, inputs) -> str:
        """Decode only the newly generated tokens (exclude the prompt)."""
        # try:
        #     in_len = int(inputs['input_ids'].shape[-1])
        #     gen = out_ids[0, in_len:]
        #     return self.tokenizer.decode(gen, skip_special_tokens=True)
        # except Exception:
        return self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]

    # -------- Simple fallback ranking (no-LLM) ---------
    @staticmethod
    def _tok(s: str) -> List[str]:
        import re
        return [t for t in re.findall(r"[a-z0-9]+", (s or '').lower()) if t]

    @staticmethod
    def _score_element(task: str, e: Dict[str, Any]) -> float:
        task_tokens = set(DecisionAgent._tok(task))
        text = " ".join([
            str(e.get('description') or ''),
            " ".join(e.get('secondary_actions') or []),
            str(e.get('primary_action') or ''),
        ])
        elem_tokens = set(DecisionAgent._tok(text))
        inter = len(task_tokens & elem_tokens)
        base = (inter / max(1, len(task_tokens)))
        conf = float(e.get('confidence', 0.0))
        # Prefer clickable/typable/selectable when task mentions keyword
        action_bias = 0.0
        pa = str(e.get('primary_action') or '').lower()
        if 'type' in task_tokens and pa == 'type':
            action_bias += 0.3
        if 'click' in task_tokens and pa == 'click':
            action_bias += 0.2
        if 'select' in task_tokens and pa == 'select':
            action_bias += 0.25
        return base + 0.3 * conf + action_bias

    @staticmethod
    def _best_action_for_element(e: Dict[str, Any]) -> str:
        pa = str(e.get('primary_action') or '').lower().strip()
        if pa in ALLOWED_ACTIONS:
            return pa
        for a in (e.get('secondary_actions') or []):
            a = str(a or '').lower().strip()
            if a in ALLOWED_ACTIONS:
                return a
        return 'click'

    def _build_decide_prompt(self, task: str, ctx: Dict[str, Any], *, actions_history: Optional[List[Dict[str, Any]]] = None, last_thinking: Optional[str] = None) -> str:
        # Render context as plain text (avoid embedding JSON that can confuse extraction)
        lines = []
        lines.append("You are an expert UI automation planner.")
        lines.append("Select EXACTLY ONE element and ONE action to progress the Task.")
        lines.append("Rules (be precise and conservative):")
        lines.append("- Use ONLY elements listed below; do NOT invent element ids.")
        lines.append("- Prefer strong semantic match between Task and element description/secondary_actions.")
        lines.append("- Prefer higher-confidence; tie-break by earlier order.")
        lines.append(f"- Allowed actions: {', '.join(ALLOWED_ACTIONS)}.")
        lines.append("- If no element clearly advances the Task OR you are unsure, return action='none' and element_id=''.")
        lines.append("- When returning 'none', begin thoughts with 'BLOCKED:' and give a concrete reason (<= 15 words).")
        lines.append("- Keep thoughts decisive; avoid speculative words like 'maybe', 'perhaps', 'try', 'guess'.")
        lines.append("- Never repeat any prior (element_id, action) pair from history.")
        # History-aware guidance
        if actions_history:
            lines.append("- Assume all previous actions were successfully executed; advance to the NEXT step.")
            lines.append("- Do NOT repeat any prior (element_id, action) pair.")
        if last_thinking:
            lines.append("- Follow the previous plan unless the current UI requires updating.")
        lines.append("")
        lines.append(f"Task: {task}")
        lines.append("")
        if actions_history:
            lines.append("Previous Actions (all steps):")
            for i, a in enumerate(actions_history, 1):
                step = a.get('step', i)
                act = a.get('action', '')
                eid = a.get('element_id', '')
                note = a.get('note') or a.get('thoughts') or ''
                note = (note or '')[:160]
                lines.append(f"- step {step}: {act} on element_id={eid} {('- ' + note) if note else ''}")
            lines.append("")
            # Explicitly mark forbidden repeats
            lines.append("Forbidden (already executed, do NOT choose again):")
            seen = set()
            for a in actions_history[-60:]:
                eid = str(a.get('element_id') or '')
                act = str(a.get('action') or '')
                key = (eid, act)
                if key in seen:
                    continue
                seen.add(key)
                if eid and act:
                    lines.append(f"- {act} on element_id={eid}")
            lines.append("")
        if last_thinking:
            lines.append("Previous Thinking (last step):")
            lines.append(str(last_thinking).strip())
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
        lines.append("Return ONLY one JSON object with keys exactly: thoughts, element_id, action, wrapped inside <decision>...</decision> tags.")
        lines.append("If blocked/unsure, set element_id='' and action='none' and start thoughts with 'BLOCKED:'.")
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

    def decide(self, understanding: Dict[str, Any], task: str, max_new_tokens: int = 1000, *, actions_history: Optional[List[Dict[str, Any]]] = None, last_thinking: Optional[str] = None) -> Dict[str, Any]:
        ctx = self._sanitize(understanding)
        prompt = self._build_decide_prompt(task, ctx, actions_history=actions_history, last_thinking=last_thinking)

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
        text = self._decode_new_text(out_ids, inputs)
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

    # -------- Planning (pre-decision) ---------
    def _build_plan_prompt(self, task: str, ctx: Dict[str, Any], *, actions_history: Optional[List[Dict[str, Any]]] = None, prev_thinking: Optional[str] = None) -> str:
        lines: List[str] = []
        schema = (
            "Return ONLY one JSON object with keys: "
            "plan (string), steps (array of objects with element_id, actions [strings], details [string]), "
            "success_criteria (array of strings). Keep any extra natural-language instructions under 'details'. Do not include any other text."
        )

        # Intro and rules
        lines.append("You are an expert UI automation planner.")
        lines.append("Produce the most reliable next 1–3 steps for the Task.")
        lines.append("Rules (favor safety and certainty):")
        lines.append("- Use ONLY elements listed below; do NOT invent element ids.")
        lines.append("- Include a step ONLY if it clearly advances the Task.")
        lines.append("- Avoid speculative language (e.g., 'maybe', 'try', 'guess'). Keep details concise.")
        lines.append("- Update the previous plan: drop completed steps, keep still-relevant steps, and propose the next 1-3 steps.")
        lines.append("- Assume all previously planned or executed steps have succeeded; continue from there.")
        lines.append("- Do NOT repeat previously planned steps or any prior (element_id, action).")
        lines.append("- If nothing applicable or the screen blocks progress, set steps=[] and make plan start with 'BLOCKED: <reason>'.")
        if actions_history:
            lines.append("- Assume previous steps succeeded; DO NOT repeat any prior (element_id, action).")
        lines.append("")
        lines.append(f"Task: {task}")
        lines.append("")

        # History context
        if actions_history:
            lines.append("Actions Taken So Far (all steps):")
            for i, a in enumerate(actions_history, 1):
                step = a.get('step', i)
                act = a.get('action', '')
                eid = a.get('element_id', '')
                note = a.get('note') or a.get('thoughts') or ''
                note = (note or '')[:160]
                lines.append(f"- step {step}: {act} on {eid} {('- ' + note) if note else ''}")
            lines.append("")
        if prev_thinking:
            lines.append("Previous Plan (last step):")
            lines.append(str(prev_thinking).strip())
            # Summary of previous plan and steps for continuity
            try:
                _s = str(prev_thinking).strip()
                _i = _s.find('{'); _j = _s.rfind('}')
                _payload = _s[_i:_j+1] if (_i != -1 and _j != -1 and _j > _i) else _s
                _obj = json.loads(_payload)
            except Exception:
                _obj = None
            if isinstance(_obj, dict):
                _plan_txt = str(_obj.get('plan') or '').strip()
                if _plan_txt:
                    lines.append("Previous Plan Summary:")
                    lines.append(_plan_txt)
                _steps_in = _obj.get('steps') or []
                if isinstance(_steps_in, list) and _steps_in:
                    lines.append("Previously Planned Steps (assume completed):")
                    for _idx2, _it in enumerate(_steps_in, 1):
                        if isinstance(_it, dict):
                            _eid = str(_it.get('element_id') or '')
                            _actions = _it.get('actions')
                            if isinstance(_actions, list):
                                _acts_txt = ",".join([str(a) for a in _actions])
                            elif _actions is None:
                                _acts_txt = ""
                            else:
                                _acts_txt = str(_actions)
                            lines.append(f"- step {_idx2}: {_eid} [{_acts_txt}]")
            lines.append("")

        # Current UI summary and affordances
        lines.append("Current UI Summary:")
        lines.append(str(ctx.get('summary','')).strip())
        lines.append("")
        aff = ctx.get('affordance') or []
        if aff:
            lines.append("Affordances:")
            for a in aff[:50]:
                lines.append(f"- {a}")
            lines.append("")

        # Available elements
        elems = ctx.get('elements') or []
        if elems:
            lines.append("Available Elements (use these ids in steps.element_id):")
            for e in elems[:100]:
                eid = e.get('element_id')
                desc = e.get('description') or ''
                pa = str(e.get('primary_action') or '').lower().strip()
                sec = [str(a or '').lower().strip() for a in (e.get('secondary_actions') or [])]
                acts = [a for a in ([pa] + sec) if a in ALLOWED_ACTIONS]
                acts_str = ",".join(sorted(set(acts))) if acts else "click,type,select,open,navigate,toggle,none"
                lines.append(f"- id={eid} | actions=[{acts_str}] | desc={desc}")
            lines.append("")

        # Output contract
        lines.append(f"Output schema: {schema}")
        lines.append("Return ONLY the JSON object; do not echo the context.")
        return "\n".join(lines)

    def _extract_thinking(self, text: str) -> str:
        import re
        # Prefer the LAST <thinking> ... </thinking>
        matches = re.findall(r"<thinking>\s*(.*?)\s*</thinking>", text, re.S | re.I)
        for seg in reversed(matches):
            val = (seg or '').strip()
            # Ignore trivial captures like the literal "and" from the prompt line
            if len(val) >= 12 and val.lower() != 'and':
                return val
        # Fallback: JSON with key 'thinking'
        try:
            candidates = re.findall(r"\{[^\{\}]*\}", text, re.S)
            for c in candidates[::-1]:  # prefer last
                try:
                    obj = json.loads(c)
                except Exception:
                    continue
                if isinstance(obj, dict) and 'thinking' in obj:
                    return str(obj['thinking']).strip()
        except Exception:
            pass
        # Final fallback: return last 2000 chars trimmed
        return text[-2000:].strip()

    def _normalize_thinking_json(self, text_out: str) -> Dict[str, Any]:
        """Ensure the thinking content is a JSON object with desired keys.

        If the model returned plain text, wrap it as {
          plan: <text>, steps: [], success_criteria: []
        } to guarantee downstream JSON shape without parsing natural language.
        """
        parsed: Dict[str, Any] = {}
        results: Dict[str, Any] = {}
        import json as _json
        try:
            s = text_out.strip()
            if s.startswith('[') and s.endswith(']'):
                arr = _json.loads(s)
                if isinstance(arr, list) and arr:
                    parsed = arr[0] if isinstance(arr[0], dict) else {}
            else:
                i = s.find('{'); j = s.rfind('}')
                payload = s[i:j+1] if (i != -1 and j != -1 and j > i) else s
                obj = _json.loads(payload)
                if isinstance(obj, dict):
                    parsed = obj
        except Exception:
            parsed = {}
        import copy

        plan = str(parsed.get('plan','')).strip()
        results["plan"] = plan
        steps_in = parsed.get('steps')
        steps_out: List[Dict[str, Any]] = []
        if isinstance(steps_in, list):
            for it in steps_in:
                if not isinstance(it, dict):
                    continue
                eid = str(it.get('element_id') or '')
                actions = it.get('actions')
                if isinstance(actions, list):
                    acts = [str(a) for a in actions if isinstance(a, (str, int, float))]
                elif actions is None:
                    acts = []
                else:
                    acts = [str(actions)]
                details = it.get('details')
                details = str(details) if details is not None else ''
                steps_out.append({'element_id': eid, 'actions': acts, 'details': details})

        results["success_criteria"] = parsed.get('success_criteria')
        results["steps"] = copy.deepcopy(steps_out) if isinstance(steps_out, list) else []
        return results

    def plan(self, understanding: Dict[str, Any], task: str, *, actions_history: Optional[List[Dict[str, Any]]] = None, prev_thinking: Optional[str] = None, max_new_tokens: int = 2400) -> Dict[str, Any]:
        ctx = self._sanitize(understanding)
        prompt = self._build_plan_prompt(task, ctx, actions_history=actions_history, prev_thinking=prev_thinking)
        model_device = torch.device('cuda' if self.device == 'cuda' else 'cpu')
        inputs = self.tokenizer(text = [prompt], return_tensors='pt')
        inputs = {k: (v.to(model_device) if hasattr(v, 'to') else v) for k, v in inputs.items()}
        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.2,
                num_beams=1,
            )
        text_out = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]
        results = self._normalize_thinking_json(text_out)
        return results

        import re
        arr: List[Dict[str, Any]] = []
        try:
            m = re.search(r"<candidates>\s*(\[.*?\])\s*</candidates>", text, re.S | re.I)
            if m:
                obj = json.loads(m.group(1))
                if isinstance(obj, list):
                    arr = obj
        except Exception:
            pass
        if not arr:
            # Fallback: find last JSON array in text
            try:
                candidates = re.findall(r"\[[^\[\]]*\]", text, re.S)
                for c in candidates[::-1]:
                    try:
                        obj = json.loads(c)
                    except Exception:
                        continue
                    if isinstance(obj, list):
                        arr = obj
                        break
            except Exception:
                arr = []
        # Normalize and validate
        out: List[Dict[str, Any]] = []
        seen = set()
        for it in arr:
            if not isinstance(it, dict):
                continue
            eid = str(it.get('element_id') or '')
            act = str(it.get('action') or '').lower().strip()
            reason = str(it.get('reason') or '')
            prio = it.get('priority')
            try:
                prio = int(prio)
            except Exception:
                prio = 9999
            if not eid or act not in ALLOWED_ACTIONS:
                continue
            key = (eid, act)
            if key in seen:
                continue
            seen.add(key)
            out.append({'element_id': eid, 'action': act, 'reason': reason, 'priority': prio})
        # Sort by priority ascending
        out.sort(key=lambda x: x.get('priority', 9999))
        return out

    def suggest_actions(self, understanding: Dict[str, Any], task: str, *, actions_history: Optional[List[Dict[str, Any]]] = None, last_thinking: Optional[str] = None, max_candidates: int = 12, max_new_tokens: int = 512) -> List[Dict[str, Any]]:
        ctx = self._sanitize(understanding)
        prompt = self._build_candidates_prompt(task, ctx, actions_history=actions_history, last_thinking=last_thinking, max_candidates=max_candidates)

        inputs = self.tokenizer([prompt], return_tensors='pt')
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
            logger.warning(f"Candidates generation failed, attempting reload fallback: {e}")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype="auto",
                    device_map=("auto" if self.device == 'cuda' else None),
                    trust_remote_code=True
                ).eval()
                inputs = self.tokenizer([prompt], return_tensors='pt')
            except Exception as e2:
                logger.error(f"Candidates reload failed: {e2}")
                raise e
            with torch.no_grad():
                out_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.2,
                    num_beams=1,
                )
        text = self._decode_new_text(out_ids, inputs)
        cand = self._extract_candidates(text)
        if cand:
            return cand
        # Fallback: simple heuristic candidates
        elems = ctx.get('elements') or []
        ranked = []
        for e in elems:
            sc = self._score_element(task, e)
            act = self._best_action_for_element(e)
            ranked.append((sc, e, act))
        ranked.sort(key=lambda x: x[0], reverse=True)
        out = []
        pr = 1
        used = set()
        for sc, e, act in ranked:
            if pr > max_candidates:
                break
            if sc <= 0:
                # allow continue to include top by confidence later
                continue
            key = (str(e.get('element_id') or ''), act)
            if key in used:
                continue
            used.add(key)
            out.append({
                'element_id': e.get('element_id'),
                'action': act,
                'reason': f'fallback by heuristic score={sc:.2f}',
                'priority': pr,
            })
            pr += 1
        if out:
            return out
        # Last resort: top by confidence
        try:
            elems_sorted = sorted(elems, key=lambda ee: float(ee.get('confidence', 0.0)), reverse=True)
            for e in elems_sorted[:max_candidates]:
                out.append({
                    'element_id': e.get('element_id'),
                    'action': self._best_action_for_element(e),
                    'reason': 'fallback by confidence',
                    'priority': pr,
                })
                pr += 1
        except Exception:
            pass
        return out

    # -------- Visualization for planning steps ---------
    @staticmethod
    def render_thinking_actions(image, understanding: Dict[str, Any], thinking: Dict[str, Any]):
        """Return a PIL.Image annotated with planned steps (element boxes + actions/details).

        - image: PIL.Image (RGB) of the current screen
        - understanding: minimal dict containing elements with bbox/center
        - thinking: dict with keys plan, steps(list of {element_id, actions, details})
        """
        try:
            from PIL import ImageDraw, ImageFont
        except Exception as e:
            raise RuntimeError(f"PIL not available for rendering: {e}")

        if image is None:
            raise ValueError("render_thinking_actions: image is None")

        annotated = image.convert('RGB').copy()
        draw = ImageDraw.Draw(annotated)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        # Map element_id -> element dict (expect bbox)
        id_to_elem = {}
        try:
            for e in (understanding.get('elements') or []):
                eid = str(e.get('element_id') or '')
                if eid:
                    id_to_elem[eid] = e
        except Exception:
            pass

        steps = thinking.get('steps') or []
        colors = [
            (255, 0, 0), (0, 153, 255), (0, 180, 0), (255, 140, 0), (153, 0, 255),
            (220, 20, 60), (0, 128, 128), (128, 0, 128), (46, 139, 87), (70, 130, 180)
        ]

        def _textsize(text):
            try:
                tw = draw.textlength(text, font=font)
                th = font.size if font else 12
            except Exception:
                tw, th = len(text) * 6, 12
            return int(tw), int(th)

        for idx, st in enumerate(steps, 1):
            try:
                eid = str(st.get('element_id') or '')
                acts = st.get('actions') or []
                details = str(st.get('details') or '')
                if not eid or eid not in id_to_elem:
                    continue
                el = id_to_elem[eid]
                b = el.get('bbox') or {}
                x, y = int(b.get('x', 0)), int(b.get('y', 0))
                w, h = int(max(1, b.get('width', 0))), int(max(1, b.get('height', 0)))
                x1, y1, x2, y2 = x, y, x + w, y + h
                color = colors[(idx - 1) % len(colors)]
                # Draw rectangle
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                # Compose label
                acts_txt = ",".join([str(a) for a in acts]) if isinstance(acts, list) else str(acts)
                det = details.strip()
                if len(det) > 80:
                    det = det[:77] + '...'
                label = f"{idx}. {eid} [{acts_txt}]"
                if det:
                    label += f" - {det}"
                # Measure & draw background above box
                pad = 3
                tw, th = _textsize(label)
                bg = [x1, max(0, y1 - th - 2 * pad), min(annotated.width, x1 + tw + 2 * pad), y1]
                draw.rectangle(bg, fill=color)
                draw.text((bg[0] + pad, bg[1] + pad), label, fill=(255, 255, 255), font=font)
            except Exception:
                continue

        # Optional: draw global plan snippet at top-left
        try:
            plan_txt = str(thinking.get('plan') or '').strip()
            if plan_txt:
                plan = plan_txt if len(plan_txt) <= 140 else (plan_txt[:137] + '...')
                header = f"Plan: {plan}"
                pad = 6
                tw, th = _textsize(header)
                draw.rectangle([0, 0, min(annotated.width, tw + 2 * pad), th + 2 * pad], fill=(0, 0, 0))
                draw.text((pad, pad), header, fill=(255, 255, 255), font=font)
        except Exception:
            pass

        return annotated

