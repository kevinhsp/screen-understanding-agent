"""Interactive session runner for multi-step screenshots (English-only prompts).

Flow:
  1) Ask the user for the task (English)
  2) Create a new session folder under pipeline_outputs/sessions/
  3) For step N, the user drops N.png into the session folder
  4) Run the pipeline and decision agent; save step_N_* outputs
  5) Repeat for N+1 until the user quits
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional

from PIL import Image

from models import ProcessingConfig
from pipeline import ScreenUnderstandingPipeline
from decision_agent import DecisionAgent
from difflib import SequenceMatcher


def ask(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        return ""


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ----------------- Persistent Element Registry (per-session) -----------------
def _norm_text(s: str) -> str:
    if not s:
        return ""
    return " ".join(str(s).lower().strip().split())


def _quant_bbox_norm(elem: dict, img_w: int, img_h: int):
    # Returns normalized [x1,y1,x2,y2] rounded to 0.01
    bx = elem.get('bbox') or {}
    try:
        x = float(bx.get('x', 0))
        y = float(bx.get('y', 0))
        w = float(bx.get('width', 0))
        h = float(bx.get('height', 0))
        if img_w > 0 and img_h > 0:
            x1 = max(0.0, min(1.0, x / img_w))
            y1 = max(0.0, min(1.0, y / img_h))
            x2 = max(0.0, min(1.0, (x + w) / img_w))
            y2 = max(0.0, min(1.0, (y + h) / img_h))
        else:
            # Fallback to centers if pixel dims unknown
            c = elem.get('center') or {}
            cx = float(c.get('x', 0))
            cy = float(c.get('y', 0))
            x1 = max(0.0, min(1.0, cx / 1.0))
            y1 = max(0.0, min(1.0, cy / 1.0))
            x2 = x1
            y2 = y1
    except Exception:
        x1 = y1 = x2 = y2 = 0.0
    # round to 2 decimals to reduce jitter
    return [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]


def _grid_code(elem: dict, img_w: int, img_h: int, grid: int = 6) -> str:
    c = elem.get('center') or {}
    try:
        cx = float(c.get('x', 0))
        cy = float(c.get('y', 0))
        if img_w > 0 and img_h > 0:
            gx = int(min(grid - 1, max(0, cx * grid / img_w)))
            gy = int(min(grid - 1, max(0, cy * grid / img_h)))
        else:
            # if absolute unknown, treat center as [0..1]
            gx = int(min(grid - 1, max(0, cx * grid)))
            gy = int(min(grid - 1, max(0, cy * grid)))
    except Exception:
        gx = gy = 0
    # Row code: A..F, Col: 1..grid
    row = chr(ord('A') + gy)
    col = str(gx + 1)
    return f"{row}{col}"


def _crop_bbox(img: Image.Image, elem: dict) -> Optional[Image.Image]:
    bx = elem.get('bbox') or {}
    try:
        x = int(max(0, bx.get('x', 0)))
        y = int(max(0, bx.get('y', 0)))
        w = int(max(1, bx.get('width', 0)))
        h = int(max(1, bx.get('height', 0)))
        x2 = min(img.width, x + w)
        y2 = min(img.height, y + h)
        x1 = min(max(0, x), img.width)
        y1 = min(max(0, y), img.height)
        if x2 <= x1 or y2 <= y1:
            return None
        return img.crop((x1, y1, x2, y2))
    except Exception:
        return None


def _dhash(img: Image.Image) -> str:
    try:
        # Difference hash (8x8 bits): resize to 9x8 grayscale, compare adjacent pixels horizontally
        g = img.convert('L').resize((9, 8))
        pix = list(g.getdata())
        bits = []
        for row in range(8):
            row_start = row * 9
            for col in range(8):
                left = pix[row_start + col]
                right = pix[row_start + col + 1]
                bits.append(1 if right > left else 0)
        # Pack bits to hex
        val = 0
        for b in bits:
            val = (val << 1) | (1 if b else 0)
        return f"{val:016x}"
    except Exception:
        return ""


def _hamming(a: str, b: str) -> int:
    try:
        if not a or not b:
            return 64
        ia = int(a, 16)
        ib = int(b, 16)
        x = ia ^ ib
        return x.bit_count()
    except Exception:
        return 64


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    ub = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    den = ua + ub - inter
    return inter / den if den > 0 else 0.0


def _load_registry(session_dir: Path) -> dict:
    reg_path = session_dir / "elements_registry.json"
    if reg_path.exists():
        try:
            return json.loads(reg_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"next_index": 1, "items": []}


def _save_registry(session_dir: Path, reg: dict) -> None:
    (session_dir / "elements_registry.json").write_text(
        json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8"
    )



def _assign_persistent_ids(session_dir: Path, elements: list, img: Image.Image, img_w: int, img_h: int, step: int):
    """Assign/lookup persistent_id per element and return mapping + enhanced elements.

    Strategy: level-1 match on (grid, text_norm, primary_action),
              else level-1.5 by dHash (Hamming <= 10) with same primary_action,
              else level-2 by IoU>0.6 and text similarity>0.85.
    """
    reg = _load_registry(session_dir)
    items = reg.get("items", [])

    # Build quick index structures
    idx_grid_text_role = {}
    for it in items:
        key = (it.get('grid'), it.get('text_norm'), it.get('primary_action'))
        idx_grid_text_role.setdefault(key, []).append(it)

    def new_pid():
        i = int(reg.get('next_index', 1))
        reg['next_index'] = i + 1
        return f"PID{i:04d}"

    mapping = {}
    enhanced = []
    for e in elements:
        desc = _norm_text(e.get('description') or '')
        pa = (e.get('primary_action') or '').lower().strip()
        bboxn = _quant_bbox_norm(e, img_w, img_h)
        grid = _grid_code(e, img_w, img_h)
        # compute perceptual dhash of the cropped element
        dh = ""
        try:
            crop = _crop_bbox(img, e)
            if crop is not None:
                dh = _dhash(crop)
        except Exception:
            dh = ""

        # Level-1 exact on (grid, text_norm, primary_action)
        pid = None
        candidates = idx_grid_text_role.get((grid, desc, pa), [])
        if candidates:
            pid = candidates[0].get('persistent_id')
        else:
            # Level-2 fuzzy: IoU + text similarity
            # Level-1.5: dHash
            if dh:
                best_pid, best_d = None, 65
                for it in items:
                    if (it.get('primary_action') or '') != pa:
                        continue
                    d = _hamming(dh, it.get('dhash') or '')
                    if d < best_d:
                        best_d, best_pid = d, it.get('persistent_id')
                if best_pid is not None and best_d <= 10:
                    pid = best_pid
            # Level-2: IoU + text similarity
            if not pid:
                best_pid, best_score = None, 0.0
                for it in items:
                    if (it.get('primary_action') or '') != pa:
                        continue
                    iou = _iou(bboxn, it.get('bbox_norm') or [0, 0, 0, 0])
                    sim = 0.0
                    try:
                        sim = SequenceMatcher(a=desc, b=it.get('text_norm') or '').ratio()
                    except Exception:
                        sim = 0.0
                    score = 0.7 * iou + 0.3 * sim
                    if iou >= 0.6 and sim >= 0.85 and score > best_score:
                        best_score, best_pid = score, it.get('persistent_id')
                pid = best_pid

        if not pid:
            pid = new_pid()
            item = {
                'persistent_id': pid,
                'text_norm': desc,
                'primary_action': pa,
                'bbox_norm': bboxn,
                'grid': grid,
                'dhash': dh,
                'times_seen': 1,
                'last_seen_step': step,
            }
            items.append(item)
            idx_grid_text_role.setdefault((grid, desc, pa), []).append(item)
        else:
            # Update existing
            for it in items:
                if it.get('persistent_id') == pid:
                    it['bbox_norm'] = bboxn
                    it['grid'] = grid
                    it['text_norm'] = desc or it.get('text_norm')
                    it['primary_action'] = pa or it.get('primary_action')
                    it['times_seen'] = int(it.get('times_seen', 0)) + 1
                    it['last_seen_step'] = step
                    if dh:
                        it['dhash'] = dh
                    break

        mapping[str(e.get('element_id'))] = pid
        # attach pid for downstream
        e2 = dict(e)
        e2['persistent_id'] = pid
        e2['grid'] = grid
        enhanced.append(e2)

    reg['items'] = items
    _save_registry(session_dir, reg)
    return mapping, enhanced


def main() -> int:
    print("\n==== Screen Understanding Session Runner ====")
    print("This is an interactive terminal runner. One task = one session.")
    print("Flow: 1) enter task (English) → 2) place 1.png → 3) results → 4) place 2.png → …; type 'q' anytime to quit.\n")

    # 1) Ask for task (English)
    task = ask("Please enter the overall task in English (or 'q' to quit): ").strip()
    if not task or task.lower() in {"q", "quit", "exit"}:
        print("Bye.")
        return 0

    # 2) Create session folder
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sessions_root = Path("pipeline_outputs") / "sessions"
    ensure_dir(sessions_root)
    session_dir = sessions_root / f"session_{ts}"
    ensure_dir(session_dir)
    (session_dir / "task.txt").write_text(task, encoding="utf-8")
    print(f"Session created: {session_dir}")
    print("Please place step screenshots named '1.png', '2.png', ... into this session folder.\n")

    # 3) Init pipeline and decision agent (lazy)
    cfg = ProcessingConfig(
        use_gpu=True,
        debug_mode=True,
        save_intermediate_results=False,
    )
    pipeline = ScreenUnderstandingPipeline(cfg)
    agent: Optional[DecisionAgent] = None
    dec_model = os.environ.get("DECIDER_MODEL") or getattr(cfg, "decider_model_name", "openai/gpt-oss-20b")

    # Thinking-only mode support: skip candidates and decision generation
    def _env_true(val: Optional[str]) -> bool:
        if val is None:
            return False
        return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}

    # Default ON to honor the request to keep only thinking
    thinking_only = _env_true(os.environ.get("THINKING_ONLY", "1")) or _env_true(os.environ.get("DW_THINKING_ONLY"))

    # Maintain session-level history
    actions_history = []  # list of {step, action, element_id, thoughts}
    last_thinking: Optional[str] = None

    step = 1
    while True:
        img_name = f"{step}.png"
        img_path = session_dir / img_name

        # 4) Wait for screenshot or quit
        print(f"[Step {step}] Put the screenshot at: {img_path}")
        cmd = ask("Press Enter when ready, or type 'q' to quit: ").strip().lower()
        if cmd in {"q", "quit", "exit"}:
            print("Session ended.")
            break

        if not img_path.exists():
            print("Image not found. Please check file name and location, then try again.\n")
            continue

        # 5) Process image
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Failed to open image: {e}")
            continue

        print(f"Processing {img_name} …")
        try:
            understanding = asyncio.run(pipeline.process(img, str(img_path)))
        except Exception as e:
            print(f"Processing failed: {e}")
            continue

        # 6) Save outputs into the session
        step_prefix = session_dir / f"step_{step}"
        ensure_dir(step_prefix)
        # Copy the input screenshot into the step folder for self-contained records
        input_dst = step_prefix / f"step_{step}_input.png"
        try:
            input_dst.write_bytes(img_path.read_bytes())
        except Exception:
            pass


        # Save understanding JSON inside the step folder (with persistent_id), deterministically sorted
        out_understanding = step_prefix / f"step_{step}_screen_understanding.json"
        understanding_out = dict(understanding)

        def _sort_key(e: dict):
            bx = e.get('bbox') or {}
            y = float(bx.get('y', 0))
            x = float(bx.get('x', 0))
            # tie-breakers: grid, description, primary_action, element_id, persistent_id
            return (
                round(y, 1),
                round(x, 1),
                str(e.get('grid') or ''),
                _norm_text(e.get('description') or ''),
                str(e.get('primary_action') or ''),
                str(e.get('element_id') or ''),
                str(e.get('persistent_id') or ''),
            )

        # understanding_out['elements'] = sorted(elems_with_pid, key=_sort_key)
        Path(out_understanding).write_text(json.dumps(understanding_out, ensure_ascii=False, indent=2), encoding="utf-8")

        # Copy annotated PNG from pipeline_outputs if available
        basename = img_path.stem
        anno_src = Path("pipeline_outputs") / f"{basename}_element_actions.png"
        anno_dst = step_prefix / f"step_{step}_element_actions.png"
        try:
            if anno_src.exists():
                anno_dst.write_bytes(anno_src.read_bytes())
        except Exception:
            pass

        # 7) Planning (pre-decision): produce/update thinking plan based on current UI and history
        thinking_obj = {}
        try:
            agent = getattr(pipeline, 'decision_agent', None)
            if agent is None:
                agent = DecisionAgent(model_name=dec_model, use_gpu=cfg.use_gpu)
            thinking_obj = agent.plan(understanding_out, task, actions_history=actions_history, prev_thinking=last_thinking)
        except Exception as e:
            thinking_obj = {"plan": f"planning failed: {e}", "steps": [], "success_criteria": []}

        # Save structured thinking JSON (dict) to file
        think_path = step_prefix / f"step_{step}_thinking.json"
        try:
            Path(think_path).write_text(json.dumps(thinking_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            # last-resort: write string form
            Path(think_path).write_text(str(thinking_obj), encoding="utf-8")
        # Update session-level latest thinking (text-only, for prompt continuity)
        last_thinking = str(thinking_obj.get('plan') or '').strip()

        # Generate annotated image for planned actions
        actions_img_path = step_prefix / f"step_{step}_thinking_actions.png"
        try:
            from decision_agent import DecisionAgent as _DA
            ann = _DA.render_thinking_actions(img, understanding_out, thinking_obj)
            ann.save(actions_img_path, format='PNG')
        except Exception as e:
            # Non-fatal; proceed without image
            pass

        # 8/9) Optionally skip candidates and decision in thinking-only mode
        cand_path = step_prefix / f"step_{step}_candidates.json"
        dec_path = step_prefix / f"step_{step}_decision.json"
        decision_obj = {}
        if not thinking_only:
            # Candidates (enumerate possible next actions on current screen)
            candidates = []
            try:
                agent = getattr(pipeline, 'decision_agent', None) or agent
                if agent is None:
                    agent = DecisionAgent(model_name=dec_model, use_gpu=cfg.use_gpu)
                candidates = agent.suggest_actions(understanding_out, task, actions_history=actions_history, last_thinking=last_thinking)
            except Exception as e:
                candidates = [{"error": f"candidates failed: {e}"}]

            Path(cand_path).write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")

            # Decision (use full actions history + last step thinking)
            try:
                agent = getattr(pipeline, 'decision_agent', None) or agent
                if agent is None:
                    agent = DecisionAgent(model_name=dec_model, use_gpu=cfg.use_gpu)
                decision_obj = agent.decide(understanding_out, task, actions_history=actions_history, last_thinking=last_thinking)
            except Exception as e:
                decision_obj = {"error": f"decision failed: {e}"}

            Path(dec_path).write_text(json.dumps({"task": task, "decision": decision_obj}, ensure_ascii=False, indent=2), encoding="utf-8")


        # Save rolling history for convenience
        (session_dir / "actions_history.json").write_text(
            json.dumps(actions_history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (session_dir / "thinking_latest.txt").write_text(last_thinking or "", encoding="utf-8")

        # 10) Summary
        print("\n=== Step Completed ===")
        print(f"Input: {input_dst}")
        print(f"Understanding: {out_understanding}")
        if Path(anno_dst).exists():
            print(f"Annotation: {anno_dst}")
        print(f"Thinking: {think_path}")
        if (step_prefix / f"step_{step}_thinking_actions.png").exists():
            print(f"Thinking Actions: {actions_img_path}")
        print("=================\n")

        step += 1

    print(f"Session folder: {session_dir}")
    print("To continue this session later, run this tool again and add subsequent images (3.png, 4.png, …) to this folder.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
