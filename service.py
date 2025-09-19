"""
service.py - Minimal local web service to capture screen and run the pipeline

Usage:
  python service.py  # opens http://127.0.0.1:5000

Features:
  - Simple HTML UI to choose capture mode (full / region / window / upload)
  - Enter a task; when provided, decision agent runs and saves decision JSON
  - Saves outputs under pipeline_outputs and shows links on the results page

Notes:
  - Window title capture works on Windows (exact title match)
  - On Linux Wayland, ImageGrab may not work; consider installing mss
"""
import io
import os
import json
import time
import asyncio
from pathlib import Path
from typing import Optional

from flask import Flask, request, send_from_directory, redirect
from PIL import Image

from models import ProcessingConfig
from pipeline import ScreenUnderstandingPipeline
from tools.capture import capture_full, capture_region, capture_window_by_title, list_window_titles


app = Flask(__name__)


@app.route("/")
def index():
    return (
        "<html><head><title>Screen Understanding Service</title>"
        "<style>body{font-family:sans-serif;max-width:920px;margin:24px auto;}"
        "fieldset{margin:12px 0;padding:12px;} label{display:block;margin:6px 0;} "
        "input[type=text]{width:100%;max-width:600px;} .row{display:flex;gap:16px;} "
        "select{max-width:600px;width:100%;} small{color:#666;}"
        "</style></head><body>"
        "<h2>Screen Understanding Service</h2>"
        "<form method='POST' action='/prepare' enctype='multipart/form-data'>"
        "<fieldset><legend>Capture Mode</legend>"
        "<label><input type='radio' name='mode' value='full' checked> Full screen</label>"
        "<label><input type='radio' name='mode' value='region'> Region (x,y,w,h)</label>"
        "<input type='text' name='region' placeholder='100,200,800,600'>"
        "<label><input type='radio' name='mode' value='window'> Window title (Windows)</label>"
        "<input type='text' id='window_title' name='window_title' placeholder='Untitled - Notepad'>"
        "<div class='row'>"
        "<button type='button' onclick='loadWindows()'>Refresh windows</button>"
        "<select id='win_list' size='6' onchange=\"document.getElementById('window_title').value=this.value\"></select>"
        "</div><small>Pick a title from the list to fill the input.</small>"
        "<label><input type='radio' name='mode' value='upload'> Upload image</label>"
        "<input type='file' name='image'>"
        "<div style='margin-top:8px;'>Or use <a href='/select' target='_blank'>interactive selector</a> (drag to pick a region)</div>"
        "<div style='margin-top:8px;'>Or <button type='button' onclick='doBrowserCapture()'>Capture via browser (best for WSL)</button>"
        " <small>Uses getDisplayMedia to capture your screen/window with permission.</small></div>"
        "</fieldset>"
        "<fieldset><legend>Task (optional)</legend>"
        "<input type='text' name='task' placeholder='e.g., Select LAX as departure city'>"
        "</fieldset>"
        "<fieldset class='row'><legend>Options</legend>"
        "<label>VLM model (optional)<br><input type='text' name='vlm_model' value='Qwen/Qwen2.5-VL-3B-Instruct' placeholder='Qwen/Qwen2.5-VL-3B-Instruct'></label>"
        "<label>Base name (optional)<br><input type='text' name='basename' placeholder='my_capture'></label>"
        "</fieldset>"
        "<button type='submit'>Preview</button>"
        "</form>"
        "<p>Outputs go to <code>pipeline_outputs</code> and will be linked on the results page.</p>"
        "<script>\n"
        "async function loadWindows(){\n"
        "  const sel = document.getElementById('win_list'); sel.innerHTML='';\n"
        "  try{\n"
        "    const res = await fetch('/windows'); const j = await res.json();\n"
        "    (j.titles||[]).forEach(t=>{ const o=document.createElement('option'); o.value=t; o.textContent=t; sel.appendChild(o); });\n"
        "  }catch(e){ console.log('loadWindows failed', e); }\n"
        "}\n"
        "window.addEventListener('load', loadWindows);\n"
        "async function doBrowserCapture(){\n"
        "  try{\n"
        "    const stream = await navigator.mediaDevices.getDisplayMedia({video:true});\n"
        "    const track = stream.getVideoTracks()[0];\n"
        "    const imageCapture = new ImageCapture(track);\n"
        "    let bitmap;\n"
        "    try{ bitmap = await imageCapture.grabFrame(); } catch(e){\n"
        "      // Fallback: draw from <video> element if grabFrame unsupported\n"
        "      const video = document.createElement('video'); video.srcObject=stream; await video.play();\n"
        "      await new Promise(r=> setTimeout(r, 200));\n"
        "      const c=document.createElement('canvas'); c.width=video.videoWidth; c.height=video.videoHeight;\n"
        "      c.getContext('2d').drawImage(video,0,0); bitmap = await createImageBitmap(c); video.pause();\n"
        "    }\n"
        "    const c=document.createElement('canvas'); c.width=bitmap.width; c.height=bitmap.height; c.getContext('2d').drawImage(bitmap,0,0);\n"
        "    track.stop();\n"
        "    const blob = await new Promise(res=> c.toBlob(res,'image/png',0.95));\n"
        "    const fd = new FormData(); fd.append('mode','upload'); fd.append('image', blob, 'browser_capture.png');\n"
        "    const taskEl = document.querySelector('input[name=task]'); if(taskEl && taskEl.value) fd.append('task', taskEl.value);\n"
        "    const mdlEl = document.querySelector('input[name=vlm_model]'); if(mdlEl && mdlEl.value) fd.append('vlm_model', mdlEl.value);\n"
        "    const baseEl = document.querySelector('input[name=basename]'); if(baseEl && baseEl.value) fd.append('basename', baseEl.value);\n"
        "    const resp = await fetch('/prepare', {method:'POST', body: fd}); const html = await resp.text();\n"
        "    const w = window.open('about:blank','_blank'); w.document.open(); w.document.write(html); w.document.close();\n"
        "  }catch(e){ alert('Browser capture failed: '+ e); }\n"
        "}\n"
        "</script>"
        "</body></html>"
    )


def _run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        try:
            import nest_asyncio

            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(coro)
        except Exception:
            raise


@app.route("/outputs/<path:filename>")
def outputs(filename):
    d = Path("pipeline_outputs").resolve()
    return send_from_directory(str(d), filename, as_attachment=False)


@app.post("/run")
def run_once():
    mode = request.form.get("mode", "full")
    region = request.form.get("region")
    window_title = request.form.get("window_title")
    task = request.form.get("task") or os.environ.get("DW_TASK")
    vlm_model = request.form.get("vlm_model")
    basename = request.form.get("basename")
    file = request.files.get("image")

    # Capture image
    img: Optional[Image.Image] = None
    base = basename.strip() if basename else "capture"
    if mode == "full":
        img = capture_full()
        base = f"{base}_full_{int(time.time())}"
    elif mode == "region" and region:
        try:
            x, y, w, h = [int(p.strip()) for p in region.split(',')]
        except Exception:
            return "Invalid region. Use x,y,w,h", 400
        img = capture_region(x, y, w, h)
        base = f"{base}_r_{x}_{y}_{w}_{h}_{int(time.time())}"
    elif mode == "window" and window_title:
        img = capture_window_by_title(window_title)
        safe = ''.join(c for c in window_title if c.isalnum() or c in ('-','_'))[:40]
        base = f"{base}_w_{safe}_{int(time.time())}"
    elif mode == "upload" and file:
        try:
            img = Image.open(io.BytesIO(file.read())).convert('RGB')
            base = f"{base}_upload_{int(time.time())}"
        except Exception:
            return "Failed to read uploaded image", 400
    else:
        return "Bad request: select a capture mode and provide required fields.", 400

    # Configure pipeline
    cfg = ProcessingConfig(debug_mode=True, save_intermediate_results=False)
    if vlm_model:
        cfg.vlm_model_name = vlm_model
    cfg.decider_enabled = True if task else False

    pipe = ScreenUnderstandingPipeline(cfg)

    # Run pipeline
    understanding = _run_async(pipe.process(img, base))

    # Save JSON alongside PNG
    out_dir = Path('pipeline_outputs'); out_dir.mkdir(exist_ok=True)
    json_path = out_dir / f"{base}_screen_understanding_output.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(understanding, f, indent=2, ensure_ascii=False)

    # Decision (optional)
    decision = None
    if task and isinstance(understanding, dict):
        try:
            from decision_agent import DecisionAgent

            agent = getattr(pipe, 'decision_agent', None) or DecisionAgent(model_name=cfg.decider_model_name, use_gpu=cfg.use_gpu)
            decision = agent.decide(understanding, task)
            dec_path = out_dir / f"{base}_decision.json"
            with open(dec_path, 'w', encoding='utf-8') as f:
                json.dump({'task': task, 'decision': decision}, f, indent=2, ensure_ascii=False)
        except Exception as e:
            decision = {'error': str(e)}

    # Build result HTML
    png_rel = f"{base}_element_actions.png"
    html = [
        "<html><head><title>Results</title>"
        "<style>body{font-family:sans-serif;max-width:920px;margin:24px auto;} code{padding:2px 4px;background:#f4f4f4;}</style>"
        "</head><body>",
        f"<h3>Completed: {base}</h3>",
        f"<p>JSON: <a href='/outputs/{json_path.name}' target='_blank'>{json_path.name}</a></p>",
        f"<p>PNG: <a href='/outputs/{png_rel}' target='_blank'>{png_rel}</a></p>",
    ]
    if decision is not None:
        dec_name = f"{base}_decision.json"
        html.append(f"<p>Decision: <a href='/outputs/{dec_name}' target='_blank'>{dec_name}</a></p>")
        if isinstance(decision, dict) and not decision.get('error'):
            html.append("<pre>" + json.dumps(decision, ensure_ascii=False, indent=2) + "</pre>")
        elif isinstance(decision, dict):
            html.append("<pre style='color:#a00'>Decision error: " + decision.get('error','') + "</pre>")
    html.append("<p><a href='/'>Run another</a></p>")
    html.append("</body></html>")
    return "".join(html)


# New: prepare step to capture/upload and show a preview before running the pipeline
@app.post('/prepare')
def prepare_capture():
    mode = request.form.get("mode", "full")
    region = request.form.get("region")
    window_title = request.form.get("window_title")
    task = request.form.get("task") or os.environ.get("DW_TASK")
    vlm_model = request.form.get("vlm_model")
    basename = (request.form.get("basename") or "capture").strip()
    file = request.files.get("image")

    out_dir = Path('pipeline_outputs'); out_dir.mkdir(exist_ok=True)
    ts = int(time.time())
    base = f"{basename}_prep_{ts}"
    image_name = f"{base}.png"
    image_path = out_dir / image_name

    # Capture/save image only (no processing yet)
    try:
        if mode == 'full':
            img = capture_full()
        elif mode == 'region' and region:
            x, y, w, h = [int(p.strip()) for p in region.split(',')]
            img = capture_region(x, y, w, h)
        elif mode == 'window' and window_title:
            img = capture_window_by_title(window_title)
        elif mode == 'upload' and file:
            img = Image.open(io.BytesIO(file.read())).convert('RGB')
        else:
            return "Bad request: select a capture mode and provide required fields.", 400
    except Exception as e:
        return f"Failed to capture image: {e}", 500

    img.save(image_path, format='PNG')

<<<<<<< ours
    # Render preview page with a Run button and progress bar, results appended below
    html = [
        "<html><head><title>Preview</title>",
        "<style>body{font-family:sans-serif;max-width:920px;margin:24px auto;} img{max-width:100%;height:auto;border:1px solid #ddd} #bar{width:100%;background:#eee;height:10px;border-radius:6px;overflow:hidden;margin:8px 0} #barfill{height:100%;width:0;background:#3b82f6;transition:width .3s} #progress{color:#555;margin:6px 0}</style>",
        "</head><body>",
        f"<h3>Preview: {image_name}</h3>",
        f"<p><img src='/outputs/{image_name}' alt='preview'></p>",
        f"<input type='hidden' id='prepared_image' value='{image_name}'>",
        f"Task (optional): <input type='text' id='task' value='{task or ''}' placeholder='e.g., Select LAX as departure city'> ",
        f"VLM model (optional): <input type='text' id='vlm_model' value='{vlm_model or 'Qwen/Qwen2.5-VL-3B-Instruct'}' placeholder='Qwen/Qwen2.5-VL-3B-Instruct'> ",
        f"Base name: <input type='text' id='basename' value='{basename}'> ",
<<<<<<< ours
        "<button id='run_btn'>Run</button> ",
        "<a href='/' style='margin-left:8px'>Back</a>",
        "<div id='progress'></div>",
        "<div id='bar'><div id='barfill'></div></div>",
        "<div id='result'></div>",
        "<script>\n",
        "const btn=document.getElementById('run_btn'); const bar=document.getElementById('barfill'); const prog=document.getElementById('progress'); const res=document.getElementById('result');\n",
        "btn.addEventListener('click', async ()=>{ try{ btn.disabled=true; res.innerHTML=''; let pct=0; prog.textContent='Starting...'; bar.style.width='5%'; const fd=new FormData(); fd.append('prepared_image', document.getElementById('prepared_image').value); fd.append('task', document.getElementById('task').value); fd.append('vlm_model', document.getElementById('vlm_model').value); fd.append('basename', document.getElementById('basename').value); let t = setInterval(()=>{ pct=Math.min(95, pct+3); bar.style.width=pct+'%'; }, 500); const r = await fetch('/run_prepared_ajax', {method:'POST', body: fd}); clearInterval(t); if(!r.ok){ throw new Error('HTTP '+r.status); } const j = await r.json(); bar.style.width='100%'; prog.textContent='Completed.'; let html=''; if(j.decision){ html += '<h4>Decision</h4><pre>'+JSON.stringify(j.decision, null, 2)+'</pre>'; } html += '<p>Understanding JSON: <a target=_blank href=\''+j.json_path_rel+'\'>'+j.json_name+'</a></p>'; if(j.png_path_rel){ html += '<p>Actions PNG: <a target=_blank href=\''+j.png_path_rel+'\'>'+j.png_name+'</a></p>'; } if(j.decision_png_rel){ html += '<p>Decision overlay: <a target=_blank href=\''+j.decision_png_rel+'\'>'+j.decision_png_name+'</a></p>'; } res.innerHTML = html; }catch(e){ prog.textContent='Failed: '+e; btn.disabled=false; } });\n",
=======
        f"<button id='run_btn'>Run</button> ",
        f"<button id='stop_btn' onclick=\"window.location='/'\">Stop</button>",
        "<div id='progress' class='muted'></div>",
        "</div>",
        "<div id='result'></div>",
        "<script>\n",
        f"const prepared='{image_name}';\n",
        "const runBtn=document.getElementById('run_btn'); const prog=document.getElementById('progress'); const res=document.getElementById('result');\n",
        "async function runNow(){\n",
        "  runBtn.disabled=true; prog.textContent='Running: OCR -> Elements -> VLM -> Decision ...'; res.innerHTML='';\n",
        "  const fd=new FormData(); fd.append('prepared_image', prepared); fd.append('task', document.getElementById('task').value); fd.append('vlm_model', document.getElementById('vlm_model').value); fd.append('basename', document.getElementById('basename').value);\n",
        "  try{\n",
        "    const r=await fetch('/run_prepared_api', {method:'POST', body:fd}); if(!r.ok){ throw new Error('HTTP '+r.status); }\n",
        "    const j=await r.json();\n",
        "    prog.textContent='Completed.';\n",
        "    let html='';\n",
        "    if(j.decision){ html += '<h4>Decision</h4>'; html += '<pre>'+JSON.stringify(j.decision, null, 2)+'</pre>'; }\n",
        "    html += '<p>Understanding JSON: <a target=_blank href=\''+j.json_path_rel+'\'>'+j.json_name+'</a></p>';\n",
        "    if(j.png_path_rel){ html += '<p>Actions PNG: <a target=_blank href=\''+j.png_path_rel+'\'>'+j.png_name+'</a></p>'; }\n",
        "    if(j.decision_png_rel){ html += '<p>Decision overlay: <a target=_blank href=\''+j.decision_png_rel+'\'>'+j.decision_png_name+'</a></p>'; }\n",
        "    if(j.decision && j.decision.element_id){ html += '<p><em>Please perform the action, then click Next to capture the updated screen.</em></p>'; }\n",
        "    html += '<button id=\'next_btn\'>Next (browser capture)</button> ';\n",
        "    html += '<button onclick=\"window.location=\'/'\'\">Stop</button>';\n",
        "    res.innerHTML=html;\n",
        "    const next=document.getElementById('next_btn'); if(next){ next.addEventListener('click', doBrowserCapture); }\n",
        "  }catch(e){ prog.textContent='Failed: '+e; runBtn.disabled=false; }\n",
        "}\n",
        "async function doBrowserCapture(){\n",
        "  try{\n",
        "    const stream = await navigator.mediaDevices.getDisplayMedia({video:true}); const track = stream.getVideoTracks()[0]; const imageCapture = new ImageCapture(track); let bitmap; try{ bitmap = await imageCapture.grabFrame(); } catch(e){ const video = document.createElement('video'); video.srcObject=stream; await video.play(); await new Promise(r=> setTimeout(r, 200)); const c=document.createElement('canvas'); c.width=video.videoWidth; c.height=video.videoHeight; c.getContext('2d').drawImage(video,0,0); bitmap = await createImageBitmap(c); video.pause(); } const c=document.createElement('canvas'); c.width=bitmap.width; c.height=bitmap.height; c.getContext('2d').drawImage(bitmap,0,0); track.stop(); const blob = await new Promise(res=> c.toBlob(res,'image/png',0.95)); const fd=new FormData(); fd.append('mode','upload'); fd.append('image', blob, 'browser_capture.png'); fd.append('task', document.getElementById('task').value); fd.append('vlm_model', document.getElementById('vlm_model').value); fd.append('basename', document.getElementById('basename').value); const resp = await fetch('/prepare', {method:'POST', body: fd}); const html = await resp.text(); const w = window.open('about:blank','_blank'); w.document.open(); w.document.write(html); w.document.close();\n",
        "  }catch(e){ alert('Browser capture failed: '+ e); }\n",
        "}\n",
        "runBtn.addEventListener('click', runNow);\n",
>>>>>>> theirs
        "</script>",
=======
    # Render preview page with a Run button
    html = [
        "<html><head><title>Preview</title>",
        "<style>body{font-family:sans-serif;max-width:920px;margin:24px auto;} img{max-width:100%;height:auto;border:1px solid #ddd}</style>",
        "</head><body>",
        f"<h3>Preview: {image_name}</h3>",
        f"<p><img src='/outputs/{image_name}' alt='preview'></p>",
        "<form method='POST' action='/run_prepared'>",
        f"<input type='hidden' name='prepared_image' value='{image_name}'>",
        f"Task (optional): <input type='text' name='task' value='{task or ''}' placeholder='e.g., Select LAX as departure city'>",
        f"&nbsp; VLM model (optional): <input type='text' name='vlm_model' value='{vlm_model or 'Qwen/Qwen2.5-VL-3B-Instruct'}' placeholder='Qwen/Qwen2.5-VL-3B-Instruct'>",
        f"&nbsp; Base name: <input type='text' name='basename' value='{basename}'>",
        "&nbsp; <button type='submit'>Run</button>",
        "</form>",
        "<p><a href='/'>Back</a></p>",
>>>>>>> theirs
        "</body></html>"
    ]
    return "".join(html)


@app.post('/run_prepared_ajax')
def run_prepared_ajax():
    prepared = request.form.get('prepared_image')
    task = request.form.get('task') or os.environ.get('DW_TASK')
    vlm_model = request.form.get('vlm_model')
    basename = (request.form.get('basename') or 'capture').strip()
    if not prepared:
        return jsonify({'error': 'Missing prepared image'}), 400
    out_dir = Path('pipeline_outputs'); out_dir.mkdir(exist_ok=True)
    img_path = out_dir / prepared
    if not img_path.exists():
        return jsonify({'error': 'Prepared image not found'}), 404

    # Load image and run pipeline
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        return jsonify({'error': f'Failed to open prepared image: {e}'}), 500

    cfg = ProcessingConfig(debug_mode=True, save_intermediate_results=False)
    if vlm_model:
        cfg.vlm_model_name = vlm_model
    cfg.decider_enabled = True if task else False
    pipe = ScreenUnderstandingPipeline(cfg)
    base = f"{basename}_{int(time.time())}"
    understanding = _run_async(pipe.process(img, base))

    # Save final JSON
    json_path = out_dir / f"{base}_screen_understanding_output.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(understanding, f, indent=2, ensure_ascii=False)

    # Optional decision
    decision = None
    if task and isinstance(understanding, dict):
        try:
            from decision_agent import DecisionAgent
            agent = getattr(pipe, 'decision_agent', None) or DecisionAgent(model_name=cfg.decider_model_name, use_gpu=cfg.use_gpu)
            decision = agent.decide(understanding, task)
            dec_path = out_dir / f"{base}_decision.json"
            with open(dec_path, 'w', encoding='utf-8') as f:
                json.dump({'task': task, 'decision': decision}, f, indent=2, ensure_ascii=False)
        except Exception as e:
            decision = {'error': str(e)}

<<<<<<< ours
    resp = {
        'json_name': json_path.name,
        'json_path_rel': f"/outputs/{json_path.name}",
        'png_name': f"{base}_element_actions.png",
        'png_path_rel': f"/outputs/{base}_element_actions.png",
        'decision': decision,
        'decision_png_name': None,
        'decision_png_rel': None,
    }
    return jsonify(resp)


@app.post('/run_prepared')
def run_prepared():
    prepared = request.form.get('prepared_image')
    task = request.form.get('task') or os.environ.get('DW_TASK')
    vlm_model = request.form.get('vlm_model')
    basename = (request.form.get('basename') or 'capture').strip()
    if not prepared:
        return "Missing prepared image", 400
    out_dir = Path('pipeline_outputs'); out_dir.mkdir(exist_ok=True)
    img_path = out_dir / prepared
    if not img_path.exists():
        return "Prepared image not found", 404

    # Load image and run pipeline
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception:
        return "Failed to open prepared image", 500

    cfg = ProcessingConfig(debug_mode=True, save_intermediate_results=False)
    if vlm_model:
        cfg.vlm_model_name = vlm_model
    cfg.decider_enabled = True if task else False
    pipe = ScreenUnderstandingPipeline(cfg)
    base = f"{basename}_{int(time.time())}"
    understanding = _run_async(pipe.process(img, base))

    # Save final JSON
    json_path = out_dir / f"{base}_screen_understanding_output.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(understanding, f, indent=2, ensure_ascii=False)

    # Optional decision
    decision = None
    if task and isinstance(understanding, dict):
        try:
            from decision_agent import DecisionAgent
            agent = getattr(pipe, 'decision_agent', None) or DecisionAgent(model_name=cfg.decider_model_name, use_gpu=cfg.use_gpu)
            decision = agent.decide(understanding, task)
            dec_path = out_dir / f"{base}_decision.json"
            with open(dec_path, 'w', encoding='utf-8') as f:
                json.dump({'task': task, 'decision': decision}, f, indent=2, ensure_ascii=False)
        except Exception as e:
            decision = {'error': str(e)}

<<<<<<< ours
=======
>>>>>>> theirs
    # Result page (same style as earlier)
    png_rel = f"{base}_element_actions.png"
    html = [
        "<html><head><title>Results</title></head><body>",
        f"<h3>Completed: {base}</h3>",
        f"<p>JSON: <a href='/outputs/{json_path.name}' target='_blank'>{json_path.name}</a></p>",
        f"<p>PNG: <a href='/outputs/{png_rel}' target='_blank'>{png_rel}</a></p>",
    ]
    if decision is not None:
        dec_name = f"{base}_decision.json"
        html.append(f"<p>Decision: <a href='/outputs/{dec_name}' target='_blank'>{dec_name}</a></p>")
        if isinstance(decision, dict) and not decision.get('error'):
            html.append("<pre>" + json.dumps(decision, ensure_ascii=False, indent=2) + "</pre>")
        elif isinstance(decision, dict):
            html.append("<pre style='color:#a00'>Decision error: " + decision.get('error','') + "</pre>")
    html.append("<p><a href='/'>Back</a></p>")
    html.append("</body></html>")
    return "".join(html)
<<<<<<< ours
=======
    # Decision overlay
    decision_png_name = None
    if isinstance(decision, dict) and not decision.get('error'):
        decision_png_name = _highlight_decision(base, understanding, decision)

    # Build JSON response for client rendering
    resp = {
        'json_name': json_path.name,
        'json_path_rel': f"/outputs/{json_path.name}",
        'png_name': f"{base}_element_actions.png",
        'png_path_rel': f"/outputs/{base}_element_actions.png",
        'decision': decision,
        'decision_png_name': decision_png_name,
        'decision_png_rel': (f"/outputs/{decision_png_name}" if decision_png_name else None),
    }
    return resp
>>>>>>> theirs
=======
>>>>>>> theirs


@app.get("/windows")
def list_windows_endpoint():
    try:
        titles = list_window_titles()
        return {"titles": titles}
    except Exception as e:
        return {"titles": [], "error": str(e)}


@app.get('/select')
def select_ui():
    # Capture a full-screen snapshot once for selection preview
    out_dir = Path('pipeline_outputs'); out_dir.mkdir(exist_ok=True)
    ts = int(time.time())
    snap_name = f"_select_preview_{ts}.png"
    snap_path = out_dir / snap_name
    try:
        img = capture_full()
        img.save(snap_path, format='PNG')
    except Exception as e:
        # Gracefully degrade: show an upload form and instructions
        return (
            "<html><head><title>Interactive Selector - Error</title></head><body>"
            "<h3>Failed to capture a snapshot from the desktop.</h3>"
            f"<p>Error: {str(e)}</p>"
            "<p>This can happen on headless servers or Wayland without permission. "
            "Use the main page to upload an image or try again on a local desktop.</p>"
            "<p><a href='/'>Back</a></p>"
            "</body></html>"
        )
    # Build HTML using plain strings for JS (avoid f-string braces issues)
    script = (
        "<script>\n"
        "const img = document.getElementById('img');\n"
        "const c = document.getElementById('overlay'); const ctx = c.getContext('2d');\n"
        "let start=null, end=null, scaleX=1, scaleY=1;\n"
        "function fitCanvas(){ c.width=img.clientWidth; c.height=img.clientHeight; }\n"
        "function draw(){ ctx.clearRect(0,0,c.width,c.height); if(!start||!end)return; const x=Math.min(start.x,end.x), y=Math.min(start.y,end.y); const w=Math.abs(end.x-start.x), h=Math.abs(end.y-start.y); ctx.strokeStyle='red'; ctx.lineWidth=2; ctx.strokeRect(x,y,w,h); ctx.fillStyle='rgba(255,0,0,0.15)'; ctx.fillRect(x,y,w,h);}\n"
        "img.addEventListener('load', ()=>{ fitCanvas(); const nw=img.naturalWidth, nh=img.naturalHeight; scaleX=nw/img.clientWidth; scaleY=nh/img.clientHeight; });\n"
        "window.addEventListener('resize', ()=>{ fitCanvas(); const nw=img.naturalWidth, nh=img.naturalHeight; scaleX=nw/img.clientWidth; scaleY=nh/img.clientHeight; draw(); });\n"
        "c.addEventListener('mousedown', e=>{ const r=c.getBoundingClientRect(); start={x:e.clientX-r.left,y:e.clientY-r.top}; end=null; draw(); });\n"
        "c.addEventListener('mousemove', e=>{ if(!start)return; const r=c.getBoundingClientRect(); end={x:e.clientX-r.left,y:e.clientY-r.top}; draw(); });\n"
        "c.addEventListener('mouseup', e=>{ const r=c.getBoundingClientRect(); end={x:e.clientX-r.left,y:e.clientY-r.top}; draw(); const x=Math.round(Math.min(start.x,end.x)*scaleX); const y=Math.round(Math.min(start.y,end.y)*scaleY); const w=Math.round(Math.abs(end.x-start.x)*scaleX); const h=Math.round(Math.abs(end.y-start.y)*scaleY); document.getElementById('x').value=x; document.getElementById('y').value=y; document.getElementById('w').value=w; document.getElementById('h').value=h; });\n"
        "</script>"
    )
    parts = [
        "<html><head><title>Interactive Selector</title>",
        "<style>body{font-family:sans-serif;max-width:96vw;margin:12px auto;} #wrap{position:relative;display:inline-block;}",
        "#overlay{position:absolute;left:0;top:0;} #panel{margin:8px 0;} input[type=text]{width:420px;}",
        "</style></head><body>",
        "<h3>Drag to select region on snapshot</h3>",
        "<div id='wrap'>",
        f"<img id='img' src='/outputs/{snap_name}' alt='snapshot'/>",
        "<canvas id='overlay'></canvas>",
        "</div>",
        "<div id='panel'>",
        "<form id='f' method='POST' action='/run_select'>",
        f"<input type='hidden' name='image_name' value='{snap_name}'>",
        "<input type='hidden' id='x' name='x'>",
        "<input type='hidden' id='y' name='y'>",
        "<input type='hidden' id='w' name='w'>",
        "<input type='hidden' id='h' name='h'>",
        "Task (optional): <input type='text' name='task' placeholder='e.g., Select LAX as departure city'>",
        "&nbsp; VLM model (optional): <input type='text' name='vlm_model' placeholder='Qwen/Qwen2.5-VL-7B-Instruct'>",
        "&nbsp; Base name: <input type='text' name='basename' placeholder='my_capture'>",
        "&nbsp; <button type='submit'>Run</button>",
        "</form>",
        "</div>",
        script,
        "</body></html>",
    ]
    return "".join(parts)


@app.post('/run_select')
def run_from_select():
    image_name = request.form.get('image_name')
    task = request.form.get('task') or os.environ.get('DW_TASK')
    vlm_model = request.form.get('vlm_model')
    basename = (request.form.get('basename') or 'capture').strip()
    try:
        x = int(request.form.get('x') or '0')
        y = int(request.form.get('y') or '0')
        w = int(request.form.get('w') or '0')
        h = int(request.form.get('h') or '0')
    except Exception:
        return "Invalid region", 400

    if not image_name:
        return "Missing image", 400
    out_dir = Path('pipeline_outputs'); out_dir.mkdir(exist_ok=True)
    snap_path = out_dir / image_name
    if not snap_path.exists():
        return "Snapshot not found", 404

    # Open snapshot and crop region
    try:
        img = Image.open(snap_path).convert('RGB')
        crop = img.crop((x, y, x + w, y + h)) if w > 0 and h > 0 else img
    except Exception:
        return "Failed to crop image", 500

    # Configure and run pipeline
    cfg = ProcessingConfig(debug_mode=True, save_intermediate_results=False)
    if vlm_model:
        cfg.vlm_model_name = vlm_model
    cfg.decider_enabled = True if task else False
    pipe = ScreenUnderstandingPipeline(cfg)
    base = f"{basename}_select_{int(time.time())}"
    understanding = _run_async(pipe.process(crop, base))

    # Save understanding JSON
    json_path = out_dir / f"{base}_screen_understanding_output.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(understanding, f, indent=2, ensure_ascii=False)

    # Decision (optional)
    decision = None
    if task and isinstance(understanding, dict):
        try:
            from decision_agent import DecisionAgent
            agent = getattr(pipe, 'decision_agent', None) or DecisionAgent(model_name=cfg.decider_model_name, use_gpu=cfg.use_gpu)
            decision = agent.decide(understanding, task)
            dec_path = out_dir / f"{base}_decision.json"
            with open(dec_path, 'w', encoding='utf-8') as f:
                json.dump({'task': task, 'decision': decision}, f, indent=2, ensure_ascii=False)
        except Exception as e:
            decision = {'error': str(e)}

    # Result page
    png_rel = f"{base}_element_actions.png"
    html = [
        "<html><head><title>Results</title></head><body>",
        f"<h3>Completed: {base}</h3>",
        f"<p>JSON: <a href='/outputs/{json_path.name}' target='_blank'>{json_path.name}</a></p>",
        f"<p>PNG: <a href='/outputs/{png_rel}' target='_blank'>{png_rel}</a></p>",
    ]
    if decision is not None:
        dec_name = f"{base}_decision.json"
        html.append(f"<p>Decision: <a href='/outputs/{dec_name}' target='_blank'>{dec_name}</a></p>")
        if isinstance(decision, dict) and not decision.get('error'):
            html.append("<pre>" + json.dumps(decision, ensure_ascii=False, indent=2) + "</pre>")
        elif isinstance(decision, dict):
            html.append("<pre style='color:#a00'>Decision error: " + decision.get('error','') + "</pre>")
    html.append("<p><a href='/'>Back</a></p>")
    html.append("</body></html>")
    return "".join(html)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
