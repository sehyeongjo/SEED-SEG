import random
import json
import base64
import io
import threading
import importlib
from io import BytesIO
from pathlib import Path
from urllib.parse import quote

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageOps

app = FastAPI()

templates = Jinja2Templates(directory="app/templates")
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DEFAULT_DATA_ROOT = Path("/data")
ANGLE_JSON_PATH = APP_DIR / "angle.json"
VALID_IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
app.mount("/static", StaticFiles(directory="app/static"), name="static")

if Path("/data").exists():
    app.mount("/data", StaticFiles(directory="/data"), name="data")


TRAYSEG_JOBS = {}
TRAYSEG_JOBS_LOCK = threading.Lock()
TRAYSEG_CANCEL_FLAGS = {}
app.state.data_root = str(DEFAULT_DATA_ROOT.resolve())


def _get_data_root() -> Path:
    return Path(app.state.data_root)


def _get_data_tray_dir() -> Path:
    return _get_data_root() / "tray"


def _get_data_mask_dir() -> Path:
    return _get_data_root() / "mask_data"


def _get_trayseg_output_dir() -> Path:
    p = _get_data_root() / "trayseg_output"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _get_trayseg_control_dir() -> Path:
    p = _get_trayseg_output_dir() / ".control"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _to_preview_image(img: Image.Image) -> Image.Image:
    mode = img.mode
    if mode in {"I;16", "I", "F"}:
        gray = img.convert("I")
        minv, maxv = gray.getextrema()
        if maxv <= minv:
            return Image.new("RGB", img.size, (0, 0, 0))
        scaled = gray.point(lambda x: (x - minv) * 255.0 / (maxv - minv))
        scaled = scaled.convert("L")
        scaled = ImageOps.autocontrast(scaled, cutoff=1)
        return scaled.convert("RGB")

    if mode in {"L", "P", "1"}:
        return ImageOps.autocontrast(img.convert("L"), cutoff=1).convert("RGB")

    return img.convert("RGB")


def _read_angle_store() -> dict:
    if not ANGLE_JSON_PATH.exists():
        return {"angles": []}
    try:
        loaded = json.loads(ANGLE_JSON_PATH.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            if "angles" not in loaded or not isinstance(loaded["angles"], list):
                loaded["angles"] = []
            return loaded
    except json.JSONDecodeError:
        pass
    return {"angles": []}


def _saved_angle_for_tray(tray_id: str) -> float:
    store = _read_angle_store()
    for item in store.get("angles", []):
        if item.get("tray_num") == tray_id:
            try:
                return float(item.get("angle"))
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _cv_angle_from_saved(tray_id: str) -> float:
    # Web(CSS) rotate: + is clockwise, OpenCV rotate: + is counter-clockwise.
    # Keep UI angle as-is and invert only for OpenCV pipeline.
    return -_saved_angle_for_tray(tray_id)


def _load_tray_seg_module():
    return importlib.import_module("app.angle.tray_seg")


def _collect_overlay_preview_urls(tray_id: str, k: int = 3) -> list[str]:
    tray_out = _get_trayseg_output_dir() / tray_id
    if not tray_out.exists():
        return []

    candidates = sorted(tray_out.glob("*/debug/*_overlay_original.png"))
    if not candidates:
        candidates = sorted(tray_out.glob("*/debug/*_overlay_resized.png"))
    if not candidates:
        return []

    selected = random.sample(candidates, min(k, len(candidates)))
    urls = []
    for p in selected:
        rel = p.resolve().relative_to(_get_trayseg_output_dir().resolve()).as_posix()
        urls.append(f"/api/trayseg/files/{quote(rel, safe='/')}")
    return urls


def _has_template(tray_id: str) -> bool:
    template_dir = _get_trayseg_output_dir() / tray_id / "template"
    return (template_dir / "template_masks.npz").exists()


def _has_process_output(tray_id: str) -> bool:
    tray_out = _get_trayseg_output_dir() / tray_id
    if not tray_out.exists():
        return False
    for d in tray_out.iterdir():
        if not d.is_dir():
            continue
        if d.name == "template":
            continue
        if (d / "cells").exists() or (d / "debug").exists():
            return True
    return False


def _default_trayseg_status(tray_id: str) -> dict:
    if _cancel_file_requested(tray_id):
        return {
            "tray_num": tray_id,
            "stage": "cancelling",
            "message": "Cancellation requested...",
            "angle": _saved_angle_for_tray(tray_id),
            "template_progress": {"done": 0, "total": 0, "percent": 0.0},
            "process_progress": {"done": 0, "total": 0, "percent": 0.0},
            "overlay_images": [],
        }

    has_template = _has_template(tray_id)
    has_process = _has_process_output(tray_id)
    if has_process:
        stage = "process_done"
        message = "Segmentation output available."
    elif has_template:
        stage = "template_done"
        message = "Template available."
    else:
        stage = "idle"
        message = "Idle"
    return {
        "tray_num": tray_id,
        "stage": stage,
        "message": message,
        "angle": _saved_angle_for_tray(tray_id),
        "template_progress": {"done": 0, "total": 0, "percent": 0.0},
        "process_progress": {"done": 0, "total": 0, "percent": 0.0},
        "overlay_images": _collect_overlay_preview_urls(tray_id, k=3) if has_template else [],
    }


def _set_trayseg_status(tray_id: str, **updates):
    with TRAYSEG_JOBS_LOCK:
        current = TRAYSEG_JOBS.get(tray_id, _default_trayseg_status(tray_id))
        current.update(updates)
        TRAYSEG_JOBS[tray_id] = current


def _ensure_cancel_flag(tray_id: str) -> threading.Event:
    with TRAYSEG_JOBS_LOCK:
        flag = TRAYSEG_CANCEL_FLAGS.get(tray_id)
        if flag is None:
            flag = threading.Event()
            TRAYSEG_CANCEL_FLAGS[tray_id] = flag
        return flag


def _cancel_file_path(tray_id: str) -> Path:
    return _get_trayseg_control_dir() / f"{tray_id}.cancel"


def _cancel_file_requested(tray_id: str) -> bool:
    return _cancel_file_path(tray_id).exists()


def _cancel_requested(tray_id: str) -> bool:
    return _ensure_cancel_flag(tray_id).is_set() or _cancel_file_requested(tray_id)


def _clear_cancel_flag(tray_id: str):
    _ensure_cancel_flag(tray_id).clear()
    _cancel_file_path(tray_id).unlink(missing_ok=True)


def _request_cancel_flag(tray_id: str):
    _ensure_cancel_flag(tray_id).set()
    _cancel_file_path(tray_id).write_text("1", encoding="utf-8")


def _get_trayseg_status(tray_id: str) -> dict:
    with TRAYSEG_JOBS_LOCK:
        current = TRAYSEG_JOBS.get(tray_id)
        if current is None:
            current = _default_trayseg_status(tray_id)
            TRAYSEG_JOBS[tray_id] = current
            return dict(current)

        # Persist only active runtime states in memory.
        # For completed/idle states, always reflect current disk outputs.
        if current.get("stage") not in {"template_running", "process_running", "cancelling", "cancelled", "error"}:
            current = _default_trayseg_status(tray_id)
            TRAYSEG_JOBS[tray_id] = current
        return dict(current)


def _run_template_job(tray_id: str):
    try:
        _clear_cancel_flag(tray_id)
        tray_seg = _load_tray_seg_module()
        input_dir = _get_data_tray_dir() / tray_id
        tray_out = _get_trayseg_output_dir() / tray_id
        template_dir = tray_out / "template"
        tray_out.mkdir(parents=True, exist_ok=True)
        template_dir.mkdir(parents=True, exist_ok=True)
        angle = _saved_angle_for_tray(tray_id)
        cv_angle = _cv_angle_from_saved(tray_id)

        def progress(done, total, good, bad, current):
            if _cancel_requested(tray_id):
                raise RuntimeError("Cancelled by user")
            percent = (done / total * 100.0) if total else 0.0
            _set_trayseg_status(
                tray_id,
                stage="template_running",
                message=f"Building template... {done}/{total}",
                angle=angle,
                template_progress={"done": done, "total": total, "percent": percent, "good": good, "bad": bad, "current": current},
            )

        tray_seg.build_template_from_dataset(
            input_dir=input_dir,
            template_dir=template_dir,
            output_root=tray_out,
            manual_angle=cv_angle,
            progress_callback=progress,
            out_dir_suffix="",
            should_stop=lambda: _cancel_requested(tray_id),
        )

        overlays = _collect_overlay_preview_urls(tray_id, k=3)
        _set_trayseg_status(
            tray_id,
            stage="template_done",
            message="Template build complete.",
            overlay_images=overlays,
            template_progress={**_get_trayseg_status(tray_id)["template_progress"], "percent": 100.0},
        )
    except Exception as e:
        if "Cancelled by user" in str(e):
            _set_trayseg_status(tray_id, stage="cancelled", message="Template build cancelled.")
        else:
            _set_trayseg_status(tray_id, stage="error", message=f"Template build failed: {e}")
    finally:
        _clear_cancel_flag(tray_id)


def _run_process_job(tray_id: str):
    try:
        _clear_cancel_flag(tray_id)
        tray_seg = _load_tray_seg_module()
        input_dir = _get_data_tray_dir() / tray_id
        tray_out = _get_trayseg_output_dir() / tray_id
        template_dir = tray_out / "template"
        angle = _saved_angle_for_tray(tray_id)
        cv_angle = _cv_angle_from_saved(tray_id)

        def progress(done, total, ok, template, fail):
            if _cancel_requested(tray_id):
                raise RuntimeError("Cancelled by user")
            percent = (done / total * 100.0) if total else 0.0
            _set_trayseg_status(
                tray_id,
                stage="process_running",
                message=f"Processing segmentation... {done}/{total}",
                angle=angle,
                process_progress={"done": done, "total": total, "percent": percent, "ok": ok, "template_used": template, "fail": fail},
            )

        tray_seg.process_dataset_with_optional_template(
            input_dir=input_dir,
            output_root=tray_out,
            template_dir=template_dir,
            manual_angle=cv_angle,
            progress_callback=progress,
            out_dir_suffix="",
            should_stop=lambda: _cancel_requested(tray_id),
        )

        _set_trayseg_status(
            tray_id,
            stage="process_done",
            message="Segmentation process complete.",
            process_progress={**_get_trayseg_status(tray_id)["process_progress"], "percent": 100.0},
        )
    except Exception as e:
        if "Cancelled by user" in str(e):
            _set_trayseg_status(tray_id, stage="cancelled", message="Segmentation cancelled.")
        else:
            _set_trayseg_status(tray_id, stage="error", message=f"Segmentation failed: {e}")
    finally:
        _clear_cancel_flag(tray_id)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.get("/api/config/data-root")
def get_data_root_config():
    root = _get_data_root()
    return {"data_root": str(root), "exists": root.exists(), "is_dir": root.is_dir()}


@app.post("/api/config/data-root")
def set_data_root_config(payload: dict = Body(...)):
    data_root = str(payload.get("data_root", "")).strip()
    if not data_root:
        raise HTTPException(status_code=400, detail="data_root is required")

    root = Path(data_root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=400, detail=f"Directory does not exist: {root}")

    app.state.data_root = str(root)
    # Clear runtime statuses when switching root to avoid stale states from previous root.
    with TRAYSEG_JOBS_LOCK:
        TRAYSEG_JOBS.clear()
        TRAYSEG_CANCEL_FLAGS.clear()
    _get_trayseg_control_dir()
    return {"applied": True, "data_root": str(root)}


@app.get("/angle", response_class=HTMLResponse)
def angle_page(request: Request):
    return templates.TemplateResponse(
        "angle.html",
        {"request": request}
    )


@app.get("/mask", response_class=HTMLResponse)
def mask_page(request: Request):
    return templates.TemplateResponse(
        "mask.html",
        {"request": request}
    )


def _resolve_mask_data_path(rel_path: str) -> Path:
    p = (_get_data_mask_dir() / rel_path).resolve()
    try:
        p.relative_to(_get_data_mask_dir().resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid path") from exc
    return p


@app.get("/api/mask/image")
def mask_image(path: str):
    p = _resolve_mask_data_path(path)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path=p)


@app.get("/api/mask/browse/merged")
def mask_list_merged_folders(tray: str):
    base = _resolve_mask_data_path(f"{tray}/mask")
    if not base.exists() or not base.is_dir():
        return {"items": []}

    items = []
    for p in base.iterdir():
        if p.is_dir() and p.name.startswith("merged_") and p.name.endswith("_output"):
            items.append(p.name)
    items.sort()
    return {"items": items}


@app.get("/api/mask/trays")
def mask_list_trays():
    if not _get_data_mask_dir().exists() or not _get_data_mask_dir().is_dir():
        return {"trays": []}

    trays = sorted(
        p.name for p in _get_data_mask_dir().iterdir()
        if p.is_dir()
    )
    return {"trays": trays}


@app.get("/api/mask/browse/images")
def mask_list_images(tray: str, merged: str):
    base = _resolve_mask_data_path(f"{tray}/mask/{merged}")
    if not base.exists() or not base.is_dir():
        return {"items": []}

    items = []
    for p in base.glob("*.png"):
        if p.is_file():
            items.append(p.name)
    items.sort()
    return {"items": items}


@app.post("/api/mask/save")
def mask_save(payload: dict = Body(...)):
    rel_path = payload.get("path", "")
    png_base64 = payload.get("png_base64", "")
    if not rel_path or not png_base64:
        raise HTTPException(status_code=400, detail="path and png_base64 are required")

    p = _resolve_mask_data_path(rel_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    try:
        b64 = png_base64.split(",")[-1]
        raw = base64.b64decode(b64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid base64 image data") from exc

    try:
        im = Image.open(io.BytesIO(raw)).convert("L")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="failed to decode PNG image") from exc

    im = im.point(lambda v: 255 if v >= 128 else 0)
    im.save(p)
    return {"saved": str(p)}


@app.get("/trayseg", response_class=HTMLResponse)
def trayseg_page(request: Request):
    return templates.TemplateResponse(
        "trayseg.html",
        {"request": request}
    )


@app.get("/api/trays")
def list_trays():
    if not _get_data_tray_dir().exists():
        return {"trays": []}

    tray_names = sorted(
        entry.name
        for entry in _get_data_tray_dir().iterdir()
        if entry.is_dir()
    )
    return {"trays": tray_names}


@app.get("/api/trays/{tray_id}/preview")
def tray_preview(tray_id: str):
    tray_dir = _get_data_tray_dir() / tray_id
    if not tray_dir.exists() or not tray_dir.is_dir():
        raise HTTPException(status_code=404, detail="Tray not found")

    image_files = [
        path for path in tray_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTS
    ]
    if not image_files:
        return {"images": []}

    sample_count = min(3, len(image_files))
    selected = random.sample(image_files, sample_count)
    image_urls = []
    for img in selected:
        encoded_name = quote(img.name, safe="")
        image_urls.append(f"/api/trays/{tray_id}/preview-image/{encoded_name}")
    return {"images": image_urls}


@app.get("/api/trays/{tray_id}/preview-image/{image_name:path}")
def tray_preview_image(tray_id: str, image_name: str):
    tray_dir = _get_data_tray_dir() / tray_id
    if not tray_dir.exists() or not tray_dir.is_dir():
        raise HTTPException(status_code=404, detail="Tray not found")

    image_path = (tray_dir / image_name).resolve()
    tray_root = tray_dir.resolve()
    if tray_root not in image_path.parents or not image_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")

    suffix = image_path.suffix.lower()
    if suffix not in VALID_IMAGE_EXTS:
        raise HTTPException(status_code=400, detail="Unsupported image extension")

    if suffix in {".tif", ".tiff"}:
        with Image.open(image_path) as img:
            preview = _to_preview_image(img)
            out = BytesIO()
            preview.save(out, format="PNG")
        return Response(content=out.getvalue(), media_type="image/png")

    return FileResponse(path=image_path)


@app.post("/api/trays/{tray_id}/angle")
def save_tray_angle(tray_id: str, payload: dict = Body(...)):
    tray_dir = _get_data_tray_dir() / tray_id
    if not tray_dir.exists() or not tray_dir.is_dir():
        raise HTTPException(status_code=404, detail="Tray not found")

    try:
        angle = float(payload.get("angle"))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid angle value")

    output_path = ANGLE_JSON_PATH
    store = _read_angle_store()

    updated = False
    for item in store["angles"]:
        if item.get("tray_num") == tray_id:
            item["angle"] = angle
            updated = True
            break

    if not updated:
        store["angles"].append({
            "tray_num": tray_id,
            "angle": angle,
        })

    output_path.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"saved": True, "path": str(output_path), "data": {"tray_num": tray_id, "angle": angle}}


@app.get("/api/trays/{tray_id}/angle")
def get_tray_angle(tray_id: str):
    tray_dir = _get_data_tray_dir() / tray_id
    if not tray_dir.exists() or not tray_dir.is_dir():
        raise HTTPException(status_code=404, detail="Tray not found")

    store = _read_angle_store()
    angles = store.get("angles", [])

    for item in angles:
        if item.get("tray_num") == tray_id:
            try:
                angle = float(item.get("angle"))
            except (TypeError, ValueError):
                angle = 0.0
            return {"tray_num": tray_id, "angle": angle, "found": True}

    return {"tray_num": tray_id, "angle": 0.0, "found": False}


@app.post("/api/trayseg/{tray_id}/build-template")
def start_build_template(tray_id: str):
    tray_dir = _get_data_tray_dir() / tray_id
    if not tray_dir.exists() or not tray_dir.is_dir():
        raise HTTPException(status_code=404, detail="Tray not found")

    current = _get_trayseg_status(tray_id)
    if current["stage"] in {"template_running", "process_running"}:
        raise HTTPException(status_code=409, detail="A trayseg job is already running")

    _clear_cancel_flag(tray_id)
    _set_trayseg_status(
        tray_id,
        stage="template_running",
        message="Starting template build...",
        angle=_saved_angle_for_tray(tray_id),
        template_progress={"done": 0, "total": 0, "percent": 0.0},
        process_progress={"done": 0, "total": 0, "percent": 0.0},
        overlay_images=[],
    )
    thread = threading.Thread(target=_run_template_job, args=(tray_id,), daemon=True)
    thread.start()
    return _get_trayseg_status(tray_id)


@app.post("/api/trayseg/{tray_id}/process")
def start_process_segmentation(tray_id: str):
    tray_dir = _get_data_tray_dir() / tray_id
    if not tray_dir.exists() or not tray_dir.is_dir():
        raise HTTPException(status_code=404, detail="Tray not found")

    current = _get_trayseg_status(tray_id)
    if current["stage"] in {"template_running", "process_running"}:
        raise HTTPException(status_code=409, detail="A trayseg job is already running")

    template_dir = _get_trayseg_output_dir() / tray_id / "template"
    if not (template_dir / "template_masks.npz").exists():
        raise HTTPException(status_code=400, detail="Template does not exist. Run build_template first.")

    _clear_cancel_flag(tray_id)
    _set_trayseg_status(
        tray_id,
        stage="process_running",
        message="Starting segmentation process...",
        angle=_saved_angle_for_tray(tray_id),
        process_progress={"done": 0, "total": 0, "percent": 0.0},
    )
    thread = threading.Thread(target=_run_process_job, args=(tray_id,), daemon=True)
    thread.start()
    return _get_trayseg_status(tray_id)


@app.get("/api/trayseg/{tray_id}/status")
def trayseg_status(tray_id: str):
    tray_dir = _get_data_tray_dir() / tray_id
    if not tray_dir.exists() or not tray_dir.is_dir():
        raise HTTPException(status_code=404, detail="Tray not found")
    return _get_trayseg_status(tray_id)


@app.post("/api/trayseg/{tray_id}/cancel")
def cancel_trayseg_job(tray_id: str):
    tray_dir = _get_data_tray_dir() / tray_id
    if not tray_dir.exists() or not tray_dir.is_dir():
        raise HTTPException(status_code=404, detail="Tray not found")

    _request_cancel_flag(tray_id)
    _set_trayseg_status(tray_id, stage="cancelling", message="Cancellation requested...")
    return _get_trayseg_status(tray_id)


@app.get("/api/trayseg/files/{rel_path:path}")
def trayseg_file(rel_path: str):
    root = _get_trayseg_output_dir().resolve()
    file_path = (_get_trayseg_output_dir() / rel_path).resolve()
    if root not in file_path.parents or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=file_path)
