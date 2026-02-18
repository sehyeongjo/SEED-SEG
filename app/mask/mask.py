import base64
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
import io

DATA_ROOT = Path("/data/mask_data")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


def resolve_data_path(rel_path: str) -> Path:
    p = (DATA_ROOT / rel_path).resolve()
    try:
        p.relative_to(DATA_ROOT.resolve())
    except ValueError as exc:
        raise HTTPException(400, "invalid path") from exc
    return p

@app.get("/")
def index():
    return FileResponse("static/index.html")

@app.get("/image")
def get_image(path: str):
    p = resolve_data_path(path)
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


@app.get("/originals")
def list_originals():
    originals = []
    for p in DATA_ROOT.glob("*/original/**/*.png"):
        try:
            rel = p.resolve().relative_to(DATA_ROOT.resolve()).as_posix()
        except ValueError:
            continue
        originals.append(rel)
    originals.sort()
    return {"items": originals}


@app.get("/browse/merged")
def list_merged_folders(tray: str):
    base = resolve_data_path(f"{tray}/mask")
    if not base.exists() or not base.is_dir():
        return {"items": []}

    items = []
    for p in base.iterdir():
        if p.is_dir() and p.name.startswith("merged_") and p.name.endswith("_output"):
            items.append(p.name)
    items.sort()
    return {"items": items}


@app.get("/browse/images")
def list_images(tray: str, merged: str):
    base = resolve_data_path(f"{tray}/mask/{merged}")
    if not base.exists() or not base.is_dir():
        return {"items": []}

    items = []
    for p in base.glob("*.png"):
        if p.is_file():
            items.append(p.name)
    items.sort()
    return {"items": items}

class SaveReq(BaseModel):
    path: str
    png_base64: str

@app.post("/save")
def save_mask(req: SaveReq):
    p = resolve_data_path(req.path)
    p.parent.mkdir(parents=True, exist_ok=True)

    b64 = req.png_base64.split(",")[-1]
    raw = base64.b64decode(b64)

    im = Image.open(io.BytesIO(raw)).convert("L")
    im = im.point(lambda v: 255 if v >= 128 else 0)
    im.save(p)

    return {"saved": str(p)}
