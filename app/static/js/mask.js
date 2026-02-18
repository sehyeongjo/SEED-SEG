const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const previewCanvas = document.getElementById("previewCanvas");
const previewCtx = previewCanvas.getContext("2d");
const imgPathInput = document.getElementById("imgPath");
const traySelect = document.getElementById("traySelect");
const reloadTrayBtn = document.getElementById("reloadTrayBtn");
const mergedSelect = document.getElementById("mergedSelect");
const pngSelect = document.getElementById("pngSelect");
const applySelectionBtn = document.getElementById("applySelectionBtn");
const invertBtn = document.getElementById("invertBtn");
const saveMaskBtn = document.getElementById("saveMaskBtn");
const resetMaskBtn = document.getElementById("resetMaskBtn");
const saveExistingMaskBtn = document.getElementById("saveExistingMaskBtn");

const brushInput = document.getElementById("brushSize");
const brushNumberInput = document.getElementById("brushSizeNumber");
const brushValue = document.getElementById("brushValue");
const paintValueSelect = document.getElementById("paintValue");
const eraserToggle = document.getElementById("eraserToggle");
const showBrushOverlayInput = document.getElementById("showBrushOverlay");
const canvasViewModeSelect = document.getElementById("canvasViewMode");
const previewModeSelect = document.getElementById("previewMode");
const pathInfo = document.getElementById("pathInfo");

let img = new Image();
let maskCanvas = document.createElement("canvas");
let maskCtx = maskCanvas.getContext("2d");
let brushOverlayCanvas = document.createElement("canvas");
let brushOverlayCtx = brushOverlayCanvas.getContext("2d");

let baseMaskCanvas = document.createElement("canvas");
let baseMaskCtx = baseMaskCanvas.getContext("2d");
let existingMaskCanvas = document.createElement("canvas");
let existingMaskCtx = existingMaskCanvas.getContext("2d");

let currentOriginalPath = "";
let currentMaskPath = "";
let hasExistingMask = false;
let hasUnsavedEdits = false;
let previewOriginalPath = "";
let previewFolderMaskPath = "";
let previewOriginalImg = null;
let previewFolderMaskImg = null;

let drawing = false;
let brush = Number(brushInput.value) || 15;
let paintValue = Number(paintValueSelect.value) || 255;
let activePaintValue = paintValue;
let showCursor = false;
let cursorX = 0;
let cursorY = 0;

setBrush(brush);
updateEraserLabel();

brushInput.addEventListener("input", (e) => {
    setBrush(Number(e.target.value));
    render();
});

brushNumberInput.addEventListener("input", (e) => {
    setBrush(Number(e.target.value));
    render();
});

paintValueSelect.addEventListener("change", (e) => {
    paintValue = Number(e.target.value);
    activePaintValue = paintValue;
    updateEraserLabel();
    render();
});

eraserToggle.addEventListener("click", () => {
    paintValue = paintValue === 0 ? 255 : 0;
    activePaintValue = paintValue;
    paintValueSelect.value = String(paintValue);
    updateEraserLabel();
    render();
});

showBrushOverlayInput.addEventListener("change", () => {
    render();
});

canvasViewModeSelect.addEventListener("change", () => {
    render();
});

previewModeSelect.addEventListener("change", () => {
    renderPreview();
});

reloadTrayBtn.addEventListener("click", () => {
    loadTrays();
});

traySelect.addEventListener("change", () => {
    loadMergedFolders();
});

mergedSelect.addEventListener("change", () => {
    loadPngFiles();
});

pngSelect.addEventListener("change", () => {
    updateSelectedPath();
});

applySelectionBtn.addEventListener("click", async () => {
    updateSelectedPath();
    if (imgPathInput.value.trim()) {
        await loadImage();
    }
});
invertBtn.addEventListener("click", invert);
saveMaskBtn.addEventListener("click", saveMask);
resetMaskBtn.addEventListener("click", resetMaskEdits);
saveExistingMaskBtn.addEventListener("click", saveExistingMask);

canvas.addEventListener("wheel", (e) => {
    if (!e.ctrlKey) {
        return;
    }
    e.preventDefault();
    const delta = e.deltaY < 0 ? 1 : -1;
    setBrush(brush + delta);
    render();
}, {passive: false});

function clamp(v, min, max) {
    return Math.min(Math.max(v, min), max);
}

function setBrush(next) {
    const safe = Number.isFinite(next) ? clamp(Math.round(next), 1, 50) : brush;
    brush = safe;
    brushInput.value = String(safe);
    brushNumberInput.value = String(safe);
    brushValue.textContent = String(safe);
}

function updateEraserLabel() {
    eraserToggle.textContent = paintValue === 0 ? "Eraser: On" : "Eraser: Off";
}

function resolveImagePaths(inputPath) {
    if (inputPath.includes("/original/")) {
        return {
            originalPath: inputPath,
            maskPath: inputPath.replace("/original/", "/mask/"),
            loadExistingMask: true,
            displayPath: inputPath,
        };
    }

    if (inputPath.includes("/mask/")) {
        return {
            originalPath: inputPath.replace("/mask/", "/original/"),
            maskPath: inputPath,
            loadExistingMask: true,
            displayPath: inputPath,
        };
    }
    throw new Error("Path must include /original/ or /mask/ segment.");
}

function resolvePreviewFolderPaths(inputPath, fallbackOriginalPath) {
    if (inputPath.includes("/original/")) {
        return {
            originalPath: inputPath,
            maskPath: inputPath.replace("/original/", "/mask/"),
        };
    }

    if (inputPath.includes("/mask/")) {
        return {
            originalPath: inputPath.replace("/mask/", "/original/"),
            maskPath: inputPath,
        };
    }

    return {
        originalPath: fallbackOriginalPath,
        maskPath: "",
    };
}

function loadImageByPath(path) {
    return new Promise((resolve, reject) => {
        const loaded = new Image();
        loaded.onload = () => resolve(loaded);
        loaded.onerror = () => reject(new Error(`failed to load image: ${path}`));
        loaded.src = `/api/mask/image?path=${encodeURIComponent(path)}&v=${Date.now()}`;
    });
}

function loadImageFromDataUrl(dataUrl) {
    return new Promise((resolve, reject) => {
        const loaded = new Image();
        loaded.onload = () => resolve(loaded);
        loaded.onerror = () => reject(new Error("failed to load image from canvas"));
        loaded.src = dataUrl;
    });
}

function clearSelect(selectEl, placeholder) {
    selectEl.innerHTML = "";
    const option = document.createElement("option");
    option.value = "";
    option.textContent = placeholder;
    selectEl.appendChild(option);
}

async function loadMergedFolders() {
    const tray = traySelect.value.trim();
    if (!tray) {
        clearSelect(mergedSelect, "Select merged folder");
        clearSelect(pngSelect, "Select png");
        mergedSelect.disabled = true;
        pngSelect.disabled = true;
        applySelectionBtn.disabled = true;
        pathInfo.textContent = "Select tray first.";
        return;
    }

    clearSelect(mergedSelect, "Select merged folder");
    clearSelect(pngSelect, "Select png");
    mergedSelect.disabled = true;
    pngSelect.disabled = true;
    applySelectionBtn.disabled = true;

    const res = await fetch(`/api/mask/browse/merged?tray=${encodeURIComponent(tray)}`);
    if (!res.ok) {
        pathInfo.textContent = "Failed to load merged folders";
        return;
    }
    const data = await res.json();
    const items = Array.isArray(data.items) ? data.items : [];

    for (const name of items) {
        const option = document.createElement("option");
        option.value = name;
        option.textContent = name;
        mergedSelect.appendChild(option);
    }

    mergedSelect.disabled = items.length === 0;
    if (items.length === 0) {
        pathInfo.textContent = `No merged folders found under ${tray}/mask/`;
    }
}

async function loadPngFiles() {
    const tray = traySelect.value.trim();
    const merged = mergedSelect.value;
    clearSelect(pngSelect, "Select png");
    pngSelect.disabled = true;
    applySelectionBtn.disabled = true;

    if (!tray || !merged) {
        return;
    }

    const res = await fetch(`/api/mask/browse/images?tray=${encodeURIComponent(tray)}&merged=${encodeURIComponent(merged)}`);
    if (!res.ok) {
        pathInfo.textContent = "Failed to load png files";
        return;
    }
    const data = await res.json();
    const items = Array.isArray(data.items) ? data.items : [];

    for (const name of items) {
        const option = document.createElement("option");
        option.value = name;
        option.textContent = name;
        pngSelect.appendChild(option);
    }

    pngSelect.disabled = items.length === 0;
    if (items.length === 0) {
        pathInfo.textContent = `No png files found under ${tray}/mask/${merged}/`;
    }
}

function updateSelectedPath() {
    const tray = traySelect.value.trim();
    const merged = mergedSelect.value;
    const png = pngSelect.value;
    const valid = Boolean(tray && merged && png);
    applySelectionBtn.disabled = !valid;
    if (!valid) {
        return;
    }
    imgPathInput.value = `${tray}/mask/${merged}/${png}`;
}

function copyCanvas(src, dst, dstCtx) {
    dst.width = src.width;
    dst.height = src.height;
    dstCtx.clearRect(0, 0, dst.width, dst.height);
    dstCtx.drawImage(src, 0, 0);
}

async function loadImage() {
    const inputPath = imgPathInput.value.trim();
    if (!inputPath) {
        alert("Enter image path first");
        return;
    }

    try {
        const resolved = resolveImagePaths(inputPath);
        currentOriginalPath = resolved.originalPath;
        currentMaskPath = resolved.maskPath;
        const previewPaths = resolvePreviewFolderPaths(inputPath, currentOriginalPath);
        previewOriginalPath = previewPaths.originalPath;
        previewFolderMaskPath = previewPaths.maskPath;

        img = await loadImageByPath(resolved.displayPath);
        previewOriginalImg = img;
        previewFolderMaskImg = null;

        canvas.width = img.width;
        canvas.height = img.height;

        previewCanvas.width = img.width;
        previewCanvas.height = img.height;

        maskCanvas.width = img.width;
        maskCanvas.height = img.height;

        baseMaskCanvas.width = img.width;
        baseMaskCanvas.height = img.height;

        existingMaskCanvas.width = img.width;
        existingMaskCanvas.height = img.height;

        hasExistingMask = false;

        if (resolved.loadExistingMask) {
            try {
                const existingMask = await loadImageByPath(currentMaskPath);

                maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
                maskCtx.drawImage(existingMask, 0, 0, maskCanvas.width, maskCanvas.height);

                existingMaskCtx.clearRect(0, 0, existingMaskCanvas.width, existingMaskCanvas.height);
                existingMaskCtx.drawImage(existingMask, 0, 0, existingMaskCanvas.width, existingMaskCanvas.height);

                hasExistingMask = true;
            } catch {
                maskCtx.fillStyle = "black";
                maskCtx.fillRect(0, 0, img.width, img.height);
                existingMaskCtx.clearRect(0, 0, existingMaskCanvas.width, existingMaskCanvas.height);
            }
        } else {
            maskCtx.fillStyle = "black";
            maskCtx.fillRect(0, 0, img.width, img.height);
            existingMaskCtx.clearRect(0, 0, existingMaskCanvas.width, existingMaskCanvas.height);
        }

        if (previewOriginalPath) {
            try {
                previewOriginalImg = await loadImageByPath(previewOriginalPath);
            } catch {
                previewOriginalImg = null;
            }
        }

        if (resolved.displayPath === currentMaskPath) {
            canvasViewModeSelect.value = "mask";
        } else {
            canvasViewModeSelect.value = "original";
        }

        if (previewFolderMaskPath) {
            try {
                previewFolderMaskImg = await loadImageByPath(previewFolderMaskPath);
            } catch {
                previewFolderMaskImg = null;
            }
        }

        copyCanvas(maskCanvas, baseMaskCanvas, baseMaskCtx);

        brushOverlayCanvas.width = img.width;
        brushOverlayCanvas.height = img.height;
        brushOverlayCtx.clearRect(0, 0, img.width, img.height);

        pathInfo.textContent = `original: ${currentOriginalPath} | save mask: ${currentMaskPath} | preview mask: ${previewFolderMaskPath || "(none)"}`;
        hasUnsavedEdits = false;

        render();
        renderPreview();
    } catch (err) {
        alert(err.message || "failed to load image");
    }
}

async function loadTrays() {
    clearSelect(traySelect, "Select tray");
    traySelect.disabled = true;
    clearSelect(mergedSelect, "Select merged folder");
    clearSelect(pngSelect, "Select png");
    mergedSelect.disabled = true;
    pngSelect.disabled = true;
    applySelectionBtn.disabled = true;

    try {
        const res = await fetch("/api/mask/trays");
        if (!res.ok) {
            throw new Error(`Failed to load trays (${res.status})`);
        }
        const data = await res.json();
        const trays = Array.isArray(data.trays) ? data.trays : [];

        for (const tray of trays) {
            const option = document.createElement("option");
            option.value = tray;
            option.textContent = tray;
            traySelect.appendChild(option);
        }
        traySelect.disabled = trays.length === 0;
        pathInfo.textContent = trays.length
            ? "Select tray, merged folder, and png."
            : "No trays found in /data/mask_data.";
    } catch (err) {
        pathInfo.textContent = err.message || "Failed to load trays";
        traySelect.disabled = true;
    }
}

function render() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (canvasViewModeSelect.value === "mask") {
        ctx.drawImage(maskCanvas, 0, 0, canvas.width, canvas.height);
    } else if (previewOriginalImg) {
        ctx.drawImage(previewOriginalImg, 0, 0, canvas.width, canvas.height);
    } else {
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    }

    if (showBrushOverlayInput.checked && canvasViewModeSelect.value === "original") {
        ctx.drawImage(brushOverlayCanvas, 0, 0);
    }

    if (showCursor) {
        ctx.beginPath();
        ctx.arc(cursorX, cursorY, brush, 0, Math.PI * 2);
        ctx.strokeStyle = activePaintValue === 0 ? "#ff4d4f" : "#00e5ff";
        ctx.lineWidth = 2;
        ctx.stroke();
    }
}

function renderPreview() {
    previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);

    if (previewModeSelect.value === "mask") {
        if (hasUnsavedEdits) {
            previewCtx.drawImage(maskCanvas, 0, 0, previewCanvas.width, previewCanvas.height);
            return;
        }

        if (previewFolderMaskImg) {
            previewCtx.drawImage(previewFolderMaskImg, 0, 0, previewCanvas.width, previewCanvas.height);
        }
        return;
    }

    if (previewOriginalImg) {
        previewCtx.drawImage(previewOriginalImg, 0, 0, previewCanvas.width, previewCanvas.height);
    }
}

function getCanvasPos(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY,
    };
}

function applyBrush(x, y) {
    const gray = activePaintValue;
    maskCtx.fillStyle = `rgb(${gray}, ${gray}, ${gray})`;
    maskCtx.beginPath();
    maskCtx.arc(x, y, brush, 0, Math.PI * 2);
    maskCtx.fill();

    brushOverlayCtx.fillStyle = activePaintValue === 0 ? "rgba(255, 77, 79, 0.30)" : "rgba(0, 229, 255, 0.30)";
    brushOverlayCtx.beginPath();
    brushOverlayCtx.arc(x, y, brush, 0, Math.PI * 2);
    brushOverlayCtx.fill();

    hasUnsavedEdits = true;
    renderPreview();
}

canvas.onmousedown = (e) => {
    if (!currentOriginalPath) {
        return;
    }
    if (e.button !== 0 && e.button !== 2) {
        return;
    }

    activePaintValue = e.button === 2 ? 0 : 255;

    drawing = true;
    const pos = getCanvasPos(e);
    cursorX = pos.x;
    cursorY = pos.y;
    applyBrush(pos.x, pos.y);
    render();
};

canvas.onmouseup = () => {
    drawing = false;
    activePaintValue = paintValue;
};

canvas.oncontextmenu = (e) => {
    e.preventDefault();
};

canvas.onmouseleave = () => {
    drawing = false;
    showCursor = false;
    render();
};

canvas.onmouseenter = () => {
    showCursor = true;
    render();
};

canvas.onmousemove = (e) => {
    if (!currentOriginalPath) {
        return;
    }

    const pos = getCanvasPos(e);
    cursorX = pos.x;
    cursorY = pos.y;
    showCursor = true;

    if (drawing) {
        applyBrush(pos.x, pos.y);
    }

    render();
};

function invert() {
    if (!currentOriginalPath) {
        return;
    }

    const imgData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
    const d = imgData.data;

    for (let i = 0; i < d.length; i += 4) {
        const v = d[i] > 127 ? 0 : 255;
        d[i] = v;
        d[i + 1] = v;
        d[i + 2] = v;
        d[i + 3] = 255;
    }

    maskCtx.putImageData(imgData, 0, 0);

    brushOverlayCtx.clearRect(0, 0, brushOverlayCanvas.width, brushOverlayCanvas.height);
    hasUnsavedEdits = true;
    render();
    renderPreview();
}

function resetMaskEdits() {
    if (!currentOriginalPath) {
        return;
    }

    maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    maskCtx.drawImage(baseMaskCanvas, 0, 0);

    brushOverlayCtx.clearRect(0, 0, brushOverlayCanvas.width, brushOverlayCanvas.height);
    hasUnsavedEdits = false;
    render();
    renderPreview();
}

async function saveCanvasToMask(sourceCanvas) {
    const data = sourceCanvas.toDataURL("image/png");

    await fetch("/api/mask/save", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({path: currentMaskPath, png_base64: data})
    });
}

async function saveMask() {
    if (!currentMaskPath) {
        alert("load image first");
        return;
    }

    await saveCanvasToMask(maskCanvas);
    previewFolderMaskImg = await loadImageFromDataUrl(maskCanvas.toDataURL("image/png"));
    hasUnsavedEdits = false;
    renderPreview();
    alert(`saved: ${currentMaskPath}`);
}

async function saveExistingMask() {
    if (!currentMaskPath) {
        alert("load image first");
        return;
    }

    if (!hasExistingMask) {
        alert("no existing mask file was loaded");
        return;
    }

    await saveCanvasToMask(existingMaskCanvas);
    alert(`saved existing mask: ${currentMaskPath}`);
}

loadTrays();
