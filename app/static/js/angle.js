const traySelect = document.getElementById("traySelect");
const refreshBtn = document.getElementById("refreshBtn");
const previewGrid = document.getElementById("previewGrid");
const statusEl = document.getElementById("status");
const editor = document.getElementById("editor");
const selectedImage = document.getElementById("selectedImage");
const selectedFile = document.getElementById("selectedFile");
const rotateMinusBtn = document.getElementById("rotateMinusBtn");
const rotatePlusBtn = document.getElementById("rotatePlusBtn");
const angleRange = document.getElementById("angleRange");
const angleValue = document.getElementById("angleValue");
const saveAngleBtn = document.getElementById("saveAngleBtn");

let currentAngle = 0;
let selectedImageUrl = "";
let selectedCard = null;

function setStatus(message) {
    statusEl.textContent = message;
}

function setAngle(newAngle) {
    const parsed = Number(newAngle);
    currentAngle = Math.max(-180, Math.min(180, parsed));
    angleRange.value = currentAngle.toFixed(1);
    angleValue.textContent = currentAngle.toFixed(1);
    selectedImage.style.transform = `translate(-50%, -50%) rotate(${currentAngle}deg)`;
}

function clearEditor() {
    editor.hidden = true;
    selectedImage.removeAttribute("src");
    selectedFile.textContent = "No image selected";
    selectedImageUrl = "";
    currentAngle = 0;
    angleRange.value = "0";
    angleValue.textContent = "0.0";
    selectedCard = null;
}

function selectImage(url, cardEl) {
    selectedImageUrl = url;
    selectedImage.src = `${url}${url.includes("?") ? "&" : "?"}t=${Date.now()}`;
    selectedFile.textContent = decodeURIComponent((url.split("/").pop() || url).split("?")[0]);
    editor.hidden = false;

    if (selectedCard) {
        selectedCard.classList.remove("selected");
    }
    selectedCard = cardEl;
    selectedCard.classList.add("selected");
    setAngle(0);
}

function renderPreviews(images) {
    previewGrid.innerHTML = "";
    clearEditor();

    if (!images.length) {
        setStatus("No images found in this tray.");
        return;
    }

    images.forEach((url) => {
        const card = document.createElement("div");
        card.className = "preview-card";

        const img = document.createElement("img");
        img.src = `${url}?t=${Date.now()}`;
        img.alt = "tray preview image";
        img.addEventListener("error", () => {
            img.alt = "preview load failed";
            img.style.background = "#fee2e2";
        });

        const meta = document.createElement("div");
        meta.className = "meta";
        const fileName = url.split("/").pop() || url;
        meta.textContent = decodeURIComponent(fileName.split("?")[0]);

        card.appendChild(img);
        card.appendChild(meta);
        card.addEventListener("click", () => selectImage(url, card));
        previewGrid.appendChild(card);
    });

    setStatus(`Showing ${images.length} random preview image(s).`);
}

async function loadTrays() {
    setStatus("Loading tray list...");
    try {
        const res = await fetch("/api/trays");
        if (!res.ok) {
            throw new Error(`Failed to load trays (${res.status})`);
        }

        const data = await res.json();
        data.trays.forEach((tray) => {
            const opt = document.createElement("option");
            opt.value = tray;
            opt.textContent = tray;
            traySelect.appendChild(opt);
        });

        if (!data.trays.length) {
            setStatus("No tray folders found in data root tray directory.");
            return;
        }

        setStatus("Select a tray to load random preview images.");
    } catch (err) {
        setStatus(err.message);
    }
}

async function loadPreview() {
    const trayId = traySelect.value;
    previewGrid.innerHTML = "";
    clearEditor();

    if (!trayId) {
        setStatus("Select a tray first.");
        return;
    }

    setStatus(`Loading preview for tray ${trayId}...`);
    try {
        const res = await fetch(`/api/trays/${encodeURIComponent(trayId)}/preview`);
        if (!res.ok) {
            throw new Error(`Failed to load preview (${res.status})`);
        }

        const data = await res.json();
        renderPreviews(data.images || []);
    } catch (err) {
        setStatus(err.message);
    }
}

traySelect.addEventListener("change", loadPreview);
refreshBtn.addEventListener("click", loadPreview);

rotateMinusBtn.addEventListener("click", () => setAngle(currentAngle - 1));
rotatePlusBtn.addEventListener("click", () => setAngle(currentAngle + 1));
angleRange.addEventListener("input", (event) => setAngle(event.target.value));

saveAngleBtn.addEventListener("click", async () => {
    const trayId = traySelect.value;
    if (!trayId) {
        setStatus("Select a tray first.");
        return;
    }
    if (!selectedImageUrl) {
        setStatus("Select one preview image first.");
        return;
    }

    setStatus(`Saving angle ${currentAngle.toFixed(1)} for tray ${trayId}...`);
    try {
        const res = await fetch(`/api/trays/${encodeURIComponent(trayId)}/angle`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({angle: currentAngle})
        });
        if (!res.ok) {
            throw new Error(`Failed to save angle (${res.status})`);
        }
        const message = `Saved angle ${currentAngle.toFixed(1)} to angle.json`;
        setStatus(message);
        alert(message);
    } catch (err) {
        setStatus(err.message);
    }
});

loadTrays();
