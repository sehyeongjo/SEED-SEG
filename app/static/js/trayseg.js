const traySelect = document.getElementById("traySelect");
const refreshBtn = document.getElementById("refreshBtn");
const statusEl = document.getElementById("status");
const angleBadge = document.getElementById("angleBadge");
const previewGrid = document.getElementById("previewGrid");
const buildTemplateBtn = document.getElementById("buildTemplateBtn");
const cancelJobBtn = document.getElementById("cancelJobBtn");
const segmentationBtn = document.getElementById("segmentationBtn");
const overlaySection = document.getElementById("overlaySection");
const overlayGrid = document.getElementById("overlayGrid");
const templateProgressFill = document.getElementById("templateProgressFill");
const templateProgressText = document.getElementById("templateProgressText");
const processProgressFill = document.getElementById("processProgressFill");
const processProgressText = document.getElementById("processProgressText");

let currentAngle = 0;
let pollTimer = null;
let lastStage = "";

function setStatus(message) {
    statusEl.textContent = message;
}

function setAngleBadge(angle, found) {
    currentAngle = Number(angle) || 0;
    if (found) {
        angleBadge.textContent = `Applied Angle: ${currentAngle.toFixed(1)}°`;
        return;
    }
    angleBadge.textContent = `Applied Angle: ${currentAngle.toFixed(1)}° (default)`;
}

function setProgress(fillEl, textEl, progress) {
    const done = Number(progress?.done || 0);
    const total = Number(progress?.total || 0);
    const percent = Number(progress?.percent || 0);
    fillEl.style.width = `${Math.max(0, Math.min(100, percent)).toFixed(1)}%`;
    if (total > 0) {
        textEl.textContent = `${percent.toFixed(1)}% (${done}/${total})`;
        return;
    }
    textEl.textContent = `${percent.toFixed(1)}%`;
}

function renderOverlayImages(urls, forceShow = false) {
    overlayGrid.innerHTML = "";
    if (!urls.length) {
        if (forceShow) {
            const empty = document.createElement("div");
            empty.className = "meta";
            empty.textContent = "No overlay preview image found.";
            overlayGrid.appendChild(empty);
            overlaySection.hidden = false;
            return;
        }
        overlaySection.hidden = true;
        return;
    }

    urls.forEach((url) => {
        const img = document.createElement("img");
        img.src = `${url}?t=${Date.now()}`;
        img.alt = "overlay preview";
        overlayGrid.appendChild(img);
    });
    overlaySection.hidden = false;
}

function applyStatusState(data) {
    const stage = data.stage || "idle";
    const running = stage === "template_running" || stage === "process_running" || stage === "cancelling";

    setProgress(templateProgressFill, templateProgressText, data.template_progress || {});
    setProgress(processProgressFill, processProgressText, data.process_progress || {});
    renderOverlayImages(data.overlay_images || [], stage === "template_done");

    buildTemplateBtn.disabled = running;
    cancelJobBtn.disabled = !running;
    segmentationBtn.disabled = !(stage === "template_done");

    if (stage === "error" || stage === "cancelled") {
        setStatus(data.message || "TraySeg error");
    }

    if (lastStage === "process_running" && stage === "process_done") {
        alert("Segmentation finished.");
    }
    lastStage = stage;
}

function stopPolling() {
    if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
    }
}

function startPolling() {
    stopPolling();
    pollTimer = setInterval(async () => {
        const trayId = traySelect.value;
        if (!trayId) {
            stopPolling();
            return;
        }
        try {
            const res = await fetch(`/api/trayseg/${encodeURIComponent(trayId)}/status`);
            if (!res.ok) {
                return;
            }
            const data = await res.json();
            applyStatusState(data);
            if (data.stage !== "template_running" && data.stage !== "process_running" && data.stage !== "cancelling") {
                stopPolling();
            }
        } catch (_) {
            // ignore polling errors
        }
    }, 1200);
}

function renderPreviews(images) {
    previewGrid.innerHTML = "";

    if (!images.length) {
        setStatus("No images found in this tray.");
        return;
    }

    images.forEach((url) => {
        const card = document.createElement("div");
        card.className = "preview-card";

        const canvas = document.createElement("div");
        canvas.className = "preview-canvas";

        const img = document.createElement("img");
        img.src = `${url}?t=${Date.now()}`;
        img.alt = "tray preview image";
        img.style.transform = `translate(-50%, -50%) rotate(${currentAngle}deg)`;
        img.addEventListener("error", () => {
            img.alt = "preview load failed";
            img.style.background = "#fee2e2";
        });

        const meta = document.createElement("div");
        meta.className = "meta";
        const fileName = url.split("/").pop() || url;
        meta.textContent = decodeURIComponent(fileName.split("?")[0]);

        canvas.appendChild(img);
        card.appendChild(canvas);
        card.appendChild(meta);
        previewGrid.appendChild(card);
    });

    setStatus(`Showing ${images.length} rotated preview image(s).`);
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

        setStatus("Select a tray to load rotated preview images.");
    } catch (err) {
        setStatus(err.message);
    }
}

async function loadTraySegPreview() {
    const trayId = traySelect.value;
    previewGrid.innerHTML = "";
    overlaySection.hidden = true;
    overlayGrid.innerHTML = "";
    setProgress(templateProgressFill, templateProgressText, {percent: 0});
    setProgress(processProgressFill, processProgressText, {percent: 0});
    stopPolling();
    lastStage = "";

    if (!trayId) {
        setStatus("Select a tray first.");
        buildTemplateBtn.disabled = true;
        segmentationBtn.disabled = true;
        return;
    }

    buildTemplateBtn.disabled = false;
    segmentationBtn.disabled = true;
    setStatus(`Loading tray ${trayId} preview and saved angle...`);
    try {
        const [previewRes, angleRes, statusRes] = await Promise.all([
            fetch(`/api/trays/${encodeURIComponent(trayId)}/preview`),
            fetch(`/api/trays/${encodeURIComponent(trayId)}/angle`),
            fetch(`/api/trayseg/${encodeURIComponent(trayId)}/status`)
        ]);

        if (!previewRes.ok) {
            throw new Error(`Failed to load preview (${previewRes.status})`);
        }
        if (!angleRes.ok) {
            throw new Error(`Failed to load angle (${angleRes.status})`);
        }
        if (!statusRes.ok) {
            throw new Error(`Failed to load trayseg status (${statusRes.status})`);
        }

        const previewData = await previewRes.json();
        const angleData = await angleRes.json();
        const statusData = await statusRes.json();
        setAngleBadge(angleData.angle, angleData.found);
        renderPreviews(previewData.images || []);
        applyStatusState(statusData);
        if (statusData.stage === "template_running" || statusData.stage === "process_running") {
            startPolling();
        }
    } catch (err) {
        setStatus(err.message);
    }
}

async function startBuildTemplate() {
    const trayId = traySelect.value;
    if (!trayId) {
        setStatus("Select a tray first.");
        return;
    }
    setProgress(templateProgressFill, templateProgressText, {done: 0, total: 0, percent: 0});
    setProgress(processProgressFill, processProgressText, {done: 0, total: 0, percent: 0});
    overlaySection.hidden = true;
    overlayGrid.innerHTML = "";
    buildTemplateBtn.disabled = true;
    segmentationBtn.disabled = true;
    cancelJobBtn.disabled = false;
    try {
        const res = await fetch(`/api/trayseg/${encodeURIComponent(trayId)}/build-template`, {method: "POST"});
        if (!res.ok) {
            throw new Error(`Failed to start build_template (${res.status})`);
        }
        const data = await res.json();
        applyStatusState(data);
        setStatus("Build template started.");
        startPolling();
    } catch (err) {
        setStatus(err.message);
    }
}

async function startSegmentation() {
    const trayId = traySelect.value;
    if (!trayId) {
        setStatus("Select a tray first.");
        return;
    }
    try {
        const res = await fetch(`/api/trayseg/${encodeURIComponent(trayId)}/process`, {method: "POST"});
        if (!res.ok) {
            throw new Error(`Failed to start process (${res.status})`);
        }
        const data = await res.json();
        applyStatusState(data);
        setStatus("Segmentation started.");
        startPolling();
    } catch (err) {
        setStatus(err.message);
    }
}

async function cancelJob() {
    const trayId = traySelect.value;
    if (!trayId) {
        setStatus("Select a tray first.");
        return;
    }
    try {
        const res = await fetch(`/api/trayseg/${encodeURIComponent(trayId)}/cancel`, {method: "POST"});
        if (!res.ok) {
            let detail = "";
            try {
                const errData = await res.json();
                detail = errData.detail ? `: ${errData.detail}` : "";
            } catch (_) {
                detail = "";
            }
            throw new Error(`Failed to cancel job (${res.status})${detail}`);
        }
        const data = await res.json();
        applyStatusState(data);
        setStatus("Cancellation requested...");
        cancelJobBtn.disabled = true;
        startPolling();
    } catch (err) {
        setStatus(err.message);
    }
}

traySelect.addEventListener("change", loadTraySegPreview);
refreshBtn.addEventListener("click", loadTraySegPreview);
buildTemplateBtn.addEventListener("click", startBuildTemplate);
segmentationBtn.addEventListener("click", startSegmentation);
cancelJobBtn.addEventListener("click", cancelJob);

buildTemplateBtn.disabled = true;
cancelJobBtn.disabled = true;
segmentationBtn.disabled = true;
loadTrays();
