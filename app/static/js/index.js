const dataRootInput = document.getElementById("dataRootInput");
const applyDataRootBtn = document.getElementById("applyDataRootBtn");
const dataRootStatus = document.getElementById("dataRootStatus");

function setStatus(msg) {
    dataRootStatus.textContent = msg;
}

async function loadDataRoot() {
    setStatus("Loading current data root...");
    try {
        const res = await fetch("/api/config/data-root");
        if (!res.ok) {
            throw new Error(`Failed to load data root (${res.status})`);
        }
        const data = await res.json();
        dataRootInput.value = data.data_root || "";
        setStatus(`Current data root: ${data.data_root}`);
    } catch (err) {
        setStatus(err.message || "Failed to load data root");
    }
}

async function applyDataRoot() {
    const nextRoot = dataRootInput.value.trim();
    if (!nextRoot) {
        setStatus("Enter a valid data root path.");
        return;
    }
    applyDataRootBtn.disabled = true;
    setStatus(`Applying data root: ${nextRoot} ...`);
    try {
        const res = await fetch("/api/config/data-root", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({data_root: nextRoot}),
        });
        if (!res.ok) {
            let detail = "";
            try {
                const errData = await res.json();
                detail = errData.detail ? `: ${errData.detail}` : "";
            } catch (_) {
                detail = "";
            }
            throw new Error(`Failed to apply (${res.status})${detail}`);
        }
        const data = await res.json();
        dataRootInput.value = data.data_root || nextRoot;
        setStatus(`Applied data root: ${data.data_root}`);
    } catch (err) {
        setStatus(err.message || "Failed to apply data root");
    } finally {
        applyDataRootBtn.disabled = false;
    }
}

applyDataRootBtn.addEventListener("click", applyDataRoot);
loadDataRoot();
