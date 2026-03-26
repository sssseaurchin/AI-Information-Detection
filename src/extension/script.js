let currentActiveEntry = null;

document.addEventListener("DOMContentLoaded", async () => {
    const {history = []} = await browser.storage.local.get("history");

    if (history.length === 0) {
        document.getElementById("idleState").style.display = "block";
        return;
    }

    document.getElementById("idleState").style.display = "none";
    browser.action.setBadgeText({text: ""});

    currentActiveEntry = history[0];
    renderMainDisplay(currentActiveEntry);

    if (history.length > 1) {
        renderHistoryList(history);
    }
});

function renderMainDisplay(entry) {
    const container = document.getElementById("resultDisplay");
    const imgEl = document.getElementById("previewImage");
    const textEl = document.getElementById("previewText");
    const radial = container.querySelector(".radial");
    const text = container.querySelector(".radial-text");

    container.style.display = "block";

    if (entry.input.type === 'image') {
        imgEl.src = entry.input.content;
        imgEl.style.display = "block";
        textEl.style.display = "none";
    } else {
        textEl.textContent = `"${entry.input.content.substring(0, 100)}..."`;
        textEl.style.display = "block";
        imgEl.style.display = "none";
    }

    document.getElementById("resultLabel").textContent = entry.result.label;
    document.getElementById("resultDetails").textContent = entry.result.details || entry.result.error || "";

    const val = Math.round((entry.result.confidence || 0) * 100);
    radial.style.setProperty("--value", val);
    text.textContent = `${val}%`;
}

function renderHistoryList(history) {
    const listContainer = document.getElementById("historyList");
    const section = document.getElementById("historySection");

    section.style.display = "block";
    listContainer.innerHTML = "";

    history.forEach((entry, index) => {
        const item = document.createElement("div");
        item.className = "history-item";
        if (entry === currentActiveEntry) item.classList.add("selected");

        item.innerHTML = `
            <span>${entry.input.type === 'image' ? "🖼" : "||"} <strong>${entry.result.label}</strong></span>
            <span class="history-conf">${Math.round(entry.result.confidence * 100)}%</span>
        `;

        item.addEventListener("click", () => {
            if (currentActiveEntry === entry && index !== 0) {
                currentActiveEntry = history[0];
            } else {
                currentActiveEntry = entry;
            }

            renderMainDisplay(currentActiveEntry);
            renderHistoryList(history);
        });

        listContainer.appendChild(item);
    });
}

document.getElementById("viewFullBtn").addEventListener("click", () => {
    if (!currentActiveEntry) return;

    const fullView = document.getElementById("fullViewModal");
    const content = document.getElementById("fullViewContent");

    let html = `<h2>${currentActiveEntry.result.label}</h2>`;
    html += `<p><strong>Confidence:</strong> ${Math.round(currentActiveEntry.result.confidence * 100)}%</p>`;

    if (currentActiveEntry.input.type === 'image') {
        html += `<img src="${currentActiveEntry.input.content}" style="max-width:100%; border:1px solid #ccc;">`;
    } else {
        html += `<div style="background:#eee; padding:10px; border-radius:4px; font-family:monospace;">${currentActiveEntry.input.content}</div>`;
    }

    html += `<hr><p>${currentActiveEntry.result.details || "No further details provided."}</p>`;

    content.innerHTML = html;
    fullView.style.display = "block";
});

document.getElementById("viewFullBtn").addEventListener("click", () => {
    if (!currentActiveEntry) return;

    const overlay = document.getElementById("fullViewModal");
    const content = document.getElementById("fullViewContent");

    let html = `<h3 style="margin-top:0;">${currentActiveEntry.result.label}</h3>`;
    html += `<p style="font-size:11px; color:var(--text-muted); margin-top:-10px; margin-bottom:15px;">Confidence: ${Math.round(currentActiveEntry.result.confidence * 100)}%</p>`;

    if (currentActiveEntry.input.type === 'image') {
        html += `<img src="${currentActiveEntry.input.content}" style="width:100%; border-radius:8px; margin-bottom:10px; border:1px solid var(--border-idle);">`;
    } else {
        html += `<div class="full-text-box">${currentActiveEntry.input.content}</div>`;
    }

    html += `<p style="font-size:12px; line-height:1.4; margin-top:15px;">${currentActiveEntry.result.details || "No details provided."}</p>`;

    content.innerHTML = html;
    overlay.classList.add("active");
});

document.getElementById("closeFullView").addEventListener("click", () => {
    const overlay = document.getElementById("fullViewModal");
    overlay.classList.remove("active");
});

document.getElementById("exportJson").addEventListener("click", (e) => {
    e.stopPropagation();
    if (!currentActiveEntry) return;

    const dataStr = JSON.stringify(currentActiveEntry, null, 2);
    const blob = new Blob([dataStr], {type: "application/json"});
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `AID_${currentActiveEntry.result.label}_${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(url);
});