document.addEventListener("DOMContentLoaded", async () => {
    const { history = [] } = await browser.storage.local.get("history");

    if (history.length > 0) {
        document.getElementById("idleState").style.display = "none";
        renderMainDisplay(history[0]);

        if (history.length > 1) {
            renderHistoryList(history.slice(1));
        }
    } else {
        document.getElementById("idleState").style.display = "block";
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

    document.getElementById("resultLabel").textContent = entry.result.label || "Error";
    document.getElementById("resultDetails").textContent = entry.result.details || entry.result.error || "";

    if (entry.result.confidence) {
        let current = 0;
        const target = Math.round(entry.result.confidence * 100);
        const animate = () => {
            current++;
            radial.style.setProperty("--value", current);
            text.textContent = `${current}%`;
            if (current < target) requestAnimationFrame(animate);
        };
        animate();
    }
}

function renderHistoryList(pastEntries) {
    const listContainer = document.getElementById("historyList");
    const section = document.getElementById("historySection");

    if (!listContainer) return;
    section.style.display = "block";
    listContainer.innerHTML = "";

    pastEntries.forEach((entry) => {
        const item = document.createElement("div");
        item.className = "history-item";
        item.style.cursor = "pointer";

        const typeIcon = entry.input.type === 'image' ? "🖼" : "|≡";
        const label = entry.result.label || "Unknown";
        const conf = Math.round((entry.result.confidence || 0) * 100);

        item.innerHTML = `
            <span>${typeIcon} <strong>${label}</strong></span>
            <span class="history-conf">${conf}%</span>
        `;

        item.addEventListener("click", () => {
            renderMainDisplay(entry);
            document.querySelectorAll(".history-item").forEach(el => el.classList.remove("selected"));
            item.classList.add("selected");
        });

        listContainer.appendChild(item);
    });
}

document.getElementById("clearHistory")?.addEventListener("click", async () => {
    await browser.storage.local.remove(["history", "lastInput", "lastResult"]);
    window.location.reload();
});