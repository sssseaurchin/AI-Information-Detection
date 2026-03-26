document.addEventListener("DOMContentLoaded", async () => {
    const {lastInput, lastResult} = await browser.storage.local.get(["lastInput", "lastResult"]);

    if (lastInput) {
        document.getElementById("idleState").style.display = "none";
        const imgEl = document.getElementById("previewImage");
        const textEl = document.getElementById("previewText");

        if (lastInput.type === 'image') {
            imgEl.src = lastInput.content;
            imgEl.style.display = "block";
            textEl.style.display = "none";
        } else {
            textEl.textContent = `"${lastInput.content.substring(0, 100)}..."`;
            textEl.style.display = "block";
            imgEl.style.display = "none";
        }
    }

    if (lastResult) {
        browser.action.setBadgeText({text: ""});
        renderResult(lastResult);
    }
});

function renderResult(data) {
    const container = document.getElementById("resultDisplay");
    const radial = container.querySelector(".radial");
    const text = container.querySelector(".radial-text");

    container.style.display = "block";
    document.getElementById("resultLabel").textContent = data.label || "Error";
    document.getElementById("resultDetails").textContent = data.details || data.error || "";

    if (data.confidence) {
        let current = 0;
        const target = Math.round(data.confidence * 100);
        const animate = () => {
            current++;
            radial.style.setProperty("--value", current);
            text.textContent = `${current}%`;
            if (current < target) requestAnimationFrame(animate);
        };
        animate();
    }
}