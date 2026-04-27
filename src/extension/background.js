const GATEWAY_URL = "http://127.0.0.1:1337";

async function setupMenus() {
    await browser.contextMenus.removeAll();

    browser.contextMenus.create({
        id: "analyze-image",
        title: "Analyze image with AID",
        contexts: ["all"]
    });

    browser.contextMenus.create({
        id: "analyze-text",
        title: "Analyze selection with AID",
        contexts: ["selection"]
    });
}

browser.runtime.onInstalled.addListener(setupMenus);
browser.runtime.onStartup.addListener(setupMenus);

setupMenus();

browser.contextMenus.onClicked.addListener(async (info, tab) => {
    let endpoint = "";
    let payload = {};
    let previewData = {};

    if (info.menuItemId === "analyze-image") {
        endpoint = "/analyze_image";

        let targetUrl = info.srcUrl;

        try {
            const response = await browser.tabs.sendMessage(tab.id, {
                action: "GET_IMAGE_DATA"
            });

            if (response && response.url) {
                targetUrl = response.url;
            }
        } catch (e) {
            console.error("Content script not responding, using default URL.");
        }

        if (!targetUrl) {
            console.error("Could not find an image URL.");
            return;
        }

        const base64 = await fetchAndCompress(targetUrl);

        if (!base64) {
            console.error("Base64 conversion failed.");
            return;
        }

        payload.image = base64;
        previewData = { type: 'image', content: targetUrl };

    } else if (info.menuItemId === "analyze-text") {
        endpoint = "/analyze_text";
        payload.text = info.selectionText;
        previewData = { type: 'text', content: info.selectionText };
    }

    if (endpoint) {
        await browser.storage.local.set({ lastInput: previewData });
        performBackgroundAnalysis(endpoint, payload);
    }
});

async function fetchAndCompress(url) {
    try {
        const response = await fetch(url);
        const blob = await response.blob();
        return new Promise((resolve) => {
            const reader = new FileReader();
            reader.onloadend = () => resolve(reader.result.split(',')[1]);
            reader.readAsDataURL(blob);
        });
    } catch (e) {
        console.error("Failed to fetch image:", e);
        return null;
    }
}

async function performBackgroundAnalysis(endpoint, payload) {
    browser.action.setBadgeText({ text: "..." });
    browser.action.setBadgeBackgroundColor({ color: "#aaaaaa" });

    try {
        const response = await fetch(GATEWAY_URL + endpoint, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(payload),
            keepalive: true
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const data = await response.json();

        const { history = [] } = await browser.storage.local.get("history");
        const { lastInput } = await browser.storage.local.get("lastInput");

        const newEntry = {
            input: lastInput,
            result: data,
            timestamp: Date.now()
        };

        const updatedHistory = [newEntry, ...history].slice(0, 5);
        await browser.storage.local.set({ history: updatedHistory });

        browser.action.setBadgeText({ text: "!" });
        browser.action.setBadgeBackgroundColor({ color: "#22c55e" });

        browser.notifications.create({
            type: "basic",
            iconUrl: "icon.png",
            title: "Analysis Complete",
            message: `Result: ${data.label}`
        });

    } catch (err) {
        console.error("Detailed Analysis Error:", err);

        browser.action.setBadgeText({ text: "ERR" });
        browser.action.setBadgeBackgroundColor({ color: "#ef4444" });

        const { history = [] } = await browser.storage.local.get("history");
        const { lastInput } = await browser.storage.local.get("lastInput");

        const errorEntry = {
            input: lastInput,
            result: { label: "Error", confidence: 0, error: err.message },
            timestamp: Date.now()
        };

        await browser.storage.local.set({
            history: [errorEntry, ...history].slice(0, 5)
        });
    }
}

async function addToHistory(input, result) {
    const {history = []} = await browser.storage.local.get("history");

    const newEntry = {
        id: Date.now(),
        input: input, // {type: 'image'|'text', content: '...'}
        result: result, // {label: '...', confidence: 0.8}
        timestamp: new Date().toLocaleTimeString()
    };

    const updatedHistory = [newEntry, ...history].slice(0, 5);

    await browser.storage.local.set({
        history: updatedHistory,
        lastResult: result
    });
}

function getImageData(targetUrl) {
    const img = document.querySelector(`img[src="${targetUrl}"]`);
    if (!img) return null;

    try {
        const canvas = document.createElement("canvas");
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;

        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0);

        const dataUrl = canvas.toDataURL("image/jpeg", 1.0);
        return dataUrl.split(',')[1];
    } catch (e) {
        console.error("Canvas grab failed (likely strict CSP):", e);
        return null;
    }
}