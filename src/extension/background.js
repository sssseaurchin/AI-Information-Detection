const GATEWAY_URL = "http://127.0.0.1:1337";

// Global lock to prevent overlapping analysis requests
let isBusy = false;

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
    // 1. Guard Clause: Strictly prevent spamming if already processing
    if (isBusy) {
        browser.notifications.create({
            type: "basic",
            iconUrl: "icon.png",
            title: "Analysis in Progress",
            message: "Please wait for the current analysis to finish."
        });
        return;
    }

    let endpoint = "";
    let payload = {};
    let previewData = {};

    if (info.menuItemId === "analyze-image") {
        endpoint = "/analyze_image";
        let targetUrl = info.srcUrl;

        try {
            const response = await browser.tabs.sendMessage(tab.id, { action: "GET_IMAGE_DATA" });
            if (response && response.url) targetUrl = response.url;
        } catch (e) {
            console.error("Content script not responding, using default URL.");
        }

        if (!targetUrl) return;

        const base64 = await fetchAndCompress(targetUrl);
        if (!base64) return;

        payload.image = base64;
        previewData = { type: 'image', content: targetUrl };

    } else if (info.menuItemId === "analyze-text") {
        endpoint = "/analyze_text";
        payload.text = info.selectionText;
        previewData = { type: 'text', content: info.selectionText };
    }

    if (endpoint) {
        await browser.storage.local.set({ lastInput: previewData });
        // 2. Pass previewData into the analysis to maintain history integrity
        performBackgroundAnalysis(endpoint, payload, previewData);
    }
});

async function performBackgroundAnalysis(endpoint, payload, inputData) {
    // 3. Set the Lock
    isBusy = true;
    browser.action.setBadgeText({ text: "..." });
    browser.action.setBadgeBackgroundColor({ color: "#aaaaaa" });

    try {
        const response = await fetch(GATEWAY_URL + endpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
            keepalive: true
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const data = await response.json();

        // Use your existing helper for successful results
        await addToHistory(inputData, data);

        browser.action.setBadgeText({ text: "!" });
        browser.action.setBadgeBackgroundColor({ color: "#22c55e" });

        browser.notifications.create({
            type: "basic",
            iconUrl: "icon.png",
            title: "Analysis Complete",
            message: `Result: ${data.message}`
        });

    } catch (err) {
        console.error("Detailed Analysis Error:", err);
        browser.action.setBadgeText({ text: "ERR" });
        browser.action.setBadgeBackgroundColor({ color: "#ef4444" });

        // Maintain history even on error using your helper
        await addToHistory(inputData, { label: "Error", confidence: 0, error: err.message });
        
    } finally {
        // 4. THE UNLOCK: Guaranteed to run
        isBusy = false;
    }
}

// EXACT IMPLEMENTATION of your addToHistory helper
async function addToHistory(input, result) {
    const { history = [] } = await browser.storage.local.get("history");

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

// EXACT IMPLEMENTATION of your fetch helper
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