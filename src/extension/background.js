const GATEWAY_URL = "lol";

async function setupMenus() {
    await browser.contextMenus.removeAll();

    browser.contextMenus.create({
        id: "analyze-image",
        title: "Analyze image with AID",
        contexts: ["image"]
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
        const base64 = await fetchAndCompress(info.srcUrl);
        payload.image = base64;
        previewData = {type: 'image', content: info.srcUrl};
    } else if (info.menuItemId === "analyze-text") {
        endpoint = "/analyze_text";
        payload.text = info.selectionText;
        previewData = {type: 'text', content: info.selectionText};
    }

    await browser.storage.local.set({lastInput: previewData});

    performBackgroundAnalysis(endpoint, payload);
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
    browser.action.setBadgeText({text: "..."});
    browser.action.setBadgeBackgroundColor({color: "#aaaaaa"});

    try {
        const res = await fetch(GATEWAY_URL + endpoint, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error(`Server error: ${res.status}`);

        const data = await res.json();

        await browser.storage.local.set({lastResult: data});

        browser.action.setBadgeText({text: "!"});
        browser.action.setBadgeBackgroundColor({color: "#22c55e"});

        browser.notifications.create({
            type: "basic",
            iconUrl: "icons/icon-48.png",
            title: "AID Analysis Complete",
            message: `Result: ${data.label}`
        });

    } catch (err) {
        console.error("Analysis failed:", err);
        await browser.storage.local.set({
            lastResult: {label: "Error", confidence: 0, details: err.message}
        });
        browser.action.setBadgeText({text: "X"});
        browser.action.setBadgeBackgroundColor({color: "#ef4444"});
    }
}