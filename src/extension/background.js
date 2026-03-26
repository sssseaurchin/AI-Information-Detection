const GATEWAY_URL = "lol";

browser.runtime.onInstalled.addListener(() => {
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
});

browser.contextMenus.onClicked.addListener(async (info, tab) => {
    let endpoint = "";
    let payload = {};
    let previewData = {};

    if (info.menuItemId === "analyze-image") {
        endpoint = "/analyze_image";
        const base64 = await fetchAndCompress(info.srcUrl);
        payload.image = base64;
        previewData = { type: 'image', content: info.srcUrl };
    }
    else if (info.menuItemId === "analyze-text") {
        endpoint = "/analyze_text";
        payload.text = info.selectionText;
        previewData = { type: 'text', content: info.selectionText };
    }

    await browser.storage.local.set({ lastInput: previewData });

    performBackgroundAnalysis(endpoint, payload);
});

async function fetchAndCompress(url) {
    const response = await fetch(url);
    const blob = await response.blob();
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result.split(',')[1]);
        reader.readAsDataURL(blob);
    });
}

async function performBackgroundAnalysis(endpoint, payload) {
    browser.action.setBadgeText({ text: "..." });
    try {
        const res = await fetch(GATEWAY_URL + endpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        await browser.storage.local.set({ lastResult: data });

        browser.action.setBadgeText({ text: "!" });
        browser.notifications.create({
            type: "basic",
            iconUrl: "icon.png",
            title: "AID Analysis Complete",
            message: `Result: ${data.label}`
        });
    } catch (err) {
        await browser.storage.local.set({ lastResult: { error: err.message } });
        browser.action.setBadgeText({ text: "ERR" });
    }
}