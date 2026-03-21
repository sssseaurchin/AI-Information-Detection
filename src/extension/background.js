/**
 * background.js - AID Extension
 * Handles background tasks and persistent requests.
 */

const GATEWAY_URL = "lol"; // Placeholder - Match your script.js

// 1. Context Menu Setup (Runs on install/update)
browser.runtime.onInstalled.addListener(() => {
    browser.contextMenus.create({
        id: "analyze-image-aid",
        title: "Analyze image with AID",
        contexts: ["image"]
    });
});

// 2. Handle Context Menu Click
browser.contextMenus.onClicked.addListener(async (info, tab) => {
    if (info.menuItemId === "analyze-image-aid") {
        // Store image URL so the popup can grab it when opened
        await browser.storage.local.set({ externalImageUrl: info.srcUrl });
        // Optional: Notify user the image is ready for analysis in the popup
    }
});

// 3. Persistent Request Handler
// This allows the popup to "hand off" a request so it doesn't cancel if the popup closes.
browser.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === "START_ANALYSIS") {
        const { endpoint, payload } = message;

        // The background script stays "active" while this promise is pending
        performBackgroundAnalysis(endpoint, payload);

        // Return true to indicate we will respond asynchronously if needed
        return true;
    }
});

/**
 * Handles the actual fetch. Because this is in the background,
 * it won't be killed if the user closes the popup window.
 */
async function performBackgroundAnalysis(endpoint, payload) {
    console.log(`[AID BG] Starting analysis on ${endpoint}...`);

    try {
        const res = await fetch(GATEWAY_URL + endpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error(`Server error (${res.status})`);

        const data = await res.json();

        // Store the result so the popup can display it when reopened
        await browser.storage.local.set({ lastResult: data });

        // Notify the user the result is ready via a browser notification
        browser.notifications.create({
            type: "basic",
            iconUrl: "icon.png",
            title: "Analysis Complete",
            message: `Result: ${data.label} (${Math.round(data.confidence * 100)}%)`
        });

    } catch (err) {
        console.error("[AID BG] Request failed:", err);
        await browser.storage.local.set({ lastResult: { error: err.message } });
    } finally {
        console.log("[AID BG] Analysis finished. Background script returning to idle.");
    }
}