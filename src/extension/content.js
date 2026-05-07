let lastMouseX = 0;
let lastMouseY = 0;

document.addEventListener("contextmenu", (e) => {
    lastMouseX = e.clientX;
    lastMouseY = e.clientY;
}, true);

browser.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === "GET_IMAGE_DATA") {
        const elements = document.elementsFromPoint(lastMouseX, lastMouseY);

        const img = elements.find(el =>
            el.tagName === 'IMG' &&
            el.naturalWidth > 50
        );

        if (!img) {
            console.warn("AID: No substantial image found at click point.");
            sendResponse({ url: null });
        } else {
            sendResponse({ url: img.src });
        }
    }
    return true;
});

document.addEventListener("selectionchange", () => {
    const selection = window.getSelection().toString().trim();
    const hasSelection = selection.length > 0;

    browser.runtime.sendMessage({
        action: "TOGGLE_MENU",
        hideImageMenu: hasSelection
    });
});