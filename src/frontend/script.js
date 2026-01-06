const ALLOWED_IMAGE_TYPES = [
    "image/jpeg",
    "image/png",
    "image/webp"
];

const GATEWAY_URL = window.APP_CONFIG?.GATEWAY_URL;
if (!GATEWAY_URL) {
    console.error("frontend/script.js: GATEWAY_URL is missing. Check config.js");
}

const MAX_IMAGE_SIZE_MB = 5;
const URL_REGEX = /^https?:\/\/[^\s/$.?#].[^\s]*$/i;

function isValidHttpUrl(value) {
    return URL_REGEX.test(value.trim());
}

// inputs
const linkInput = document.querySelector(".input.wide");
const textInput = document.querySelector(".card .input");

// analyze buttons
const cardAnalyzeBtn = document.querySelector(".card .submit-btn");
const linkAnalyzeBtn = document.querySelector(".link-analyze");
const analyzeButtons = document.querySelectorAll(".submit-btn");

// dropzone
const dropzone = document.querySelector(".dropzone");
const dropzoneText = dropzone.querySelector("p");
const removeBtn = dropzone.querySelector(".remove-btn");
const uploadBtn = dropzone.querySelector(".upload-btn");

let droppedFile = null;
let cachedBase64Image = null;
let isSubmitting = false;

// hidden file input
const fileInput = document.createElement("input");
fileInput.type = "file";
fileInput.accept = ALLOWED_IMAGE_TYPES.join(",");
fileInput.style.display = "none";
document.body.appendChild(fileInput);

// analyze button
function updateAnalyzeVisibility() {
    const hasText = textInput.value.trim().length > 0;
    const hasImage = !!droppedFile;
    const hasLink = linkInput.value.trim().length > 0;

    cardAnalyzeBtn.classList.toggle("visible", hasText || hasImage);
    linkAnalyzeBtn.classList.toggle("visible", hasLink);
}

// image stuff
function validateImage(file) {
    if (!ALLOWED_IMAGE_TYPES.includes(file.type)) {
        alert("Only JPG, PNG or WEBP images are allowed.");
        return false;
    }

    if (file.size > MAX_IMAGE_SIZE_MB * 1024 * 1024) {
        alert(`Image size must be under ${MAX_IMAGE_SIZE_MB}MB.`);
        return false;
    }

    return true;
}

function renderImagePreview(file) {
    const img = document.createElement("img");
    img.style.maxWidth = "100%";
    img.style.maxHeight = "150px";
    img.style.objectFit = "contain";

    const reader = new FileReader();
    reader.onload = () => {
        img.src = reader.result;

        dropzone.classList.add("has-image");
        dropzoneText.style.display = "none";
        uploadBtn.style.display = "none";

        const oldImg = dropzone.querySelector("img");
        if (oldImg) oldImg.remove();

        dropzone.appendChild(img);
    };
    reader.readAsDataURL(file);
}

function handleImageFile(file) {
    if (!validateImage(file)) return;

    droppedFile = file;
    cachedBase64Image = null;

    renderImagePreview(file);
    updateAnalyzeVisibility();
}

function fileToBase64Cached(file) {
    if (cachedBase64Image) {
        return Promise.resolve(cachedBase64Image);
    }

    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
            cachedBase64Image = reader.result.split(",")[1];
            resolve(cachedBase64Image);
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

function resetImage() {
    droppedFile = null;
    cachedBase64Image = null;
    fileInput.value = "";

    const img = dropzone.querySelector("img");
    if (img) img.remove();

    dropzone.classList.remove("has-image");
    dropzoneText.style.display = "";
    uploadBtn.style.display = "";

    updateAnalyzeVisibility();
}

// input listeners
textInput.addEventListener("input", updateAnalyzeVisibility);
linkInput.addEventListener("input", updateAnalyzeVisibility);

// ENTER submits
textInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
        e.preventDefault();
        cardAnalyzeBtn.click();
    }
});

linkInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
        e.preventDefault();
        linkAnalyzeBtn.click();
    }
});

// dropzone stuff
uploadBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    fileInput.click();
});

dropzone.addEventListener("click", () => {
    fileInput.click();
});

dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
});

dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");

    const file = e.dataTransfer.files[0];
    if (!file) return;

    handleImageFile(file);
});

fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (!file) return;

    handleImageFile(file);
});

removeBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    resetImage();
});

// analyze button
cardAnalyzeBtn.addEventListener("click", () => {
    submitRequest({
        text: textInput.value || null,
        image: droppedFile || null
    });
});

linkAnalyzeBtn.addEventListener("click", () => {
    const value = linkInput.value.trim();
    if (!isValidHttpUrl(value)) {
        linkInput.classList.add("error");
        return;
    }

    linkInput.classList.remove("error");
    submitRequest({ text: value });
});

function setAnalyzeDisabled(disabled) {
    analyzeButtons.forEach(btn => {
        btn.disabled = disabled;
        btn.classList.toggle("loading", disabled);
    });
}
// input disable after trigger
function setInputsDisabled(disabled) {
    linkInput.disabled = disabled;
    textInput.disabled = disabled;

    dropzone.style.pointerEvents = disabled ? "none" : "";
    dropzone.style.opacity = disabled ? "0.6" : "";

    if (disabled) {
        linkInput.blur();
        textInput.blur();
    }
}

// overlay
function showOverlayResult({ label, confidence, details }) {
    const overlay = document.getElementById("overlay");
    const radial = overlay.querySelector(".radial");
    const text = overlay.querySelector(".radial-text");

    document.getElementById("resultLabel").textContent = label;
    document.getElementById("resultDetails").textContent = details ?? "";

    overlay.classList.add("show");
    overlay.setAttribute("aria-hidden", "false");

    if (typeof confidence !== "number") {
        radial.style.setProperty("--value", 0);
        text.textContent = "";
        radial.classList.add("error");
        return;
    }

    let current = 0;
    const target = Math.round(confidence * 100);

    const animate = () => {
        current += 1;
        radial.style.setProperty("--value", current);
        text.textContent = `${current}%`;

        if (current < target) requestAnimationFrame(animate);
    };
    animate();
}

const overlay = document.getElementById("overlay");

overlay.addEventListener("click", (e) => {
    if (e.target === overlay || e.target.classList.contains("overlay-close")) {
        overlay.classList.remove("show");
        overlay.setAttribute("aria-hidden", "true");
    }
});

document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
        overlay.classList.remove("show");
    }
});

function showError(message) {
    showOverlayResult({
        label: "Error",
        confidence: null,
        details: message
    });
}
// DEBUG...
(() => {
    const testBtn = document.getElementById("testOverlay");
    if (!testBtn) return;

    // start hidden
    testBtn.style.display = "none";

    let visible = false;

    document.addEventListener("keydown", (e) => {
        const isToggle =
            (e.ctrlKey || e.metaKey) &&
            e.shiftKey &&
            e.key.toLowerCase() === "d";

        if (!isToggle) return;

        e.preventDefault();
        visible = !visible;
        testBtn.style.display = visible ? "" : "none";

        console.log(
            `[DEBUG] Test overlay button ${visible ? "shown" : "hidden"}`
        );
    });
})();

document.getElementById("testOverlay")?.addEventListener("click", () => {
    const confidence = Math.random() * 0.6 + 0.2;

    showOverlayResult({
        label: confidence > 0.5 ? "Likely AI Generated" : "Likely Human",
        confidence,
        details: "This is dummy data used for UI testing."
    });
});
// ...DEBUG

// submit req
async function submitRequest({ text = null, image = null }) {
    if (isSubmitting) return;

    let endpoint = "";
    let payload = {};

    if (image) {
        endpoint = "/analyze_image";
        payload.image = await fileToBase64Cached(image);
    } else if (text && text.trim()) {
        endpoint = "/analyze_text";
        payload.text = text.trim();
    } else {
        return;
    }

    isSubmitting = true;
    setAnalyzeDisabled(true);
    setInputsDisabled(true);

    try {
        const res = await fetch(GATEWAY_URL + endpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!res.ok) {
            let msg = `Server error (${res.status})`;
            try {
                const err = await res.json();
                msg = err?.detail || err?.error || msg;
            } catch {}
            throw new Error(msg);
        }

        const data = await res.json();

        // Validate payload shape
        if (
            typeof data !== "object" ||
            typeof data.label !== "string" ||
            typeof data.confidence !== "number"
        ) {
            throw new Error("Invalid response from server");
        }

        showOverlayResult(data);

    } catch (err) {
        console.error("Request failed:", err);
        showError(err.message || "Something went wrong");
    } finally {
        isSubmitting = false;
        setAnalyzeDisabled(false);
        setInputsDisabled(false);
    }
}

// theme toggle
document.addEventListener("DOMContentLoaded", () => {
    const body = document.body;
    const toggle = document.getElementById("themeToggle");
    if (!toggle) return;

    const theme =
        localStorage.getItem("theme") ??
        (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");

    body.dataset.theme = theme;
    toggle.textContent = theme === "dark" ? "☀️" : "🌙";

    toggle.addEventListener("click", () => {
        const next = body.dataset.theme === "dark" ? "light" : "dark";
        body.dataset.theme = next;
        localStorage.setItem("theme", next);
        toggle.textContent = next === "dark" ? "☀️" : "🌙";
    });
});
