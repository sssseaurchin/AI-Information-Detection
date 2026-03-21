const ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp"];
const GATEWAY_URL = "lol"; // Placeholder
const MAX_IMAGE_SIZE_MB = 5;

// Inputs
const textInput = document.querySelector(".card .input");
const cardAnalyzeBtn = document.querySelector(".card .submit-btn");
const analyzeButtons = document.querySelectorAll(".submit-btn");

// Dropzone
const dropzone = document.querySelector(".dropzone");
const dropzoneText = dropzone.querySelector("p");
const removeBtn = dropzone.querySelector(".remove-btn");
const uploadBtn = dropzone.querySelector(".upload-btn");

let droppedFile = null;
let cachedBase64Image = null;
let isSubmitting = false;

// Hidden file input
const fileInput = document.createElement("input");
fileInput.type = "file";
fileInput.accept = ALLOWED_IMAGE_TYPES.join(",");
fileInput.style.display = "none";
document.body.appendChild(fileInput);

// Toggle Analyze Button Visibility (Unified Text/Image check)
function updateAnalyzeVisibility() {
    const hasText = textInput.value.trim().length > 0;
    const hasImage = !!droppedFile;

    // Show button if there is ANY input (Text OR Image)
    cardAnalyzeBtn.classList.toggle("visible", hasText || hasImage);
}

// Image Logic (Preserving your original flow)
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
    if (cachedBase64Image) return Promise.resolve(cachedBase64Image);
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

// Listeners
textInput.addEventListener("input", updateAnalyzeVisibility);

textInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
        e.preventDefault();
        cardAnalyzeBtn.click();
    }
});

uploadBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    fileInput.click();
});

dropzone.addEventListener("click", () => fileInput.click());

dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));

dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file) handleImageFile(file);
});

fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) handleImageFile(fileInput.files[0]);
});

removeBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    resetImage();
});

// Submit Action (Your original combined logic)
cardAnalyzeBtn.addEventListener("click", () => {
    submitRequest({
        text: textInput.value || null,
        image: droppedFile || null
    });
});

function setInputsDisabled(disabled) {
    textInput.disabled = disabled;
    analyzeButtons.forEach(btn => {
        btn.disabled = disabled;
        btn.classList.toggle("loading", disabled);
    });
    dropzone.style.pointerEvents = disabled ? "none" : "";
    dropzone.style.opacity = disabled ? "0.6" : "";
    if (disabled) textInput.blur();
}

// Overlay Logic (Your original radial animation)
function showOverlayResult({label, confidence, details}) {
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

    radial.classList.remove("error");
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

document.getElementById("overlay").addEventListener("click", (e) => {
    if (e.target.id === "overlay" || e.target.classList.contains("overlay-close")) {
        document.getElementById("overlay").classList.remove("show");
    }
});

async function submitRequest({text = null, image = null}) {
    if (isSubmitting) return;

    let endpoint = "";
    let payload = {};

    // Prioritizing endpoint based on input type as per your original logic
    if (image) {
        endpoint = "/analyze_image";
        payload.image = await fileToBase64Cached(image);
        // If your API also accepts text with the image, add it here:
        if (text) payload.text = text.trim();
    } else if (text && text.trim()) {
        endpoint = "/analyze_text";
        payload.text = text.trim();
    } else {
        return;
    }

    isSubmitting = true;
    setInputsDisabled(true);

    try {
        const res = await fetch(GATEWAY_URL + endpoint, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error(`Server error (${res.status})`);
        const data = await res.json();
        showOverlayResult(data);
    } catch (err) {
        showOverlayResult({label: "Error", confidence: null, details: err.message});
    } finally {
        isSubmitting = false;
        setInputsDisabled(false);
    }
}

document.addEventListener("DOMContentLoaded", async () => {
    // 1. Check if background.js finished a request while popup was closed
    const {lastResult} = await browser.storage.local.get("lastResult");
    if (lastResult) {
        if (lastResult.error) {
            showError(lastResult.error);
        } else {
            showOverlayResult(lastResult);
        }
        // Clear it so it doesn't persist on next manual open
        await browser.storage.local.remove("lastResult");
    }

    // 2. Check for image sent via right-click (Context Menu)
    const {externalImageUrl} = await browser.storage.local.get("externalImageUrl");
    if (externalImageUrl) {
        try {
            const response = await fetch(externalImageUrl);
            const blob = await response.blob();
            const file = new File([blob], "input-image.jpg", {type: blob.type});

            // Reuses your existing image handling logic
            handleImageFile(file);
        } catch (err) {
            console.error("Failed to fetch context menu image:", err);
        }
        // Clear it so it doesn't reload on every open
        await browser.storage.local.remove("externalImageUrl");
    }
});