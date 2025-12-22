const ALLOWED_IMAGE_TYPES = [
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif"
];
const URL_REGEX = /^https?:\/\/[^\s/$.?#].[^\s]*$/i;

const linkInput = document.querySelector(".input.wide");
const textInput = document.querySelector(".card .input");
const submitBtn = document.querySelector(".submit-btn");

textInput.addEventListener("input", () => {
    submitBtn.classList.toggle(
        "visible",
        textInput.value.trim().length > 0
    );
});

submitBtn.addEventListener("click", () => {
    submitRequest({
        text: textInput.value,
        image: droppedFile || null
    });
});

function isValidHttpUrl(value) {
    return URL_REGEX.test(value.trim());
}

linkInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
        e.preventDefault();
        const value = linkInput.value.trim();
        if (!isValidHttpUrl(value)) {
            linkInput.classList.add("error");
            return;
        }
        linkInput.classList.remove("error");
        submitRequest({ link: linkInput.value });
    }
});

const dropzone = document.querySelector(".dropzone");
let droppedFile = null;

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

    if (!ALLOWED_IMAGE_TYPES.includes(file.type)) {
        alert("Only image files are allowed");
        return;
    }

    droppedFile = file;
    dropzone.querySelector("p").textContent = file.name;
});

textInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
        e.preventDefault();
        submitRequest({
            text: textInput.value,
            image: droppedFile || null
        });
    }
});

function submitRequest({ link = null, text = null, image = null }) {
    const formData = new FormData();

    if (link) formData.append("link", link);
    if (text) formData.append("text", text);
    if (image) formData.append("image", image);

    fetch(GATEWAY_URL, {
        method: "POST",
        body: formData
    })
        .then(res => res.json())
        .then(data => {
            console.log("Response:", data);
        })
        .catch(err => {
            console.error("Error:", err);
        });
}

document.addEventListener('DOMContentLoaded', () => {
    const body = document.body;
    const toggle = document.getElementById('themeToggle');

    if (!toggle) return;

    const theme =
        localStorage.getItem('theme') ??
        (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');

    body.dataset.theme = theme;
    toggle.textContent = theme === 'dark' ? '☀️' : '🌙';

    // Toggle
    toggle.addEventListener('click', () => {
        const next = body.dataset.theme === 'dark' ? 'light' : 'dark';
        body.dataset.theme = next;
        localStorage.setItem('theme', next);
        toggle.textContent = next === 'dark' ? '☀️' : '🌙';
    });
    // File upload lol revise later
    const form = document.getElementById('upload-form');
    const responseBox = document.getElementById('response');
    const GATEWAY_URL = window.APP_CONFIG?.GATEWAY_URL;

    if (!form || !GATEWAY_URL) {
        console.error('Form or gateway URL missing');
        return;
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const formData = new FormData(form);
        responseBox.textContent = 'Sending…';

        try {
            const res = await fetch(GATEWAY_URL, {
                method: 'POST',
                body: formData
            });

            if (!res.ok) {
                throw new Error(`HTTP ${res.status}`);
            }

            const data = await res.json();
            responseBox.textContent = JSON.stringify(data, null, 2);

        } catch (err) {
            responseBox.textContent = 'Error: ' + err.message;
        }
    });
});
