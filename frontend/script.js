const form = document.getElementById("upload-form");
const responseBox = document.getElementById("response");
const GATEWAY_URL = window.APP_CONFIG.GATEWAY_URL;

form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const formData = new FormData(form);

    responseBox.textContent = "Sending...";

    try {
        const res = await fetch(GATEWAY_URL, {
            method: "POST",
            body: formData
        });

        const data = await res.json();
        responseBox.textContent = JSON.stringify(data, null, 2);

    } catch (err) {
        responseBox.textContent = "Error: " + err.message;
    }
});