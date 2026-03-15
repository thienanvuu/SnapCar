const uploadInput = document.getElementById("car_image");
const uploadZone = document.getElementById("upload-zone");
const previewImage = document.getElementById("preview-image");
const selectedFile = document.getElementById("selected-file");
const uploadForm = document.getElementById("upload-form");
const submitButton = document.getElementById("submit-button");
const buttonLabel = document.querySelector(".button-label");
const buttonLoader = document.getElementById("button-loader");

if (uploadInput && uploadZone) {
    const updatePreview = (file) => {
        if (!file) {
            previewImage.classList.add("hidden");
            previewImage.removeAttribute("src");
            selectedFile.textContent = "No image selected";
            return;
        }

        selectedFile.textContent = file.name;
        previewImage.src = URL.createObjectURL(file);
        previewImage.classList.remove("hidden");
    };

    uploadInput.addEventListener("change", (event) => {
        const [file] = event.target.files;
        updatePreview(file);
    });

    ["dragenter", "dragover"].forEach((eventName) => {
        uploadZone.addEventListener(eventName, (event) => {
            event.preventDefault();
            uploadZone.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach((eventName) => {
        uploadZone.addEventListener(eventName, (event) => {
            event.preventDefault();
            uploadZone.classList.remove("dragover");
        });
    });

    uploadZone.addEventListener("drop", (event) => {
        const [file] = event.dataTransfer.files;

        if (file) {
            uploadInput.files = event.dataTransfer.files;
            updatePreview(file);
        }
    });
}

if (uploadForm && submitButton) {
    uploadForm.addEventListener("submit", () => {
        submitButton.disabled = true;
        buttonLabel.textContent = "Analyzing...";
        buttonLoader.classList.remove("hidden");
    });
}
