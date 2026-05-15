// static/js/ui.js
export function formatSpecies(name) {
    if (!name || name === "Unknown") return "Unknown";
    return name.replace(/_/g, " ");
}

export function openLightbox(imgUrl, filename, species, score) {
    document.getElementById("lightbox-img").src = imgUrl;
    document.getElementById("lightbox-filename").textContent = filename;
    document.getElementById("lightbox-species").textContent = formatSpecies(species);
    document.getElementById("lightbox-score").textContent = score;
    document.getElementById("lightbox").style.display = "flex";
    document.body.style.overflow = "hidden";

    document.getElementById("debug-btn").dataset.filename = filename;
}

export function closeLightbox() {
    document.getElementById("lightbox").style.display = "none";
    document.body.style.overflow = "";
}

export function initLightbox() {
    document.getElementById("lightbox").addEventListener('click', closeLightbox);
    document.querySelector('.lightbox-content').addEventListener('click', e => e.stopPropagation());
    document.querySelector('.lightbox-close').addEventListener('click', closeLightbox);

    document.getElementById("debug-btn").addEventListener("click", function () {
        const filename = this.dataset.filename;
        if (filename) window.location.href = `/debug-view/${filename}`;
    });

    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") closeLightbox();
    });
}