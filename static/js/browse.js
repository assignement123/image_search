import { getSpeciesList, getSpeciesImages } from './api.js';
import { formatSpecies, openLightbox } from './ui.js';

let currentSpeciesName = null;
let currentSpeciesPage = 1;

export async function loadSpeciesList() {
    const container = document.getElementById("species-list");
    try {
        const data = await getSpeciesList();
        if (data.error) {
            container.innerHTML = `<p style="color:var(--danger);padding:12px">${data.error}</p>`;
            return;
        }
        renderSpeciesList(data.species);
    } catch (e) {
        container.innerHTML = `<p style="color:var(--text-muted);padding:12px;font-size:13px">Không kết nối được server</p>`;
    }
}

function renderSpeciesList(speciesArr) {
    const container = document.getElementById("species-list");
    if (!speciesArr || speciesArr.length === 0) {
        container.innerHTML = `<p style="color:var(--text-muted);padding:12px;font-size:13px">Chưa có dữ liệu trong DB</p>`;
        return;
    }

    container.innerHTML = "";
    speciesArr.forEach((s) => {
        const item = document.createElement("div");
        item.className = "species-item";
        item.dataset.species = s.species;
        item.innerHTML = `
            <span class="species-item-name">${formatSpecies(s.species)}</span>
            <span class="species-item-count">${s.count}</span>
        `;
        item.addEventListener("click", () => {
            document.querySelectorAll(".species-item").forEach((el) => el.classList.remove("active"));
            item.classList.add("active");
            loadSpeciesImages(s.species, 1);
        });
        container.appendChild(item);
    });

    const searchInput = document.getElementById("species-search");
    if (searchInput) {
        searchInput.addEventListener("input", function () {
            const q = this.value.toLowerCase();
            document.querySelectorAll(".species-item").forEach((el) => {
                el.style.display = el.dataset.species.toLowerCase().includes(q) ? "" : "none";
            });
        });
    }
}

async function loadSpeciesImages(speciesName, page) {
    currentSpeciesName = speciesName;
    currentSpeciesPage = page;

    document.getElementById("browse-empty").style.display = "none";
    document.getElementById("species-detail").style.display = "block";
    document.getElementById("species-detail-name").textContent = formatSpecies(speciesName);
    document.getElementById("species-grid").innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    document.getElementById("species-pagination").innerHTML = "";

    const data = await getSpeciesImages(speciesName, page);

    document.getElementById("species-detail-count").textContent = `${data.total} ảnh`;
    
    const grid = document.getElementById("species-grid");
    grid.innerHTML = "";
    
    data.images.forEach((img) => {
        const card = document.createElement("div");
        card.className = "result-card";
        card.innerHTML = `
            <div class="result-img-wrap">
                <img class="result-img" src="${img.image_url}" alt="${img.filename}" loading="lazy" />
            </div>
            <div class="result-info">
                <div class="result-filename">${img.filename}</div>
                <div class="result-species">${formatSpecies(speciesName)}</div>
            </div>
        `;
        card.addEventListener("click", () => openLightbox(img.image_url, img.filename, speciesName, ""));
        grid.appendChild(card);
    });

    const totalPages = Math.ceil(data.total / 20);
    const pag = document.getElementById("species-pagination");
    if (totalPages > 1) {
        for (let p = 1; p <= totalPages; p++) {
            const btn = document.createElement("button");
            btn.className = `page-btn ${p === page ? "active" : ""}`;
            btn.textContent = p;
            btn.addEventListener("click", () => loadSpeciesImages(speciesName, p));
            pag.appendChild(btn);
        }
    }
}