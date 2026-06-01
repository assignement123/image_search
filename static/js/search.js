// static/js/search.js
import { searchLeaf, debugUploadImage } from './api.js';
import { formatSpecies, openLightbox } from './ui.js';
import { openDebugModalFromData } from './debug.js';

let selectedFile = null;

export function initSearchTab() {
    initDropzone();
    initWeightSliders();

    const searchBtn = document.getElementById("search-btn");
    if (searchBtn) {
        searchBtn.addEventListener("click", runSearch);
    }
}

// ════════════════════════════════════════════════════════════════════
// DROPZONE & FILE HANDLING
// ════════════════════════════════════════════════════════════════════
function initDropzone() {
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("file-input");
    const selectBtn = document.getElementById("select-btn");
    const clearBtn = document.getElementById("clear-btn");
    const debugBtn = document.getElementById("input-debug-btn");
    const dzInner = document.getElementById("dropzone-inner");
    const previewWrap = document.getElementById("preview-wrapper");
    const previewImg = document.getElementById("preview-img");
    const searchBtn = document.getElementById("search-btn");

    if (!dropzone) return;

    selectBtn.addEventListener("click", () => fileInput.click());
    clearBtn.addEventListener("click", clearFile);
    if (debugBtn) debugBtn.addEventListener("click", runInputDebug);

    fileInput.addEventListener("change", () => {
        if (fileInput.files[0]) setFile(fileInput.files[0]);
    });

    dropzone.addEventListener("click", (e) => {
        if (e.target === dropzone || e.target === dzInner) fileInput.click();
    });

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("drag-over");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("drag-over");
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("drag-over");
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith("image/")) setFile(file);
    });

    function setFile(file) {
        selectedFile = file;
        const url = URL.createObjectURL(file);
        previewImg.src = url;
        dzInner.style.display = "none";
        previewWrap.style.display = "block";
        searchBtn.disabled = false;
        if (debugBtn) debugBtn.disabled = false;
    }

    function clearFile() {
        selectedFile = null;
        fileInput.value = "";
        previewImg.src = "";
        dzInner.style.display = "block";
        previewWrap.style.display = "none";
        searchBtn.disabled = true;
        if (debugBtn) debugBtn.disabled = true;
        showSearchEmpty();
    }
}

// ════════════════════════════════════════════════════════════════════
// WEIGHT SLIDERS
// ════════════════════════════════════════════════════════════════════
function initWeightSliders() {
    const sliders = [
        { id: "w-efd", valId: "v-efd" },
        { id: "w-morphology", valId: "v-morphology" },
        { id: "w-lbp", valId: "v-lbp" },
        { id: "w-glcm", valId: "v-glcm" },
        { id: "w-color", valId: "v-color" },
        { id: "w-vein", valId: "v-vein" },
    ];

    sliders.forEach(({ id, valId }) => {
        const slider = document.getElementById(id);
        const val = document.getElementById(valId);
        if (slider && val) {
            slider.addEventListener("input", () => {
                val.textContent = slider.value + "%";
            });
        }
    });
}

function getWeights() {
    return {
        w_efd: parseFloat(document.getElementById("w-efd").value) / 100,
        w_morphology: parseFloat(document.getElementById("w-morphology").value) / 100,
        w_lbp: parseFloat(document.getElementById("w-lbp").value) / 100,
        w_glcm: parseFloat(document.getElementById("w-glcm").value) / 100,
        w_color: parseFloat(document.getElementById("w-color").value) / 100,
        w_vein: parseFloat(document.getElementById("w-vein").value) / 100,
        top_k: document.getElementById("top-k").value,
    };
}

// ════════════════════════════════════════════════════════════════════
// SEARCH EXECUTION & UI UPDATES
// ════════════════════════════════════════════════════════════════════
async function runSearch() {
    if (!selectedFile) return;

    const btn = document.getElementById("search-btn");
    const txt = document.getElementById("search-btn-text");

    // Trạng thái Loading
    btn.disabled = true;
    txt.textContent = "⏳ Đang xử lý...";
    document.getElementById("search-empty").style.display = "none";
    document.getElementById("search-error").style.display = "none";
    document.getElementById("results-header").style.display = "none";
    document.getElementById("results-grid").innerHTML = "";
    document.getElementById("search-loading").style.display = "flex";

    const weights = getWeights();
    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("top_k", weights.top_k);
    formData.append("w_efd", weights.w_efd);
    formData.append("w_morphology", weights.w_morphology);
    formData.append("w_lbp", weights.w_lbp);
    formData.append("w_glcm", weights.w_glcm);
    formData.append("w_color", weights.w_color);
    formData.append("w_vein", weights.w_vein);

    try {
        // Gọi hàm fetch từ api.js
        const data = await searchLeaf(formData);

        document.getElementById("search-loading").style.display = "none";
        renderResults(data.results);
        document.getElementById("results-count").textContent = `${data.count} kết quả`;
        document.getElementById("results-header").style.display = "flex";

        if (data.debug) {
            openDebugModalFromData(data.debug);
        }

    } catch (err) {
        document.getElementById("search-loading").style.display = "none";
        showSearchError(err.message || "Không thể kết nối đến server. Đảm bảo Flask đang chạy.");
    } finally {
        // Reset nút
        btn.disabled = false;
        txt.textContent = "🔍 Tìm kiếm";
    }
}

async function runInputDebug() {
    if (!selectedFile) return;

    const debugBtn = document.getElementById("input-debug-btn");
    const previousText = debugBtn ? debugBtn.textContent : "";
    if (debugBtn) {
        debugBtn.disabled = true;
        debugBtn.textContent = "⏳ Đang debug...";
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
        const data = await debugUploadImage(formData);
        if (data) {
            openDebugModalFromData(data);
        }
    } catch (err) {
        showSearchError(err.message || "Không thể debug ảnh input.");
    } finally {
        if (debugBtn) {
            debugBtn.disabled = false;
            debugBtn.textContent = previousText || "🔬 Debug ảnh input";
        }
    }
}

function renderResults(results) {
    const grid = document.getElementById("results-grid");
    grid.innerHTML = "";

    if (!results || results.length === 0) {
        grid.innerHTML = '<p style="color:var(--text-muted);text-align:center;padding:40px">Không tìm thấy kết quả</p>';
        return;
    }

    results.forEach((r, i) => {
        const card = document.createElement("div");
        card.className = "result-card";

        // Fallback ảnh SVG nếu ảnh lỗi
        const fallbackSvg = `data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22160%22 height=%22160%22><rect width=%22160%22 height=%22160%22 fill=%22%231a2235%22/><text x=%2280%22 y=%2290%22 text-anchor=%22middle%22 fill=%22%23374151%22 font-size=%2240%22>🍃</text></svg>`;

        // Gắn icon huy chương cho top 3
        const rankHtml = i < 3
            ? `<div class="rank-badge rank-${i + 1}">${i === 0 ? "🥇" : i === 1 ? "🥈" : "🥉"}</div>`
            : `<div class="rank-badge">#${i + 1}</div>`;

        card.innerHTML = `
      ${rankHtml}
      <div class="result-img-wrap">
        <img class="result-img" src="${r.image_url}" alt="${r.filename}" loading="lazy"
             onerror="this.src='${fallbackSvg}'" />
      </div>
      <div class="result-info">
        <div class="result-filename">${r.filename}</div>
        <div class="result-species">${formatSpecies(r.species)}</div>
        <div class="result-score">${r.similarity}% tương đồng</div>
        <div class="score-bar" style="width:${Math.min(r.similarity, 100)}%"></div>
      </div>
    `;

        // Sử dụng hàm openLightbox import từ ui.js
        card.addEventListener("click", () =>
            openLightbox(
                r.image_url,
                r.filename,
                r.species,
                r.similarity + "% tương đồng"
            )
        );

        grid.appendChild(card);
    });
}

function showSearchEmpty() {
    document.getElementById("search-empty").style.display = "flex";
    document.getElementById("search-loading").style.display = "none";
    document.getElementById("search-error").style.display = "none";
    document.getElementById("results-header").style.display = "none";
    document.getElementById("results-grid").innerHTML = "";
}

function showSearchError(msg) {
    document.getElementById("search-error").style.display = "flex";
    document.getElementById("search-error-msg").textContent = msg;
}
