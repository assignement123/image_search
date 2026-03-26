/* ═══════════════════════════════════════════════════════════════════
   LeafSearch — app.js
   ═══════════════════════════════════════════════════════════════════ */

"use strict";

// ── State ─────────────────────────────────────────────────────────
let selectedFile = null;
let speciesChart = null;
let buildPoller = null;
let currentSpeciesPage = 1;
let currentSpeciesName = null;

// ── Init ──────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initDropzone();
  initWeightSliders();
  initSearchBtn();
  loadDbBadge();
  loadSpeciesList();
  loadStats();
  initBuildButtons();
});

// ════════════════════════════════════════════════════════════════════
// TABS
// ════════════════════════════════════════════════════════════════════
function initTabs() {
  const btns = document.querySelectorAll(".nav-btn");
  btns.forEach((btn) => {
    btn.addEventListener("click", () => {
      btns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      document
        .querySelectorAll(".tab-content")
        .forEach((t) => t.classList.remove("active"));
      document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");

      if (btn.dataset.tab === "stats") loadStats();
      if (
        btn.dataset.tab === "browse" &&
        !document.getElementById("species-list").querySelector(".species-item")
      )
        loadSpeciesList();
    });
  });
}

// ════════════════════════════════════════════════════════════════════
// DROPZONE & FILE HANDLING
// ════════════════════════════════════════════════════════════════════
function initDropzone() {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const selectBtn = document.getElementById("select-btn");
  const clearBtn = document.getElementById("clear-btn");
  const dzInner = document.getElementById("dropzone-inner");
  const previewWrap = document.getElementById("preview-wrapper");
  const previewImg = document.getElementById("preview-img");
  const searchBtn = document.getElementById("search-btn");

  selectBtn.addEventListener("click", () => fileInput.click());
  clearBtn.addEventListener("click", clearFile);

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
  dropzone.addEventListener("dragleave", () =>
    dropzone.classList.remove("drag-over"),
  );
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
  }

  function clearFile() {
    selectedFile = null;
    fileInput.value = "";
    previewImg.src = "";
    dzInner.style.display = "block";
    previewWrap.style.display = "none";
    searchBtn.disabled = true;
    showSearchEmpty();
  }
}

function showSearchEmpty() {
  document.getElementById("search-empty").style.display = "flex";
  document.getElementById("search-loading").style.display = "none";
  document.getElementById("search-error").style.display = "none";
  document.getElementById("results-header").style.display = "none";
  document.getElementById("results-grid").innerHTML = "";
}

// ════════════════════════════════════════════════════════════════════
// WEIGHT SLIDERS
// ════════════════════════════════════════════════════════════════════
function initWeightSliders() {
  const sliders = [
    { id: "w-efd", valId: "v-efd" },
    { id: "w-texture", valId: "v-texture" },
    { id: "w-color", valId: "v-color" },
    { id: "w-vein", valId: "v-vein" },
  ];
  sliders.forEach(({ id, valId }) => {
    const slider = document.getElementById(id);
    const val = document.getElementById(valId);
    slider.addEventListener("input", () => {
      val.textContent = slider.value + "%";
    });
  });
}

function getWeights() {
  return {
    w_efd: parseFloat(document.getElementById("w-efd").value) / 100,
    w_texture: parseFloat(document.getElementById("w-texture").value) / 100,
    w_color: parseFloat(document.getElementById("w-color").value) / 100,
    w_vein: parseFloat(document.getElementById("w-vein").value) / 100,
    top_k: document.getElementById("top-k").value,
  };
}

// ════════════════════════════════════════════════════════════════════
// SEARCH
// ════════════════════════════════════════════════════════════════════
function initSearchBtn() {
  document.getElementById("search-btn").addEventListener("click", runSearch);
}

async function runSearch() {
  if (!selectedFile) return;

  const btn = document.getElementById("search-btn");
  const txt = document.getElementById("search-btn-text");
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
  formData.append("w_texture", weights.w_texture);
  formData.append("w_color", weights.w_color);
  formData.append("w_vein", weights.w_vein);

  try {
    const res = await fetch("/api/search", { method: "POST", body: formData });
    const data = await res.json();

    document.getElementById("search-loading").style.display = "none";

    if (!res.ok) {
      showSearchError(data.error || "Lỗi không xác định");
      return;
    }

    renderResults(data.results);
    document.getElementById("results-count").textContent =
      `${data.count} kết quả`;
    document.getElementById("results-header").style.display = "flex";
  } catch (err) {
    document.getElementById("search-loading").style.display = "none";
    showSearchError("Không thể kết nối đến server. Đảm bảo Flask đang chạy.");
  } finally {
    btn.disabled = false;
    txt.textContent = "🔍 Tìm kiếm";
  }
}

function showSearchError(msg) {
  document.getElementById("search-error").style.display = "flex";
  document.getElementById("search-error-msg").textContent = msg;
}

function renderResults(results) {
  const grid = document.getElementById("results-grid");
  grid.innerHTML = "";

  if (!results || results.length === 0) {
    grid.innerHTML =
      '<p style="color:var(--text-muted);text-align:center;padding:40px">Không tìm thấy kết quả</p>';
    return;
  }

  results.forEach((r, i) => {
    const card = document.createElement("div");
    card.className = "result-card";
    card.innerHTML = `
      <div class="rank-badge ${i < 3 ? `rank-${i + 1}` : ""}">
        ${i === 0 ? "🥇" : i === 1 ? "🥈" : i === 2 ? "🥉" : `#${i + 1}`}
      </div>
      <div class="result-img-wrap">
        <img class="result-img" src="${r.image_url}" alt="${r.filename}" loading="lazy"
             onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22160%22 height=%22160%22><rect width=%22160%22 height=%22160%22 fill=%22%231a2235%22/><text x=%2280%22 y=%2290%22 text-anchor=%22middle%22 fill=%22%23374151%22 font-size=%2240%22>🍃</text></svg>'" />
      </div>
      <div class="result-info">
        <div class="result-filename">${r.filename}</div>
        <div class="result-species">${formatSpecies(r.species)}</div>
        <div class="result-score">${r.similarity}% tương đồng</div>
        <div class="score-bar" style="width:${Math.min(r.similarity, 100)}%"></div>
      </div>
    `;
    card.addEventListener("click", () =>
      openLightbox(
        r.image_url,
        r.filename,
        r.species,
        r.similarity + "% tương đồng",
      ),
    );
    grid.appendChild(card);
  });
}

// ════════════════════════════════════════════════════════════════════
// BROWSE SPECIES
// ════════════════════════════════════════════════════════════════════
async function loadSpeciesList() {
  const container = document.getElementById("species-list");
  try {
    const res = await fetch("/api/species");
    const data = await res.json();

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
      document
        .querySelectorAll(".species-item")
        .forEach((el) => el.classList.remove("active"));
      item.classList.add("active");
      loadSpeciesImages(s.species, 1);
    });
    container.appendChild(item);
  });

  // Search filter
  document
    .getElementById("species-search")
    .addEventListener("input", function () {
      const q = this.value.toLowerCase();
      document.querySelectorAll(".species-item").forEach((el) => {
        el.style.display = el.dataset.species.toLowerCase().includes(q)
          ? ""
          : "none";
      });
    });
}

async function loadSpeciesImages(speciesName, page) {
  currentSpeciesName = speciesName;
  currentSpeciesPage = page;

  document.getElementById("browse-empty").style.display = "none";
  const detail = document.getElementById("species-detail");
  detail.style.display = "block";
  document.getElementById("species-detail-name").textContent =
    formatSpecies(speciesName);
  document.getElementById("species-grid").innerHTML =
    '<div class="loading"><div class="spinner"></div></div>';
  document.getElementById("species-pagination").innerHTML = "";

  const res = await fetch(
    `/api/species/${encodeURIComponent(speciesName)}?page=${page}&per_page=20`,
  );
  const data = await res.json();

  document.getElementById("species-detail-count").textContent =
    `${data.total} ảnh`;

  const grid = document.getElementById("species-grid");
  grid.innerHTML = "";
  data.images.forEach((img) => {
    const card = document.createElement("div");
    card.className = "result-card";
    card.innerHTML = `
      <div class="result-img-wrap">
        <img class="result-img" src="${img.image_url}" alt="${img.filename}" loading="lazy"
             onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22160%22 height=%22160%22><rect width=%22160%22 height=%22160%22 fill=%22%231a2235%22/><text x=%2280%22 y=%2290%22 text-anchor=%22middle%22 fill=%22%23374151%22 font-size=%2240%22>🍃</text></svg>'" />
      </div>
      <div class="result-info">
        <div class="result-filename">${img.filename}</div>
        <div class="result-species">${formatSpecies(speciesName)}</div>
      </div>
    `;
    card.addEventListener("click", () =>
      openLightbox(img.image_url, img.filename, speciesName, ""),
    );
    grid.appendChild(card);
  });

  // Pagination
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

// ════════════════════════════════════════════════════════════════════
// STATS
// ════════════════════════════════════════════════════════════════════
async function loadStats() {
  try {
    const res = await fetch("/api/stats");
    const data = await res.json();

    if (data.error) return;

    // KPI
    animateCount("kpi-total", 0, data.total_images, 800);
    animateCount("kpi-species", 0, data.total_species, 600);

    // Vector dims from meta
    const efd = parseInt(data.extract_params?.dim_efd || 76);
    const tex = parseInt(data.extract_params?.dim_texture || 46);
    const col = parseInt(data.extract_params?.dim_color || 9);
    const vin = parseInt(data.extract_params?.dim_vein || 9);
    document.getElementById("kpi-dim").textContent = efd + tex + col + vin;

    // Chart
    renderSpeciesChart(data.species_distribution);

    // Params
    renderParams(data.extract_params);
  } catch (e) {
    console.error("Stats load error", e);
  }
}

function animateCount(id, from, to, duration) {
  const el = document.getElementById(id);
  const start = performance.now();
  function frame(now) {
    const t = Math.min((now - start) / duration, 1);
    const ease = 1 - Math.pow(1 - t, 3);
    el.textContent = Math.round(from + (to - from) * ease).toLocaleString();
    if (t < 1) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

function renderSpeciesChart(dist) {
  if (!dist || dist.length === 0) return;

  const ctx = document.getElementById("species-chart").getContext("2d");

  // Take top 20 to keep chart readable
  const top = dist.slice(0, 20);
  const labels = top.map((d) => formatSpecies(d.species));
  const values = top.map((d) => d.count);

  const colors = labels.map(
    (_, i) => `hsla(${(i * 137.5) % 360}, 65%, 55%, 0.8)`,
  );

  if (speciesChart) speciesChart.destroy();

  speciesChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Số ảnh",
          data: values,
          backgroundColor: colors,
          borderColor: colors.map((c) => c.replace("0.8", "1")),
          borderWidth: 1,
          borderRadius: 5,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: (items) => items[0].label,
            label: (item) => ` ${item.raw} ảnh`,
          },
        },
      },
      scales: {
        x: {
          ticks: {
            color: "#7a8499",
            font: { size: 11 },
            maxRotation: 45,
          },
          grid: { color: "rgba(255,255,255,0.04)" },
        },
        y: {
          ticks: { color: "#7a8499", font: { size: 11 } },
          grid: { color: "rgba(255,255,255,0.06)" },
          beginAtZero: true,
        },
      },
    },
  });
}

function renderParams(params) {
  const grid = document.getElementById("params-grid");
  if (!params || Object.keys(params).length === 0) {
    grid.innerHTML =
      '<p style="color:var(--text-muted);font-size:13px">Chưa có dữ liệu</p>';
    return;
  }

  const labelMap = {
    harmonics: "Harmonics (EFD)",
    n_resample: "N Resample",
    glcm_levels: "GLCM Levels",
    dim_efd: "Chiều EFD",
    dim_texture: "Chiều Texture",
    dim_color: "Chiều Color",
    dim_vein: "Chiều Vein",
    background: "Nền ảnh",
    created_by: "Script tạo",
  };

  grid.innerHTML = "";
  Object.entries(params).forEach(([k, v]) => {
    const item = document.createElement("div");
    item.className = "param-item";
    item.innerHTML = `<div class="param-key">${labelMap[k] || k}</div><div class="param-val">${v}</div>`;
    grid.appendChild(item);
  });
}

// ════════════════════════════════════════════════════════════════════
// DB BADGE
// ════════════════════════════════════════════════════════════════════
async function loadDbBadge() {
  try {
    const res = await fetch("/api/stats");
    const data = await res.json();
    if (data.total_images !== undefined) {
      document.getElementById("db-count").textContent =
        `${data.total_images.toLocaleString()} ảnh · ${data.total_species} loài`;
    }
  } catch (e) {
    document.getElementById("db-count").textContent = "Không kết nối DB";
    document.querySelector(".db-dot").style.background = "var(--danger)";
  }
}

// ════════════════════════════════════════════════════════════════════
// BUILD DATABASE
// ════════════════════════════════════════════════════════════════════
function initBuildButtons() {
  document
    .getElementById("btn-build")
    .addEventListener("click", () => triggerBuild(false));
  document.getElementById("btn-rebuild").addEventListener("click", () => {
    if (
      confirm(
        "⚠️ Bạn có chắc muốn xóa toàn bộ dữ liệu và build lại?\nQuá trình này không thể hoàn tác.",
      )
    ) {
      triggerBuild(true);
    }
  });
}

async function triggerBuild(rebuild) {
  const res = await fetch("/api/build", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ rebuild }),
  });
  const data = await res.json();

  if (!res.ok) {
    alert(data.error || "Lỗi");
    return;
  }

  document.getElementById("progress-section").style.display = "block";
  pollBuildStatus();
}

function pollBuildStatus() {
  if (buildPoller) clearInterval(buildPoller);
  buildPoller = setInterval(async () => {
    const res = await fetch("/api/build/status");
    const data = await res.json();
    updateProgressUI(data);
    if (!data.running) {
      clearInterval(buildPoller);
      loadDbBadge();
      loadStats();
    }
  }, 1000);
}

function updateProgressUI(s) {
  const pct = s.total > 0 ? Math.round((s.progress / s.total) * 100) : 0;
  document.getElementById("progress-bar").style.width = pct + "%";
  document.getElementById("progress-text").textContent =
    `${s.progress} / ${s.total}`;
  document.getElementById("progress-pct").textContent = pct + "%";
  document.getElementById("p-added").textContent = s.added;
  document.getElementById("p-skipped").textContent = s.skipped;
  document.getElementById("p-errors").textContent = s.errors;
  document.getElementById("progress-msg").textContent = s.message;
}

// ════════════════════════════════════════════════════════════════════
// LIGHTBOX
// ════════════════════════════════════════════════════════════════════
let currentFilename = null;
function openLightbox(imgUrl, filename, species, score) {
  document.getElementById("lightbox-img").src = imgUrl;
  document.getElementById("lightbox-filename").textContent = filename;
  document.getElementById("lightbox-species").textContent =
    formatSpecies(species);
  document.getElementById("lightbox-score").textContent = score;
  document.getElementById("lightbox").style.display = "flex";
  document.body.style.overflow = "hidden";
  currentFilename = filename;
  document.getElementById("lightbox").style.display = "flex";
  
}
// Debug button click
document.getElementById("debug-btn").addEventListener("click", () => {
  if (!currentFilename) {
    alert("Không có ảnh để debug!");
    return;
  }

  // chuyển sang trang debug
  window.location.href = `/debug-view/${currentFilename}`;
});
function closeLightbox() {
  document.getElementById("lightbox").style.display = "none";
  document.body.style.overflow = "";
}

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeLightbox();
});

// ── Utility ──────────────────────────────────────────────────────
function formatSpecies(name) {
  if (!name || name === "Unknown") return "Unknown";
  return name.replace(/_/g, " ");
}
