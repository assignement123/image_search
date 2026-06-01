// static/js/cascaded.js

export function initCascadedTab() {
  // Guard: nếu tab chưa có trong DOM thì thoát an toàn
  const selBtn = document.getElementById("cs-select-btn");
  if (!selBtn) return;

  const dz = document.getElementById("cs-dropzone");
  const dzInner = document.getElementById("cs-dropzone-inner");
  const prevWrap = document.getElementById("cs-preview-wrapper");
  const prevImg = document.getElementById("cs-preview-img");
  const fileInp = document.getElementById("cs-file-input");
  const clearBtn = document.getElementById("cs-clear-btn");
  const searchBtn = document.getElementById("cs-search-btn");

  let csFile = null;

  // ── upload / preview ───────────────────────────────────────────
  selBtn.addEventListener("click", () => fileInp.click());

  fileInp.addEventListener("change", (e) => setFile(e.target.files[0]));

  clearBtn.addEventListener("click", () => {
    csFile = null;
    prevWrap.style.display = "none";
    dzInner.style.display = "";
    searchBtn.disabled = true;
    fileInp.value = "";
  });

  dz.addEventListener("dragover", (e) => {
    e.preventDefault();
    dz.classList.add("drag-over");
  });
  dz.addEventListener("dragleave", () => dz.classList.remove("drag-over"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("drag-over");
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith("image/")) setFile(f);
  });

  function setFile(f) {
    if (!f) return;
    csFile = f;
    prevImg.src = URL.createObjectURL(f);
    dzInner.style.display = "none";
    prevWrap.style.display = "";
    searchBtn.disabled = false;
  }

  // ── search ─────────────────────────────────────────────────────
  searchBtn.addEventListener("click", async () => {
    if (!csFile) return;
    showState("loading");

    const fd = new FormData();
    fd.append("file", csFile);
    fd.append("top1", document.getElementById("cs-top1").value);
    fd.append("top2", document.getElementById("cs-top2").value);
    fd.append("top3", document.getElementById("cs-top3").value);
    fd.append("top4", document.getElementById("cs-top4").value);
    fd.append("top5", document.getElementById("cs-top5").value);
    fd.append("top6", document.getElementById("cs-top6").value);

    try {
      const res = await fetch("/api/search/cascaded", {
        method: "POST",
        body: fd,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Lỗi không xác định");
      renderFunnel(data.funnel);
      renderResults(data.results);
    } catch (err) {
      showState("error");
      document.getElementById("cs-error-msg").textContent = err.message;
    }
  });

  // ── funnel bar chart ────────────────────────────────────────────
  const STAGE_COLORS = [
    "#4a90d9",
    "#818cf8",
    "#a78bfa",
    "#22d3ee",
    "#fb923c",
    "#f472b6",
  ];
  const STAGE_LABELS = ["EFD", "Morph", "LBP", "GLCM", "Vein", "Color"];

  function renderFunnel(f) {
    const counts = [f.top1, f.top2, f.top3, f.top4, f.top5, f.top6];
    const max = counts[0];
    const barsEl = document.getElementById("cs-funnel-bars");
    const lblEl = document.getElementById("cs-funnel-labels");

    barsEl.innerHTML = counts
      .map((n, i) => {
        const h = Math.max(10, Math.round((n / max) * 60));
        return `<div style="flex:1;display:flex;flex-direction:column;align-items:center;gap:3px">
          <span style="font-size:10px;color:var(--text-muted);font-weight:500">${n}</span>
          <div style="width:100%;height:${h}px;border-radius:4px 4px 0 0;
                      background:${STAGE_COLORS[i]}26;
                      border:1px solid ${STAGE_COLORS[i]}80"></div>
        </div>`;
      })
      .join("");

    lblEl.innerHTML = STAGE_LABELS.map(
      (l, i) =>
        `<div style="flex:1;text-align:center;display:flex;align-items:center;
                     justify-content:center;gap:4px">
          <span style="width:6px;height:6px;border-radius:50%;
                       background:${STAGE_COLORS[i]};display:inline-block;
                       flex-shrink:0"></span>${l}
        </div>`,
    ).join("");

    document.getElementById("cs-funnel-wrap").style.display = "";
  }

  // ── results ─────────────────────────────────────────────────────
  const DEBUG_LABELS = [
    { key: "d_efd", label: "EFD", color: "#4a90d9" },
    { key: "d_morph", label: "Morph", color: "#818cf8" },
    { key: "d_lbp", label: "LBP", color: "#a78bfa" },
    { key: "d_glcm", label: "GLCM", color: "#22d3ee" },
    { key: "d_vein", label: "Vein", color: "#fb923c" },
    { key: "d_color", label: "Color", color: "#f472b6" },
  ];

  function renderResults(results) {
    showState("results");
    document.getElementById("cs-results-count").textContent =
      `${results.length} ảnh`;

    const grid = document.getElementById("cs-results-grid");
    if (!results.length) {
      grid.innerHTML = "";
      return;
    }

    grid.innerHTML = results
      .map((r, idx) => {
        const debugRows = DEBUG_LABELS.map(({ key, label, color }) => {
          const val = r.debug?.[key];
          return val != null
            ? `<div style="display:flex;justify-content:space-between;align-items:center;
                             font-size:10px;color:var(--text-muted)">
                   <span style="display:flex;align-items:center;gap:4px">
                     <span style="width:5px;height:5px;border-radius:50%;
                                  background:${color};display:inline-block"></span>
                     ${label}
                   </span>
                   <span style="font-variant-numeric:tabular-nums">${val}</span>
                 </div>`
            : "";
        }).join("");

        return `
        <div class="result-card"
             onclick="openLightbox('${r.image_url}','${r.filename}','${r.species || ""}','')">
          <div class="result-img-wrap">
            <img src="${r.image_url}" alt="${r.species}" class="result-img"
                 onerror="this.style.display='none';
                          this.nextElementSibling.style.display='flex'"/>
            <div class="result-img-fallback" style="display:none">🍃</div>
            <div class="result-rank">#${idx + 1}</div>
          </div>
          <div class="result-meta">
            <p class="result-species">${r.species || "—"}</p>
            <p class="result-filename">${r.filename}</p>
            <div style="margin-top:6px;display:flex;flex-direction:column;gap:2px">
              ${debugRows}
            </div>
          </div>
        </div>`;
      })
      .join("");
  }

  // ── state helper ────────────────────────────────────────────────
  function showState(state) {
    document.getElementById("cs-empty").style.display =
      state === "empty" ? "" : "none";
    document.getElementById("cs-loading").style.display =
      state === "loading" ? "" : "none";
    document.getElementById("cs-error").style.display =
      state === "error" ? "" : "none";
    document.getElementById("cs-results-header").style.display =
      state === "results" ? "" : "none";
    if (state !== "results") {
      document.getElementById("cs-funnel-wrap").style.display = "none";
    }
  }
}
