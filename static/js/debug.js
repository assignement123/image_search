// static/js/debug.js
// ──────────────────────────────────────────────────────────────────────
// Debug Modal — hiển thị kết quả debug theo các nhóm
//               (preprocess/shape/lbp/glcm/color/vein) ngay trong app,
//               không redirect trang.
// ──────────────────────────────────────────────────────────────────────

let _currentFilename = null;
let _currentDebugMode = "file";

/** Mở modal debug và gọi API */
export async function openDebugModal(filename) {
  _currentFilename = filename;
  const modal = document.getElementById("debug-modal");
  const stem = filename.replace(/\.[^.]+$/, "");

  // Reset UI
  modal.style.display = "flex";
  document.body.style.overflow = "hidden";
  _setDebugState("loading");
  document.getElementById("debug-modal-title").textContent =
    `🔬 Debug: ${filename}`;

  try {
    const res = await fetch(`/api/debug/${encodeURIComponent(filename)}`);
    const data = await res.json();

    if (!res.ok || data.error) {
      _setDebugState(
        "error",
        data.stderr || data.error || "Lỗi không xác định",
      );
      return;
    }

    _renderGroups(data.groups, stem, data.cached);
  } catch (e) {
    _setDebugState("error", e.message || "Không thể kết nối server");
  }
}

/** Mở modal debug từ payload có sẵn trong response search */
export function openDebugModalFromData(debugData) {
    _currentDebugMode = "inline";
    _currentFilename = debugData.filename || null;

    const modal = document.getElementById("debug-modal");
    modal.style.display = "flex";
    document.body.style.overflow = "hidden";

    document.getElementById("debug-rerun-btn").style.display = "none";

    const title = debugData.filename || "Ảnh input";
    document.getElementById("debug-modal-title").textContent = `🔬 Debug: ${title}`;

    if (debugData.error) {
        _setDebugState("error", debugData.error);
        return;
    }

    _renderGroups(debugData.groups, debugData.stem || title.replace(/\.[^.]+$/, ""), debugData.cached, debugData.morphology);
}

/** Đóng modal */
export function closeDebugModal() {
  document.getElementById("debug-modal").style.display = "none";
  document.body.style.overflow = "";
}

/** Xoá cache và chạy lại */
async function rerunDebug() {
  if (!_currentFilename) return;
  const stem = _currentFilename.replace(/\.[^.]+$/, "");
  await fetch(`/api/debug-clear/${encodeURIComponent(stem)}`, {
    method: "DELETE",
  });
  await openDebugModal(_currentFilename);
}

// ── Internal helpers ──────────────────────────────────────────────────

function _setDebugState(state, msg = "") {
  document.getElementById("debug-loading").style.display =
    state === "loading" ? "flex" : "none";
  document.getElementById("debug-error").style.display =
    state === "error" ? "flex" : "none";
  document.getElementById("debug-content").style.display =
    state === "done" ? "block" : "none";

  if (state === "error") {
    document.getElementById("debug-error-msg").textContent = msg;
  }
}

function _renderGroups(groups, stem, cached) {
  _setDebugState("done");
  const container = document.getElementById("debug-content");
  container.innerHTML = "";

  if (!groups || groups.length === 0) {
    container.innerHTML =
      '<p style="color:var(--text-muted);padding:40px;text-align:center">Không có kết quả debug</p>';
    return;
  }

  // Badge cached/fresh
  const badge = cached
    ? `<span style="background:rgba(245,158,11,0.15);color:#fbbf24;border:1px solid rgba(245,158,11,0.3);padding:3px 10px;border-radius:999px;font-size:11px;font-weight:600">⚡ Cached</span>`
    : `<span style="background:rgba(34,197,94,0.15);color:#4ade80;border:1px solid rgba(34,197,94,0.3);padding:3px 10px;border-radius:999px;font-size:11px;font-weight:600">✓ Mới tạo</span>`;

  // Tab navigation
  const tabNav = document.createElement("div");
  tabNav.className = "debug-tab-nav";
  tabNav.innerHTML =
    groups
      .map(
        (g, i) =>
          `<button class="debug-tab-btn${i === 0 ? " active" : ""}" data-group="${g.id}">${g.title} <span class="debug-tab-count">${g.images.length}</span></button>`,
      )
      .join("") + `<div class="debug-badge-wrap">${badge}</div>`;
  container.appendChild(tabNav);

  // Group panels
  const panels = document.createElement("div");
  panels.className = "debug-panels";

  groups.forEach((group, gi) => {
    const panel = document.createElement("div");
    panel.className = `debug-panel${gi === 0 ? " active" : ""}`;
    panel.dataset.group = group.id;

    const grid = document.createElement("div");
    grid.className = "debug-img-grid";

    group.images.forEach((img) => {
      const card = document.createElement("div");
      card.className = "debug-img-card";
      card.innerHTML = `
                <div class="debug-img-wrap">
                    <img src="${img.url}" alt="${img.name}" loading="lazy"
                         style="cursor:zoom-in" />
                </div>
                <div class="debug-img-label">${img.name}</div>
            `;
      grid.appendChild(card);
    });

    panel.appendChild(grid);
    panels.appendChild(panel);
  });

  container.appendChild(panels);

  // Wire tab buttons
  tabNav.querySelectorAll(".debug-tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      tabNav
        .querySelectorAll(".debug-tab-btn")
        .forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      panels
        .querySelectorAll(".debug-panel")
        .forEach((p) => p.classList.remove("active"));
      panels
        .querySelector(`.debug-panel[data-group="${btn.dataset.group}"]`)
        .classList.add("active");
    });
  });
}

function _renderMorphologySummary(morphology) {
    if (!morphology || !Array.isArray(morphology.vector)) return null;

    const wrap = document.createElement("div");
    wrap.className = "debug-summary-card";
    wrap.style.cssText = [
        "margin:0 0 16px",
        "padding:16px 18px",
        "border:1px solid rgba(59,130,246,0.25)",
        "border-radius:16px",
        "background:linear-gradient(135deg, rgba(16,26,46,0.96), rgba(26,42,70,0.92))",
        "box-shadow:0 10px 30px rgba(0,0,0,0.18)",
    ].join(";");

    const rows = morphology.vector.map((value, index) => {
        const label = morphology.display_labels?.[index] || morphology.labels?.[index] || `Value ${index + 1}`;
        const meaning = morphology.meanings?.[index] || "";
        return `
            <tr>
                <td style="padding:10px 12px;color:#a8d1ff;font-weight:600">${label}</td>
                <td style="padding:10px 12px;color:#e6f1ff;font-family:monospace">${_formatNumber(value)}</td>
                <td style="padding:10px 12px;color:#8a9dc0">${meaning}</td>
            </tr>
        `;
    }).join("");

    wrap.innerHTML = `
        <div style="display:flex;justify-content:space-between;gap:12px;align-items:flex-start;margin-bottom:12px;flex-wrap:wrap">
            <div>
                <div style="font-size:12px;text-transform:uppercase;letter-spacing:0.08em;color:#7ecfff;font-weight:700">Morphology Summary</div>
                <div style="color:#c8ddf0;font-size:14px;margin-top:4px">3 giá trị được lưu cùng debug output</div>
            </div>
            <div style="color:#8a9dc0;font-size:12px;font-family:monospace;display:flex;gap:12px;flex-wrap:wrap">
                <span>points=${morphology.contour_points ?? "—"}</span>
                <span>area=${_formatNumber(morphology.area)}</span>
                <span>perimeter=${_formatNumber(morphology.perimeter)}</span>
                <span>hull=${_formatNumber(morphology.hull_area)}</span>
            </div>
        </div>
        <div style="overflow-x:auto">
            <table style="width:100%;border-collapse:collapse;font-size:0.92rem">
                <thead>
                    <tr style="border-bottom:1px solid rgba(126,207,255,0.18)">
                        <th style="padding:10px 12px;text-align:left;color:#7ecfff;font-weight:700">Chỉ số</th>
                        <th style="padding:10px 12px;text-align:left;color:#7ecfff;font-weight:700">Giá trị</th>
                        <th style="padding:10px 12px;text-align:left;color:#7ecfff;font-weight:700">Ý nghĩa</th>
                    </tr>
                </thead>
                <tbody>${rows}</tbody>
            </table>
        </div>
    `;

    return wrap;
}

function _formatNumber(val) {
    if (val === null || val === undefined) return "—";
    if (typeof val !== "number") return String(val);
    return Number.isInteger(val) ? val.toString() : val.toFixed(4);
}

function _showImgPopup(src, name) {
    // Xoá popup cũ nếu còn
    const old = document.getElementById("__debug-img-popup");
    if (old) old.remove();

    const overlay = document.createElement("div");
    overlay.id = "__debug-img-popup";
    overlay.style.cssText = [
        "position:fixed","inset:0","z-index:9999",
        "background:rgba(0,0,0,0.85)","backdrop-filter:blur(6px)",
        "display:flex","align-items:center","justify-content:center",
        "padding:24px",
    ].join(";");

    const box = document.createElement("div");
    box.style.cssText = [
        "position:relative","max-width:min(92vw,1200px)","max-height:94vh",
        "display:flex","flex-direction:column","align-items:center","gap:10px",
    ].join(";");

    // Nút đóng
    const closeBtn = document.createElement("button");
    closeBtn.innerHTML = "✕";
    closeBtn.style.cssText = [
        "position:absolute","top:-14px","right:-14px",
        "width:32px","height:32px","border-radius:50%",
        "border:1px solid rgba(255,255,255,0.25)",
        "background:rgba(30,30,40,0.92)","color:#fff",
        "font-size:14px","cursor:pointer",
        "display:flex","align-items:center","justify-content:center",
        "z-index:1","transition:background 0.2s",
    ].join(";");
    closeBtn.onmouseenter = () => closeBtn.style.background = "rgba(80,80,100,0.95)";
    closeBtn.onmouseleave = () => closeBtn.style.background = "rgba(30,30,40,0.92)";
    closeBtn.onclick = () => overlay.remove();

    const img = document.createElement("img");
    img.src = src;
    img.alt = name;
    img.style.cssText = [
        "max-width:100%","max-height:calc(94vh - 50px)",
        "object-fit:contain","border-radius:10px",
        "box-shadow:0 16px 60px rgba(0,0,0,0.7)",
    ].join(";");

    const label = document.createElement("div");
    label.textContent = name;
    label.style.cssText = "color:rgba(255,255,255,0.65);font-size:12px;text-align:center;";

    box.appendChild(closeBtn);
    box.appendChild(img);
    box.appendChild(label);
    overlay.appendChild(box);

    // Click ngoài box → đóng
    overlay.addEventListener("click", (e) => { if (e.target === overlay) overlay.remove(); });

    document.body.appendChild(overlay);
}

/** Khởi tạo — gắn event listeners */
export function initDebugModal() {
  // Đóng khi click backdrop
  const modal = document.getElementById("debug-modal");
  modal.addEventListener("click", (e) => {
    if (e.target === modal) closeDebugModal();
  });

  // Nút đóng
  document
    .getElementById("debug-modal-close")
    .addEventListener("click", closeDebugModal);

  // Nút chạy lại
  document
    .getElementById("debug-rerun-btn")
    .addEventListener("click", rerunDebug);

  // ESC key
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && modal.style.display === "flex") closeDebugModal();
  });

  // Nút debug trong lightbox → mở modal thay vì redirect
  const debugBtn = document.getElementById("debug-btn");
  if (debugBtn) {
    // Clone để xoá listener cũ từ ui.js (redirect → /debug-view)
    const newBtn = debugBtn.cloneNode(true);
    debugBtn.parentNode.replaceChild(newBtn, debugBtn); // fix: thay debugBtn bằng newBtn
    // Gắn handler mới: mở modal debug inline
    document.getElementById("debug-btn").addEventListener("click", function () {
      const filename = this.dataset.filename;
      if (filename) {
        document.getElementById("lightbox").style.display = "none";
        document.body.style.overflow = "";
        openDebugModal(filename);
      }
    });
  }
}
window.openDebugModal = openDebugModal;
