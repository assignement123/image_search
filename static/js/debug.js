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
    _currentDebugMode = "file";
    _currentFilename = filename;
    const modal = document.getElementById("debug-modal");
    const stem  = filename.replace(/\.[^.]+$/, "");

    document.getElementById("debug-rerun-btn").style.display = "inline-flex";

    // Reset UI
    modal.style.display = "flex";
    document.body.style.overflow = "hidden";
    _setDebugState("loading");
    document.getElementById("debug-modal-title").textContent =
        `🔬 Debug: ${filename}`;

    try {
        const res  = await fetch(`/api/debug/${encodeURIComponent(filename)}`);
        const data = await res.json();

        if (!res.ok || data.error) {
            _setDebugState("error", data.stderr || data.error || "Lỗi không xác định");
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

    _renderGroups(debugData.groups, debugData.stem || title.replace(/\.[^.]+$/, ""), debugData.cached);
}

/** Đóng modal */
export function closeDebugModal() {
    document.getElementById("debug-modal").style.display = "none";
    document.body.style.overflow = "";
}

/** Xoá cache và chạy lại */
async function rerunDebug() {
    if (!_currentFilename || _currentDebugMode !== "file") return;
    const stem = _currentFilename.replace(/\.[^.]+$/, "");
    await fetch(`/api/debug-clear/${encodeURIComponent(stem)}`, { method: "DELETE" });
    await openDebugModal(_currentFilename);
}

// ── Internal helpers ──────────────────────────────────────────────────

function _setDebugState(state, msg = "") {
    document.getElementById("debug-loading").style.display = state === "loading" ? "flex" : "none";
    document.getElementById("debug-error").style.display   = state === "error"   ? "flex" : "none";
    document.getElementById("debug-content").style.display = state === "done"    ? "block" : "none";

    if (state === "error") {
        document.getElementById("debug-error-msg").textContent = msg;
    }
}

function _renderGroups(groups, stem, cached) {
    _setDebugState("done");
    const container = document.getElementById("debug-content");
    container.innerHTML = "";

    if (!groups || groups.length === 0) {
        container.innerHTML = '<p style="color:var(--text-muted);padding:40px;text-align:center">Không có kết quả debug</p>';
        return;
    }

    // Badge cached/fresh
    const badge = cached
        ? `<span style="background:rgba(245,158,11,0.15);color:#fbbf24;border:1px solid rgba(245,158,11,0.3);padding:3px 10px;border-radius:999px;font-size:11px;font-weight:600">⚡ Cached</span>`
        : `<span style="background:rgba(34,197,94,0.15);color:#4ade80;border:1px solid rgba(34,197,94,0.3);padding:3px 10px;border-radius:999px;font-size:11px;font-weight:600">✓ Mới tạo</span>`;

    // Tab navigation
    const tabNav = document.createElement("div");
    tabNav.className = "debug-tab-nav";
    tabNav.innerHTML = groups.map((g, i) =>
        `<button class="debug-tab-btn${i === 0 ? " active" : ""}" data-group="${g.id}">${g.title} <span class="debug-tab-count">${g.images.length}</span></button>`
    ).join("") + `<div class="debug-badge-wrap">${badge}</div>`;
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

        group.images.forEach(img => {
            const card = document.createElement("div");
            card.className = "debug-img-card";
            card.innerHTML = `
                <div class="debug-img-wrap">
                    <img src="${img.url}" alt="${img.name}" loading="lazy"
                         onclick="this.closest('.debug-img-card').classList.toggle('debug-img-fullscreen')" />
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
    tabNav.querySelectorAll(".debug-tab-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            tabNav.querySelectorAll(".debug-tab-btn").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            panels.querySelectorAll(".debug-panel").forEach(p => p.classList.remove("active"));
            panels.querySelector(`.debug-panel[data-group="${btn.dataset.group}"]`).classList.add("active");
        });
    });
}

/** Khởi tạo — gắn event listeners */
export function initDebugModal() {
    // Đóng khi click backdrop
    const modal = document.getElementById("debug-modal");
    modal.addEventListener("click", (e) => {
        if (e.target === modal) closeDebugModal();
    });

    // Nút đóng
    document.getElementById("debug-modal-close").addEventListener("click", closeDebugModal);

    // Nút chạy lại
    document.getElementById("debug-rerun-btn").addEventListener("click", rerunDebug);

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
