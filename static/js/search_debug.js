// static/js/search_debug.js
import { formatSpecies } from './ui.js';

export function openSearchDebugModal(payload) {
    const modal = document.getElementById("search-debug-modal");
    if (!modal) return;

    modal.style.display = "flex";
    document.body.style.overflow = "hidden";

    document.getElementById("search-debug-loading").style.display = "none";
    document.getElementById("search-debug-error").style.display = "none";
    document.getElementById("search-debug-content").style.display = "block";

    document.getElementById("search-debug-title").textContent =
        `🔎 Debug search: top ${payload.top_k ?? payload.count ?? 0}`;

    if (payload.error) {
        showSearchDebugError(payload.error);
        return;
    }

    renderSearchDebug(payload);
}

export function openSearchDebugLoading() {
    const modal = document.getElementById("search-debug-modal");
    if (!modal) return;

    modal.style.display = "flex";
    document.body.style.overflow = "hidden";
    document.getElementById("search-debug-title").textContent = "🔎 Debug search";
    document.getElementById("search-debug-loading").style.display = "flex";
    document.getElementById("search-debug-error").style.display = "none";
    document.getElementById("search-debug-content").style.display = "none";
}

export function closeSearchDebugModal() {
    const modal = document.getElementById("search-debug-modal");
    if (!modal) return;
    modal.style.display = "none";
    document.body.style.overflow = "";
}

export function initSearchDebugModal() {
    const modal = document.getElementById("search-debug-modal");
    if (!modal) return;

    modal.addEventListener("click", (e) => {
        if (e.target === modal) closeSearchDebugModal();
    });

    const closeBtn = document.getElementById("search-debug-close");
    if (closeBtn) {
        closeBtn.addEventListener("click", closeSearchDebugModal);
    }

    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && modal.style.display === "flex") {
            closeSearchDebugModal();
        }
    });
}

function showSearchDebugError(msg) {
    document.getElementById("search-debug-loading").style.display = "none";
    document.getElementById("search-debug-error").style.display = "flex";
    document.getElementById("search-debug-content").style.display = "none";
    document.getElementById("search-debug-error-msg").textContent = msg;
}

function renderSearchDebug(payload) {
    const content = document.getElementById("search-debug-content");
    content.innerHTML = "";

    const meta = payload.meta || {};
    const weights = meta.weights || {};
    const weightKeys = ["w_efd", "w_morphology", "w_lbp", "w_glcm", "w_color", "w_vein"];
    const weightLabels = {
        w_efd: "EFD",
        w_morphology: "Morphology",
        w_lbp: "LBP",
        w_glcm: "GLCM",
        w_color: "Color",
        w_vein: "Vein",
    };

    const summary = document.createElement("div");
    summary.className = "search-debug-summary";
    summary.innerHTML = `
        <div class="search-debug-summary-card">
            <div class="search-debug-summary-label">Top K</div>
            <div class="search-debug-summary-value">${payload.top_k ?? payload.count ?? 0}</div>
        </div>
        <div class="search-debug-summary-card">
            <div class="search-debug-summary-label">Tổng trọng số</div>
            <div class="search-debug-summary-value">${formatNum(meta.sum_weights)}</div>
        </div>
        ${weightKeys.map((key) => `
            <div class="search-debug-summary-card">
                <div class="search-debug-summary-label">${weightLabels[key]}</div>
                <div class="search-debug-summary-value">${formatNum(weights[key])}</div>
            </div>
        `).join("")}
    `;
    content.appendChild(summary);

    const note = document.createElement("div");
    note.className = "search-debug-note";
    note.textContent = "Similarity = sum(weight × exp(-gamma × distance)) / sum(weights)";
    content.appendChild(note);

    if (!payload.results || payload.results.length === 0) {
        content.innerHTML += '<p style="color:var(--text-muted);padding:40px;text-align:center">Không có kết quả debug</p>';
        return;
    }

    const wrap = document.createElement("div");
    wrap.className = "search-debug-table-wrap";

    const table = document.createElement("table");
    table.className = "search-debug-table";
    table.innerHTML = `
        <thead>
            <tr>
                <th>#</th>
                <th>Ảnh</th>
                <th>Loài</th>
                <th>Similarity</th>
                <th>EFD</th>
                <th>Morphology</th>
                <th>LBP</th>
                <th>GLCM</th>
                <th>Color</th>
                <th>Vein</th>
            </tr>
        </thead>
        <tbody>
            ${payload.results.map((row, index) => renderRow(row, index + 1)).join("")}
        </tbody>
    `;

    wrap.appendChild(table);
    content.appendChild(wrap);
}

function renderRow(row, rank) {
    const cells = {
        efd: getBreakdownCell(row.breakdown, "efd"),
        morphology: getBreakdownCell(row.breakdown, "morphology"),
        lbp: getBreakdownCell(row.breakdown, "lbp"),
        glcm: getBreakdownCell(row.breakdown, "glcm"),
        color: getBreakdownCell(row.breakdown, "color"),
        vein: getBreakdownCell(row.breakdown, "vein"),
    };

    return `
        <tr>
            <td class="search-debug-rank">${rank}</td>
            <td>
                <div class="search-debug-file">${row.filename}</div>
            </td>
            <td>${formatSpecies(row.species)}</td>
            <td><strong>${formatNum(row.similarity)}%</strong></td>
            <td>${cells.efd}</td>
            <td>${cells.morphology}</td>
            <td>${cells.lbp}</td>
            <td>${cells.glcm}</td>
            <td>${cells.color}</td>
            <td>${cells.vein}</td>
        </tr>
    `;
}

function getBreakdownCell(breakdown, key) {
    const item = (breakdown || []).find((entry) => entry.key === key);
    if (!item) return "—";

    return `
        <div class="search-debug-feature" title="weight=${formatNum(item.weight)} | gamma=${formatNum(item.gamma)} | distance=${formatNum(item.distance)} | score=${formatNum(item.score)}">
            <div class="search-debug-feature-main">${formatNum(item.contribution)}%</div>
            <div class="search-debug-feature-sub">d=${formatNum(item.distance)} · s=${formatNum(item.score)}</div>
        </div>
    `;
}

function formatNum(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
    return Number(value).toFixed(2).replace(/\.00$/, "");
}
