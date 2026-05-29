const safeName = document.body?.dataset?.safeName || "";
let currentRows = [];
let currentSelected = null;
let lastActiveFilename = safeName;

function toNumberArray(value) {
    if (!Array.isArray(value)) return [];
    return value.map(v => Number(v));
}

function topDiffIndices(a, b, limit = 5) {
    const size = Math.min(a.length, b.length);
    const pairs = [];
    for (let i = 0; i < size; i++) {
        pairs.push({ i, d: Math.abs(a[i] - b[i]) });
    }
    pairs.sort((x, y) => y.d - x.d);
    return pairs.slice(0, limit);
}

function vectorSummary(a, b) {
    const size = Math.min(a.length, b.length);
    if (!size) return { mad: 0, cosine: 0 };
    let sumAbs = 0;
    let dot = 0;
    let na = 0;
    let nb = 0;
    for (let i = 0; i < size; i++) {
        const x = a[i];
        const y = b[i];
        sumAbs += Math.abs(x - y);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    const mad = sumAbs / size;
    const cosine = dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-9);
    return { mad, cosine };
}

function renderSparkline(svgId, queryVec, resultVec) {
    const svg = document.getElementById(svgId);
    if (!svg) return;
    const w = 360;
    const h = 120;
    const pad = 10;
    const n = Math.min(queryVec.length, resultVec.length);
    if (!n) {
        svg.innerHTML = "";
        return;
    }

    const all = queryVec.concat(resultVec);
    const minV = Math.min(...all);
    const maxV = Math.max(...all);
    const range = Math.max(maxV - minV, 1e-9);

    const toPoints = (vec) => vec.slice(0, n).map((v, i) => {
        const x = pad + (i * (w - 2 * pad)) / Math.max(n - 1, 1);
        const y = h - pad - ((v - minV) / range) * (h - 2 * pad);
        return `${x.toFixed(2)},${y.toFixed(2)}`;
    }).join(" ");

    svg.innerHTML = `
        <line x1="${pad}" y1="${h - pad}" x2="${w - pad}" y2="${h - pad}" stroke="#333" stroke-width="1" />
        <polyline points="${toPoints(queryVec)}" fill="none" stroke="#37a0ff" stroke-width="2" />
        <polyline points="${toPoints(resultVec)}" fill="none" stroke="#ff9a3c" stroke-width="2" />
    `;
}

function renderDiffBars(svgId, hoverId, label, queryVec, resultVec) {
    const svg = document.getElementById(svgId);
    const hover = document.getElementById(hoverId);
    if (!svg) return;
    const w = 360;
    const h = 90;
    const pad = 8;
    const n = Math.min(queryVec.length, resultVec.length);
    if (!n) {
        svg.innerHTML = "";
        if (hover) hover.textContent = "";
        return;
    }
    const diff = [];
    for (let i = 0; i < n; i++) diff.push(Math.abs(queryVec[i] - resultVec[i]));
    const maxDiff = Math.max(...diff, 1e-9);
    const barW = (w - 2 * pad) / n;
    let bars = `<line x1="${pad}" y1="${h - pad}" x2="${w - pad}" y2="${h - pad}" stroke="#333" stroke-width="1" />`;
    diff.forEach((d, i) => {
        const x = pad + i * barW;
        const bh = (d / maxDiff) * (h - 2 * pad);
        const y = h - pad - bh;
        const bw = Math.max(barW - 0.8, 0.2).toFixed(2);
        bars += `<rect data-idx="${i}" data-diff="${d}" x="${x.toFixed(2)}" y="${y.toFixed(2)}" width="${bw}" height="${bh.toFixed(2)}" fill="#a24bff" opacity="0.9"><title>${label} #${i} | abs diff=${d.toExponential(4)}</title></rect>`;
    });
    svg.innerHTML = bars;
    if (hover) hover.textContent = "Hover cột để xem giá trị chênh lệch.";

    svg.onmousemove = (e) => {
        const target = e.target;
        if (!hover || !target || target.tagName !== "rect") return;
        const idx = target.getAttribute("data-idx");
        const d = Number(target.getAttribute("data-diff") || 0);
        hover.textContent = `${label} #${idx} | abs diff=${d.toExponential(4)}`;
    };
    svg.onmouseleave = () => {
        if (hover) hover.textContent = "Hover cột để xem giá trị chênh lệch.";
    };
}

function renderSelectedPreview(row) {
    const box = document.getElementById("selected-preview");
    if (!box || !row) return;
    box.style.display = "block";
    box.innerHTML = `
        <div class="preview-grid">
            <div class="preview-item">
                <p>Query: ${safeName}</p>
                <img src="/api/image/${safeName}" alt="query" />
            </div>
            <div class="preview-item">
                <p>Result: ${row.filename}</p>
                <img src="${row.image_url}" alt="result" />
            </div>
        </div>
    `;
}

function buildExportRows() {
    return currentRows.map((row, idx) => {
        const t = row?.explain?.texture;
        return {
            rank: idx + 1,
            filename: row.filename,
            species: row.species,
            similarity: Number(row.similarity || 0),
            lbp_distance: t ? Number(t.lbp.distance) : null,
            lbp_gamma: t ? Number(t.lbp.gamma) : null,
            lbp_score: t ? Number(t.lbp.score) : null,
            lbp_weighted: t ? Number(t.lbp.weighted_contribution) : null,
            glcm_distance: t ? Number(t.glcm.distance) : null,
            glcm_gamma: t ? Number(t.glcm.gamma) : null,
            glcm_score: t ? Number(t.glcm.score) : null,
            glcm_weighted: t ? Number(t.glcm.weighted_contribution) : null,
        };
    });
}

function contributionSummary(texture) {
    if (!texture) return null;
    const lbpW = Number(texture.lbp?.weighted_contribution || 0);
    const glcmW = Number(texture.glcm?.weighted_contribution || 0);
    const total = lbpW + glcmW + 1e-12;
    const lbpRatio = lbpW / total;
    const glcmRatio = glcmW / total;
    let lead = "LBP";
    if (glcmW > lbpW) lead = "GLCM";
    const weak = [];
    if (lbpRatio < 0.35) weak.push("LBP");
    if (glcmRatio < 0.35) weak.push("GLCM");
    return { lbpRatio, glcmRatio, lead, weak };
}

function downloadTextFile(filename, content, mime) {
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
}

function exportExplainJson() {
    const metaRaw = sessionStorage.getItem("lastSearchExplainMeta");
    let meta = null;
    try {
        meta = metaRaw ? JSON.parse(metaRaw) : null;
    } catch (_e) {}
    const payload = {
        query_filename: safeName,
        selected_filename: currentSelected?.filename || null,
        explain_meta: meta,
        results: buildExportRows(),
    };
    downloadTextFile(`texture_explain_${safeName}.json`, JSON.stringify(payload, null, 2), "application/json");
}

function exportExplainCsv() {
    const rows = buildExportRows();
    const headers = [
        "rank","filename","species","similarity","lbp_distance","lbp_gamma","lbp_score","lbp_weighted",
        "glcm_distance","glcm_gamma","glcm_score","glcm_weighted"
    ];
    const lines = [headers.join(",")];
    rows.forEach(r => {
        const vals = headers.map(h => {
            const v = r[h];
            if (v === null || v === undefined) return "";
            const s = String(v).replaceAll('"', '""');
            return `"${s}"`;
        });
        lines.push(vals.join(","));
    });
    downloadTextFile(`texture_explain_${safeName}.csv`, lines.join("\n"), "text/csv;charset=utf-8");
}

function renderVectorCompare(row) {
    const charts = document.getElementById("charts");
    const metaRaw = sessionStorage.getItem("lastSearchExplainMeta");
    const t = row?.explain?.texture;
    if (!t || !metaRaw) {
        charts.style.display = "none";
        return;
    }

    let meta = null;
    try {
        meta = JSON.parse(metaRaw);
    } catch (_e) {
        charts.style.display = "none";
        return;
    }
    const qTexture = meta?.query_texture;
    const qLbp = toNumberArray(qTexture?.lbp);
    const qGlcm = toNumberArray(qTexture?.glcm);
    const rLbp = toNumberArray(t?.lbp?.vector);
    const rGlcm = toNumberArray(t?.glcm?.vector);
    if (!qLbp.length || !qGlcm.length || !rLbp.length || !rGlcm.length) {
        charts.style.display = "none";
        return;
    }

    charts.style.display = "block";
    renderSparkline("lbp-chart", qLbp, rLbp);
    renderSparkline("glcm-chart", qGlcm, rGlcm);
    renderDiffBars("lbp-diff-chart", "lbp-hover", "LBP bin", qLbp, rLbp);
    renderDiffBars("glcm-diff-chart", "glcm-hover", "GLCM idx", qGlcm, rGlcm);

    const lbpSum = vectorSummary(qLbp, rLbp);
    const glcmSum = vectorSummary(qGlcm, rGlcm);
    const lbpTop = topDiffIndices(qLbp, rLbp, 5);
    const glcmTop = topDiffIndices(qGlcm, rGlcm, 5);
    document.getElementById("lbp-delta").innerHTML = `MAD=${lbpSum.mad.toExponential(3)}, cosine=${lbpSum.cosine.toFixed(4)} | Top diff bins: ${lbpTop.map(x => `#${x.i}(${x.d.toExponential(2)})`).join(", ")}`;
    document.getElementById("glcm-delta").innerHTML = `MAD=${glcmSum.mad.toExponential(3)}, cosine=${glcmSum.cosine.toFixed(4)} | Top diff idx: ${glcmTop.map(x => `#${x.i}(${x.d.toExponential(2)})`).join(", ")}`;
}

function renderSelectedMetrics(row) {
    const box = document.getElementById("selected-metrics");
    const t = row?.explain?.texture;
    if (!t) {
        box.innerHTML = "Không có dữ liệu explain cho ảnh này.";
        document.getElementById("charts").style.display = "none";
        return;
    }
    box.innerHTML = `
        <div style="margin:8px 0"><span class="chip">LBP metric: ${t.lbp.metric}</span><span class="chip">GLCM metric: ${t.glcm.metric}</span></div>
        <div>LBP distance: <b>${t.lbp.distance.toFixed(6)}</b>, gamma: <b>${t.lbp.gamma.toFixed(6)}</b>, score: <b>${t.lbp.score.toFixed(6)}</b>, weighted: <b>${t.lbp.weighted_contribution.toFixed(6)}</b></div>
        <div style="margin-top:6px">GLCM distance: <b>${t.glcm.distance.toFixed(6)}</b>, gamma: <b>${t.glcm.gamma.toFixed(6)}</b>, score: <b>${t.glcm.score.toFixed(6)}</b>, weighted: <b>${t.glcm.weighted_contribution.toFixed(6)}</b></div>
    `;
    currentSelected = row;
    lastActiveFilename = row.filename;
    renderSelectedPreview(row);
    renderVectorCompare(row);
}

function sortRows(rows) {
    const key = document.getElementById("sort-key")?.value || "rank";
    const order = document.getElementById("sort-order")?.value || "asc";
    const sign = order === "desc" ? -1 : 1;

    const withRank = rows.map((r, idx) => ({ ...r, __rank: idx + 1 }));
    withRank.sort((a, b) => {
        const ta = a?.explain?.texture;
        const tb = b?.explain?.texture;
        let va = 0;
        let vb = 0;
        if (key === "rank") { va = a.__rank; vb = b.__rank; }
        else if (key === "similarity") { va = Number(a.similarity || 0); vb = Number(b.similarity || 0); }
        else if (key === "lbp_distance") { va = Number(ta?.lbp?.distance || 0); vb = Number(tb?.lbp?.distance || 0); }
        else if (key === "glcm_distance") { va = Number(ta?.glcm?.distance || 0); vb = Number(tb?.glcm?.distance || 0); }
        else if (key === "lbp_weighted") { va = Number(ta?.lbp?.weighted_contribution || 0); vb = Number(tb?.lbp?.weighted_contribution || 0); }
        else if (key === "glcm_weighted") { va = Number(ta?.glcm?.weighted_contribution || 0); vb = Number(tb?.glcm?.weighted_contribution || 0); }
        if (va === vb) return 0;
        return (va > vb ? 1 : -1) * sign;
    });
    return withRank;
}

function renderExplainTable() {
    const raw = sessionStorage.getItem("lastSearchResults");
    const selectedRaw = sessionStorage.getItem("debugSelectedResult");
    if (!raw) return;

    let rows = [];
    try {
        rows = JSON.parse(raw) || [];
    } catch (_e) {
        return;
    }
    if (!Array.isArray(rows) || rows.length === 0) return;
    currentRows = rows;

    const tbody = document.getElementById("explain-body");
    const table = document.getElementById("explain-table");
    tbody.innerHTML = "";

    let activeFilename = safeName;
    if (selectedRaw) {
        try {
            const selected = JSON.parse(selectedRaw);
            if (selected && selected.filename) activeFilename = selected.filename;
        } catch (_e) {}
    }

    const sortedRows = sortRows(rows);
    sortedRows.forEach((row, idx) => {
        const texture = row?.explain?.texture;
        const c = contributionSummary(texture);
        const contribHtml = c
            ? `<div class="muted">Lead: ${c.lead} | LBP ${(c.lbpRatio * 100).toFixed(1)}% | GLCM ${(c.glcmRatio * 100).toFixed(1)}%${c.weak.length ? ` <span class="warn">⚠ Weak: ${c.weak.join("+")}</span>` : ""}</div>`
            : "";
        const tr = document.createElement("tr");
        if (row.filename === activeFilename || row.filename === lastActiveFilename) {
            tr.classList.add("active");
            renderSelectedMetrics(row);
        }
        tr.innerHTML = `
            <td>${idx + 1}</td>
            <td>${row.filename}${contribHtml}</td>
            <td>${texture ? texture.lbp.distance.toFixed(4) : "-"}</td>
            <td>${texture ? texture.glcm.distance.toFixed(4) : "-"}</td>
            <td>${Number(row.similarity || 0).toFixed(2)}</td>
        `;
        tr.addEventListener("click", () => {
            document.querySelectorAll("#explain-body tr").forEach(el => el.classList.remove("active"));
            tr.classList.add("active");
            renderSelectedMetrics(row);
        });
        tbody.appendChild(tr);
    });

    if (!document.querySelector("#explain-body tr.active")) {
        renderSelectedMetrics(rows[0]);
        const first = document.querySelector("#explain-body tr");
        if (first) first.classList.add("active");
    }

    table.style.display = "table";
}

fetch(`/api/debug/${safeName}`)
    .then(res => res.json())
    .then(data => {
        if (data.error) { document.getElementById("debug").innerHTML = "❌ " + data.error; return; }
        let html = "";
        data.images.forEach(img => html += `<h4>${img.group}</h4><img src="${img.url}" width="600"/>`);
        document.getElementById("debug").innerHTML = html;
    });

document.getElementById("export-json")?.addEventListener("click", exportExplainJson);
document.getElementById("export-csv")?.addEventListener("click", exportExplainCsv);
document.getElementById("sort-key")?.addEventListener("change", renderExplainTable);
document.getElementById("sort-order")?.addEventListener("change", renderExplainTable);

renderExplainTable();