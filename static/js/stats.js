import { getStats } from './api.js';
import { formatSpecies } from './ui.js';

let speciesChart = null;

export async function loadStats() {
    try {
        const data = await getStats();
        if (data.error) return;

        animateCount("kpi-total", 0, data.total_images, 800);
        animateCount("kpi-species", 0, data.total_species, 600);

        // 2C — Tính đúng tổng chiều vector từ key mới
        const efd   = parseInt(data.extract_params?.dim_efd        || 57);
        const morph = parseInt(data.extract_params?.dim_morphology  || 3);
        const lbp   = parseInt(data.extract_params?.dim_lbp         || 26);
        const glcm  = parseInt(data.extract_params?.dim_glcm        || 20);
        const col   = parseInt(data.extract_params?.dim_color       || 9);
        const vin   = parseInt(data.extract_params?.dim_vein        || 17);
        document.getElementById("kpi-dim").textContent = efd + morph + lbp + glcm + col + vin;

        renderSpeciesChart(data.species_distribution);
        renderParams(data.extract_params);
    } catch (e) {
        console.error("Stats load error", e);
    }
}

function animateCount(id, from, to, duration) {
    const el = document.getElementById(id);
    if (!el) return;
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

    // Tính chiều cao: mỗi loài 32px + padding
    const rowHeight = 32;
    const chartHeight = dist.length * rowHeight + 80;

    const canvas = document.getElementById("species-chart");
    const wrapper = canvas.parentElement;

    // Set kích thước wrapper và canvas trước khi Chart.js init
    wrapper.style.height = chartHeight + "px";
    canvas.style.width  = "100%";
    canvas.style.height = chartHeight + "px";
    canvas.height = chartHeight;

    const ctx = canvas.getContext("2d");

    const labels = dist.map((d) => formatSpecies(d.species));
    const values = dist.map((d) => d.count);
    const colors = labels.map((_, i) => `hsla(${(i * 137.5) % 360}, 65%, 58%, 0.85)`);
    const borderColors = labels.map((_, i) => `hsla(${(i * 137.5) % 360}, 75%, 65%, 1)`);

    if (speciesChart) {
        speciesChart.destroy();
        speciesChart = null;
    }

    speciesChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels,
            datasets: [{
                label: "Số ảnh",
                data: values,
                backgroundColor: colors,
                borderColor: borderColors,
                borderWidth: 1.5,
                borderRadius: 4,
                borderSkipped: false,
            }],
        },
        options: {
            indexAxis: 'y',
            responsive: false,          // tắt responsive để giữ kích thước đặt
            maintainAspectRatio: false,
            animation: { duration: 600, easing: 'easeOutQuart' },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(15,20,35,0.95)',
                    borderColor: 'rgba(255,255,255,0.12)',
                    borderWidth: 1,
                    titleColor: '#e8edf5',
                    bodyColor: '#a0aec0',
                    padding: 10,
                    callbacks: {
                        title: (items) => `🌿 ${items[0].label}`,
                        label: (item) => `  Số ảnh: ${item.parsed.x}`,
                    }
                }
            },
            scales: {
                y: {
                    ticks: {
                        color: "#9ab0c8",
                        font: { size: 12, family: "'Inter', sans-serif" },
                        padding: 6,
                    },
                    grid: { display: false },
                    border: { color: 'rgba(255,255,255,0.06)' },
                },
                x: {
                    ticks: {
                        color: "#5a6a80",
                        font: { size: 11 },
                        stepSize: 10,
                    },
                    grid: {
                        color: 'rgba(255,255,255,0.05)',
                        drawBorder: false,
                    },
                    border: { display: false },
                    beginAtZero: true,
                },
            },
        },
    });
}

function renderParams(params) {
    const grid = document.getElementById("params-grid");
    if (!params || Object.keys(params).length === 0) {
        grid.innerHTML = '<p style="color:var(--text-muted);font-size:13px">Chưa có dữ liệu</p>';
        return;
    }

    // 2B — labelMap với key mới đầy đủ
    const labelMap = {
        harmonics:      "Harmonics (EFD)",
        n_resample:     "N Resample (EFD)",
        glcm_levels:    "GLCM Levels",
        dim_efd:        "Chiều EFD (efd_coeffs)",
        dim_morphology: "Chiều Morphology (morphology_stats)",
        dim_lbp:        "Chiều LBP (lbp_hist)",
        dim_glcm:       "Chiều GLCM (glcm_stats)",
        dim_color:      "Chiều Color (color_moments)",
        dim_vein:       "Chiều Vein (vein_features)",
    };

    grid.innerHTML = "";
    Object.entries(params).forEach(([k, v]) => {
        const item = document.createElement("div");
        item.className = "param-item";
        item.innerHTML = `<div class="param-key">${labelMap[k] || k}</div><div class="param-val">${v}</div>`;
        grid.appendChild(item);
    });
}

/**
 * Điền dữ liệu động vào trang Giới thiệu (tab overview)
 * Gọi một lần khi trang load xong.
 */
export async function loadOverview() {
    try {
        const data = await getStats();
        if (data.error) return;

        const p = data.extract_params || {};
        const efd   = parseInt(p.dim_efd        || 57);
        const morph = parseInt(p.dim_morphology  || 3);
        const lbp   = parseInt(p.dim_lbp         || 26);
        const glcm  = parseInt(p.dim_glcm        || 20);
        const col   = parseInt(p.dim_color       || 9);
        const vin   = parseInt(p.dim_vein        || 17);
        const total = data.dim_total ?? (efd + morph + lbp + glcm + col + vin);

        const set = (id, val) => {
            const el = document.getElementById(id);
            if (el) el.textContent = val;
        };

        // Header badge
        set("ov-total-images", Number(data.total_images).toLocaleString());
        set("ov-total-species", data.total_species);
        set("ov-dim-total",     total);

        // Pipeline badges
        set("ov-dim-efd",        efd);
        set("ov-dim-morphology",  morph);
        set("ov-dim-lbp",         lbp);
        set("ov-dim-glcm",        glcm);
        set("ov-dim-color",        col);
        set("ov-dim-vein",         vin);

        // Feature table badges
        set("tbl-dim-efd",        efd + "D");
        set("tbl-dim-morphology",  morph + "D");
        set("tbl-dim-lbp",         lbp + "D");
        set("tbl-dim-glcm",        glcm + "D");
        set("tbl-dim-color",        col + "D");
        set("tbl-dim-vein",         vin + "D");

        // Params in description
        set("tbl-harmonics",   p.harmonics   || "15");
        set("tbl-glcm-levels", p.glcm_levels || "64");

        // Cascaded total
        set("ov-cascade-total", Number(data.total_images).toLocaleString());

    } catch (e) {
        console.warn("[overview] loadOverview error:", e);
    }
}