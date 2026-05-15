import { getStats } from './api.js';
import { formatSpecies } from './ui.js';

let speciesChart = null;
export async function loadStats() {
    try {
        const data = await getStats();
        if (data.error) return;

        animateCount("kpi-total", 0, data.total_images, 800);
        animateCount("kpi-species", 0, data.total_species, 600);

        const efd = parseInt(data.extract_params?.dim_efd || 76);
        const tex = parseInt(data.extract_params?.dim_texture || 46);
        const col = parseInt(data.extract_params?.dim_color || 9);
        const vin = parseInt(data.extract_params?.dim_vein || 9);
        document.getElementById("kpi-dim").textContent = efd + tex + col + vin;

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
    const ctx = document.getElementById("species-chart").getContext("2d");
    const top = dist.slice(0, 20);
    
    const labels = top.map((d) => formatSpecies(d.species));
    const values = top.map((d) => d.count);
    const colors = labels.map((_, i) => `hsla(${(i * 137.5) % 360}, 65%, 55%, 0.8)`);

    if (speciesChart) speciesChart.destroy();

    speciesChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels,
            datasets: [{
                label: "Số ảnh",
                data: values,
                backgroundColor: colors,
                borderColor: colors.map((c) => c.replace("0.8", "1")),
                borderWidth: 1,
                borderRadius: 5,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { ticks: { color: "#7a8499", font: { size: 11 }, maxRotation: 45 } },
                y: { ticks: { color: "#7a8499", font: { size: 11 } }, beginAtZero: true },
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

    const labelMap = {
        harmonics: "Harmonics (EFD)", n_resample: "N Resample", glcm_levels: "GLCM Levels",
        dim_efd: "Chiều EFD", dim_texture: "Chiều Texture", dim_color: "Chiều Color",
        dim_vein: "Chiều Vein", background: "Nền ảnh", created_by: "Script tạo",
    };

    grid.innerHTML = "";
    Object.entries(params).forEach(([k, v]) => {
        const item = document.createElement("div");
        item.className = "param-item";
        item.innerHTML = `<div class="param-key">${labelMap[k] || k}</div><div class="param-val">${v}</div>`;
        grid.appendChild(item);
    });
}