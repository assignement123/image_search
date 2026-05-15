import { getStats, buildDatabase, getBuildStatus } from './api.js';
import { loadStats } from './stats.js';

let buildPoller = null;

export async function loadDbBadge() {
    try {
        const data = await getStats();
        if (data.total_images !== undefined) {
            document.getElementById("db-count").textContent = `${data.total_images.toLocaleString()} ảnh · ${data.total_species} loài`;
        }
    } catch (e) {
        document.getElementById("db-count").textContent = "Không kết nối DB";
        const dot = document.querySelector(".db-dot");
        if (dot) dot.style.background = "var(--danger)";
    }
}

export function initManageTab() {
    document.getElementById("btn-build").addEventListener("click", () => triggerBuild(false));
    document.getElementById("btn-rebuild").addEventListener("click", () => {
        if (confirm("⚠️ Bạn có chắc muốn xóa toàn bộ dữ liệu và build lại?\nQuá trình này không thể hoàn tác.")) {
            triggerBuild(true);
        }
    });
}

async function triggerBuild(rebuild) {
    try {
        await buildDatabase(rebuild);
        document.getElementById("progress-section").style.display = "block";
        pollBuildStatus();
    } catch(err) {
        alert(err.message);
    }
}

function pollBuildStatus() {
    if (buildPoller) clearInterval(buildPoller);
    buildPoller = setInterval(async () => {
        try {
            const data = await getBuildStatus();
            updateProgressUI(data);
            if (!data.running) {
                clearInterval(buildPoller);
                loadDbBadge();
                loadStats();
            }
        } catch (e) {
            console.error("Lỗi poll build status", e);
        }
    }, 1000);
}

function updateProgressUI(s) {
    const pct = s.total > 0 ? Math.round((s.progress / s.total) * 100) : 0;
    document.getElementById("progress-bar").style.width = pct + "%";
    document.getElementById("progress-text").textContent = `${s.progress} / ${s.total}`;
    document.getElementById("progress-pct").textContent = pct + "%";
    document.getElementById("p-added").textContent = s.added;
    document.getElementById("p-skipped").textContent = s.skipped;
    document.getElementById("p-errors").textContent = s.errors;
    document.getElementById("progress-msg").textContent = s.message;
}