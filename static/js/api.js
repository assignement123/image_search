export async function searchLeaf(formData) {
    const res = await fetch("/api/search", { method: "POST", body: formData });
    if (!res.ok) throw new Error((await res.json()).error || "Lỗi tìm kiếm");
    return res.json();
}

export async function debugSearchLeaf(formData) {
    const res = await fetch("/api/search/debug", { method: "POST", body: formData });
    if (!res.ok) throw new Error((await res.json()).error || "Lỗi debug search");
    return res.json();
}

export async function debugUploadImage(formData) {
    const res = await fetch("/api/debug-upload", { method: "POST", body: formData });
    if (!res.ok) throw new Error((await res.json()).error || "Lỗi debug ảnh input");
    return res.json();
}

export async function getSpeciesList() {
    const res = await fetch("/api/species");
    return res.json();
}

export async function getSpeciesImages(speciesName, page = 1) {
    const res = await fetch(`/api/species/${encodeURIComponent(speciesName)}?page=${page}&per_page=24`);
    return res.json();
}

export async function getStats() {
    const res = await fetch("/api/stats");
    return res.json();
}

export async function buildDatabase(rebuild) {
    const res = await fetch("/api/build", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rebuild }),
    });
    if (!res.ok) throw new Error((await res.json()).error || "Lỗi khi build DB");
    return res.json();
}

export async function getBuildStatus() {
    const res = await fetch("/api/build/status");
    return res.json();
}