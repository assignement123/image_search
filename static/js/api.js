export async function searchLeaf(formData) {
    const res = await fetch("/api/search", { method: "POST", body: formData });
    if (!res.ok) throw new Error((await res.json()).error || "Lỗi tìm kiếm");
    return res.json();
}

export async function getSpeciesList() {
    const res = await fetch("/api/species");
    return res.json();
}

export async function getSpeciesImages(speciesName, page = 1) {
    const res = await fetch(`/api/species/${encodeURIComponent(speciesName)}?page=${page}&per_page=20`);
    return res.json();
}

export async function getStats() {
    const res = await fetch("/api/stats");
    return res.json();
}