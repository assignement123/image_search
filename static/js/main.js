// static/js/main.js
import { initLightbox } from "./ui.js";
import { initSearchTab } from "./search.js";
import { loadSpeciesList } from "./browse.js";
import { loadStats, loadOverview } from "./stats.js";
import { initManageTab, loadDbBadge } from "./manage.js";
import { initDebugModal } from "./debug.js";
import { initSearchDebugModal } from "./search_debug.js";
import { initCascadedTab } from "./cascaded.js";

document.addEventListener("DOMContentLoaded", () => {
  // initTabs PHẢI chạy đầu tiên — các init khác không được block nó
  initTabs();

  // Khởi tạo từng module trong try/catch riêng
  // → một module lỗi không kéo chết toàn bộ app
  tryInit("lightbox", () => initLightbox());
  tryInit("debugModal", () => initDebugModal());
  tryInit("searchDebugModal", () => initSearchDebugModal());
  tryInit("searchTab", () => initSearchTab());
  tryInit("cascaded", () => initCascadedTab());
  tryInit("dbBadge", () => loadDbBadge());
  tryInit("manage", () => initManageTab());
  tryInit("overview", () => loadOverview());
});

function tryInit(name, fn) {
  try {
    fn();
  } catch (err) {
    console.error(`[main] initError in "${name}":`, err);
  }
}

function initTabs() {
  const btns = document.querySelectorAll(".nav-btn");

  btns.forEach((btn) => {
    btn.addEventListener("click", () => {
      btns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");

      document
        .querySelectorAll(".tab-content")
        .forEach((t) => t.classList.remove("active"));

      const tabId = btn.dataset.tab;
      const targetTab = document.getElementById(`tab-${tabId}`);
      if (targetTab) targetTab.classList.add("active");

      if (tabId === "stats") {
        tryInit("stats", () => loadStats());
      }

      if (tabId === "browse") {
        const speciesList = document.getElementById("species-list");
        if (speciesList && !speciesList.querySelector(".species-item")) {
          tryInit("browse", () => loadSpeciesList());
        }
      }
    });
  });
}
