// static/js/main.js
import { initLightbox } from './ui.js';
import { initSearchTab } from './search.js';

import { loadSpeciesList } from './browse.js';
import { loadStats } from './stats.js';
import { initManageTab, loadDbBadge } from './manage.js';

document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initLightbox();
    
    initSearchTab();
    
    loadDbBadge(); 
    initManageTab();
});

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
            if (targetTab) {
                targetTab.classList.add("active");
            }

            if (tabId === "stats") {
                loadStats();
            }
            
            if (tabId === "browse") {
                const speciesList = document.getElementById("species-list");
                if (speciesList && !speciesList.querySelector(".species-item")) {
                    loadSpeciesList();
                }
            }

            // "overview" tab is static — no data loading required
        });
    });
}