// static/js/main.js
import { initLightbox } from './ui.js';
import { initSearchTab } from './search.js';

// TODO: Bỏ comment các import này khi bạn tách xong các file tương ứng
// import { loadSpeciesList } from './browse.js';
// import { loadStats } from './stats.js';
// import { initManageTab, loadDbBadge } from './manage.js';

document.addEventListener("DOMContentLoaded", () => {
    // 1. Khởi tạo UI dùng chung
    initTabs();
    initLightbox();
    
    // 2. Khởi tạo chức năng cho từng tab
    initSearchTab();
    
    // TODO: Bỏ comment khi hoàn thiện các module
    // loadDbBadge(); 
    // initManageTab();
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
                // TODO: Bỏ comment gọi hàm khi stats.js đã sẵn sàng
                // loadStats();
            }
            
            if (tabId === "browse") {
                const speciesList = document.getElementById("species-list");
                if (speciesList && !speciesList.querySelector(".species-item")) {
                    // TODO: Bỏ comment gọi hàm khi browse.js đã sẵn sàng
                    // loadSpeciesList();
                }
            }
        });
    });
}