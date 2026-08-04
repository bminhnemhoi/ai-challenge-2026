document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabPanels = document.querySelectorAll('.tab-panel');

    const inputKisQuery = document.getElementById('input-kis-query');
    const btnSearchKis = document.getElementById('btn-search-kis');
    const selectTopK = document.getElementById('select-topk');
    const selectNms = document.getElementById('select-nms');

    const keyframeGrid = document.getElementById('keyframe-grid');
    const resultsCount = document.getElementById('results-count');
    const loadingSpinner = document.getElementById('loading-spinner');
    const btnExportSub = document.getElementById('btn-export-sub');

    const modal = document.getElementById('preview-modal');
    const modalImg = document.getElementById('modal-img');
    const modalVideoTitle = document.getElementById('modal-video-title');
    const modalFrameInfo = document.getElementById('modal-frame-info');
    const modalScoreInfo = document.getElementById('modal-score-info');
    const btnCloseModal = document.getElementById('btn-close-modal');

    let currentResults = [];
    let currentQuery = "";

    // Tab Switching Logic
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => {
                b.classList.remove('active');
                b.setAttribute('aria-selected', 'false');
            });
            tabPanels.forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            btn.setAttribute('aria-selected', 'true');
            const targetTab = btn.getAttribute('data-tab');
            document.getElementById(`panel-${targetTab}`).classList.add('active');
        });
    });

    // KIS Search Execution
    async function performKisSearch() {
        const query = inputKisQuery.value.trim();
        if (!query) {
            alert('Vui lòng nhập mô tả sự kiện (Text Query)!');
            return;
        }

        currentQuery = query;
        const topK = parseInt(selectTopK.value, 10);
        const nmsGap = parseInt(selectNms.value, 10);

        showLoading(true);

        try {
            const response = await fetch('/api/search/kis', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, top_k: topK, nms_gap: nmsGap })
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'Lỗi hệ thống tìm kiếm!');
            }

            const data = await response.json();
            currentResults = data.results || [];
            renderResults(currentResults);
        } catch (err) {
            alert(`Lỗi khi tìm kiếm: ${err.message}`);
            console.error(err);
        } finally {
            showLoading(false);
        }
    }

    btnSearchKis.addEventListener('click', performKisSearch);
    inputKisQuery.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') performKisSearch();
    });

    // Render Search Results Grid
    function renderResults(results) {
        keyframeGrid.innerHTML = '';
        resultsCount.textContent = `${results.length} kết quả`;

        if (!results || results.length === 0) {
            keyframeGrid.innerHTML = `
                <div class="empty-state">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
                    <p>Không tìm thấy kết quả nào phù hợp.</p>
                </div>
            `;
            btnExportSub.disabled = true;
            return;
        }

        btnExportSub.disabled = false;

        results.forEach((item, index) => {
            const card = document.createElement('div');
            card.className = 'keyframe-card';
            
            const imgSrc = `/api/keyframe/${item.video_id}/${item.frame_filename}`;

            card.innerHTML = `
                <div class="card-img-wrapper">
                    <img src="${imgSrc}" alt="${item.video_id} - frame ${item.frame_idx}" loading="lazy">
                    <span class="card-rank">#${index + 1}</span>
                    <span class="card-score">${(item.score).toFixed(4)}</span>
                </div>
                <div class="card-body">
                    <div class="card-video-title">${item.video_id}</div>
                    <div class="card-frame-info">Frame Index: <strong>${item.frame_idx}</strong></div>
                </div>
            `;

            card.addEventListener('click', () => openModal(item, imgSrc, index + 1));
            keyframeGrid.appendChild(card);
        });
    }

    // Modal Lightbox Preview
    function openModal(item, imgSrc, rank) {
        modalImg.src = imgSrc;
        modalVideoTitle.textContent = `[#${rank}] ${item.video_id}`;
        modalFrameInfo.textContent = `Khung hình (Frame Index): ${item.frame_idx}`;
        modalScoreInfo.textContent = `Điểm Tương Quan (Cosine Score): ${item.score.toFixed(4)}`;
        modal.classList.remove('hidden');
    }

    btnCloseModal.addEventListener('click', () => modal.classList.add('hidden'));
    modal.querySelector('.modal-overlay').addEventListener('click', () => modal.classList.add('hidden'));

    // Submission Export CSV
    btnExportSub.addEventListener('click', async () => {
        if (!currentResults || currentResults.length === 0) return;

        try {
            const response = await fetch('/api/export_submission', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query_id: "kis_query",
                    predictions: currentResults
                })
            });

            if (!response.ok) throw new Error('Không thể xuất file nộp bài!');

            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = `submission_kis.csv`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
        } catch (err) {
            alert(`Lỗi xuất CSV: ${err.message}`);
        }
    });

    function showLoading(isLoading) {
        if (isLoading) {
            loadingSpinner.classList.remove('hidden');
            btnSearchKis.disabled = true;
        } else {
            loadingSpinner.classList.add('hidden');
            btnSearchKis.disabled = false;
        }
    }
});
