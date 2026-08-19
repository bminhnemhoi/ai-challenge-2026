document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabPanels = document.querySelectorAll('.tab-panel');

    const inputKisQuery = document.getElementById('input-kis-query');
    const btnSearchKis = document.getElementById('btn-search-kis');
    const selectMode = document.getElementById('select-mode');
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

    // KIS Search Execution with Instant Skeleton & Metadata Rendering (< 0.1s)
    async function performKisSearch() {
        const query = inputKisQuery.value.trim();
        if (!query) {
            alert('Vui lòng nhập mô tả sự kiện (Text Query)!');
            return;
        }

        currentQuery = query;
        const topK = parseInt(selectTopK.value, 10);
        const nmsGap = parseInt(selectNms.value, 10);
        const isRerank = selectMode ? selectMode.value === 'rerank' : false;

        showLoading(true);

        try {
            const startTime = performance.now();
            const response = await fetch('/api/search/kis_stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    query, 
                    top_k: topK, 
                    nms_gap: nmsGap, 
                    use_reranker: isRerank,
                    reranker_mode: isRerank ? 'siglip_late' : 'off',
                    use_metadata_bm25: true,
                    use_temporal_expansion: false
                })
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'Lỗi hệ thống tìm kiếm!');
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let buffer = '';
            let isFirstBatch = true;
            currentResults = [];

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop();

                for (const line of lines) {
                    if (!line.trim()) continue;
                    try {
                        const chunk = JSON.parse(line);
                        if (chunk.type === 'stage1_batch') {
                            showLoading(false);
                            if (isFirstBatch) {
                                keyframeGrid.innerHTML = '';
                                isFirstBatch = false;
                            }
                            currentResults = currentResults.concat(chunk.results || []);
                            const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);
                            appendBatchCards(chunk.results, chunk.total_count, `⚡ Hoàn tất (${elapsed}s)`);
                        }
                    } catch (e) {
                        console.error('Error parsing stream chunk:', e);
                    }
                }
            }
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

    const HF_CDN_URL = "https://huggingface.co/datasets/BaeBaeBoo1010/aic2026-keyframes/resolve/main";

    // Append Cards with Instant Metadata & Shimmer Skeleton Loading
    function appendBatchCards(batchResults, totalCount, statusBadge = '') {
        const badgeHtml = statusBadge ? `<span style="margin-left: 10px; font-size: 0.85rem; color: #10b981; font-weight: 600;">${statusBadge}</span>` : '';
        resultsCount.innerHTML = `${keyframeGrid.children.length + batchResults.length} / ${totalCount} kết quả ${badgeHtml}`;
        btnExportSub.disabled = false;

        batchResults.forEach((item, index) => {
            const globalRank = currentResults.length - batchResults.length + index + 1;
            const card = document.createElement('div');
            card.className = 'keyframe-card pop-in';
            card.style.animationDelay = `${index * 20}ms`;

            const imgSrc = `${HF_CDN_URL}/${item.video_id}/${item.frame_filename}`;

            card.innerHTML = `
                <div class="card-img-wrapper shimmer">
                    <img src="${imgSrc}" alt="${item.video_id} - frame ${item.frame_idx}" loading="lazy" decoding="async"
                        onload="this.classList.add('loaded'); this.parentElement.classList.remove('shimmer');"
                        onerror="this.parentElement.classList.remove('shimmer');">
                    <div class="card-badge">#${globalRank} • ${(item.score).toFixed(3)}</div>
                </div>
                <div class="card-content">
                    <div class="card-title">${item.video_id}</div>
                    <div class="card-sub">Frame ${item.frame_filename} (${item.pts_time ? item.pts_time.toFixed(1) + 's' : item.frame_idx})</div>
                </div>
            `;

            card.addEventListener('click', () => {
                openModal(item, imgSrc, globalRank);
            });

            keyframeGrid.appendChild(card);
        });
    }

    // Modal Lightbox Preview
    function openModal(item, imgSrc, rank) {
        modalImg.src = imgSrc;
        modalVideoTitle.textContent = `[#${rank || 1}] Video: ${item.video_id}`;
        modalFrameInfo.textContent = `Khung hình: ${item.frame_filename} (Frame Index: ${item.frame_idx}${item.pts_time ? ' • ' + item.pts_time.toFixed(1) + 's' : ''})`;
        modalScoreInfo.textContent = `Độ tương quan (Cosine Similarity): ${(item.score).toFixed(4)}`;
        modal.classList.remove('hidden');
    }

    function closeModal() {
        modal.classList.add('hidden');
        modalImg.src = '';
    }

    btnCloseModal.addEventListener('click', closeModal);
    modal.querySelector('.modal-overlay').addEventListener('click', closeModal);
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !modal.classList.contains('hidden')) {
            closeModal();
        }
    });

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

    const checkPureSiglip = document.getElementById('check-pure-siglip');
    if (checkPureSiglip) {
        checkPureSiglip.addEventListener('change', (e) => {
            if (e.target.checked) {
                inputKisQuery.placeholder = "Enter standard English prompt for raw Google SigLIP 2 (e.g. 'a red sports car on the road', 'a cute cat on a sofa')...";
            } else {
                inputKisQuery.placeholder = "Ví dụ: Tìm video về một diễn giả mặc áo đỏ phát biểu tại cuộc họp báo...";
            }
        });
    }
});
