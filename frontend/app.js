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

    // KIS Search Execution with Instant Streaming Chunk Rendering
    async function performKisSearch() {
        const query = inputKisQuery.value.trim();
        if (!query) {
            alert('Vui lòng nhập mô tả sự kiện (Text Query)!');
            return;
        }

        currentQuery = query;
        const topK = parseInt(selectTopK.value, 10);
        const nmsGap = parseInt(selectNms.value, 10);
        const useMetadataBm25 = document.getElementById('check-bm25')?.checked ?? true;
        const useTemporalExpansion = document.getElementById('check-expansion')?.checked ?? true;

        showLoading(true);

        try {
            const response = await fetch('/api/search/kis_stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    query, 
                    top_k: topK, 
                    nms_gap: nmsGap, 
                    use_reranker: false,
                    use_metadata_bm25: useMetadataBm25,
                    use_temporal_expansion: useTemporalExpansion
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

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop(); // keep last incomplete line in buffer

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
                            appendBatchCards(chunk.results, chunk.total_count, '⚡ Đang nạp ảnh tức thì...');
                        } else if (chunk.type === 'rerank_batch') {
                            if (chunk.batch_idx === 0) {
                                currentResults = [];
                            }
                            currentResults = currentResults.concat(chunk.results || []);
                            renderResults(currentResults, '🔥 Đã tái xếp hạng VLM Level 3');
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

    // Append Mini-Batch Cards Incrementally (Left to Right popIn Animation)
    function appendBatchCards(batchResults, totalCount, statusBadge = '') {
        const badgeHtml = statusBadge ? `<span style="margin-left: 10px; font-size: 0.85rem; color: #10b981; font-weight: 600;">${statusBadge}</span>` : '';
        resultsCount.innerHTML = `${keyframeGrid.children.length + batchResults.length} / ${totalCount} kết quả ${badgeHtml}`;
        btnExportSub.disabled = false;

        batchResults.forEach((item, index) => {
            const card = document.createElement('div');
            card.className = 'keyframe-card pop-in';
            card.style.animationDelay = `${index * 50}ms`;

            const imgSrc = `${HF_CDN_URL}/${item.video_id}/${item.frame_filename}`;

            card.innerHTML = `
                <div class="card-img-wrapper">
                    <img src="${imgSrc}" alt="${item.video_id} - frame ${item.frame_idx}" loading="lazy" decoding="async">
                    <div class="card-badge">Score: ${item.score.toFixed(4)}</div>
                </div>
                <div class="card-content">
                    <div class="card-title">${item.video_id}</div>
                    <div class="card-sub">Frame: ${item.frame_filename} (${item.frame_idx})</div>
                </div>
            `;

            card.addEventListener('click', () => {
                showImageModal(item, imgSrc);
            });

            keyframeGrid.appendChild(card);
        });
    }

    // Render Search Results Grid
    function renderResults(results, statusBadge = '') {
        keyframeGrid.innerHTML = '';
        const badgeHtml = statusBadge ? `<span style="margin-left: 10px; font-size: 0.85rem; color: #10b981; font-weight: 600;">${statusBadge}</span>` : '';
        resultsCount.innerHTML = `${results.length} kết quả ${badgeHtml}`;

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
            
            const imgSrc = `${HF_CDN_URL}/${item.video_id}/${item.frame_filename}`;

            card.innerHTML = `
                <div class="card-img-wrapper">
                    <img src="${imgSrc}" alt="${item.video_id} - frame ${item.frame_idx}" loading="lazy" decoding="async">
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
