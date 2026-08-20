document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabPanels = document.querySelectorAll('.tab-panel');

    // KIS Elements
    const inputKisQuery = document.getElementById('input-kis-query');
    const btnSearchKis = document.getElementById('btn-search-kis');
    const selectMode = document.getElementById('select-mode');
    const selectTopK = document.getElementById('select-topk');
    const selectNms = document.getElementById('select-nms');

    // VQA Elements
    const inputVqaContext = document.getElementById('input-vqa-context');
    const inputVqaQuestion = document.getElementById('input-vqa-question');
    const btnSearchVqa = document.getElementById('btn-search-vqa');
    const selectVqaTopK = document.getElementById('select-vqa-topk');
    const checkVqaAiSuggest = document.getElementById('check-vqa-ai-suggest');

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
    let activeTab = "kis";

    const HF_CDN_URL = "https://huggingface.co/datasets/BaeBaeBoo1010/aic2026-keyframes/resolve/main";

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
            activeTab = btn.getAttribute('data-tab');
            document.getElementById(`panel-${activeTab}`).classList.add('active');
        });
    });

    // ==========================================
    // 1. KIS SEARCH LOGIC
    // ==========================================
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

        showLoading(true, 'Đang tìm kiếm KIS...');

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
                            appendKisBatchCards(chunk.results, chunk.total_count, `⚡ Hoàn tất (${elapsed}s)`);
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

    function appendKisBatchCards(batchResults, totalCount, statusBadge = '') {
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

    // ==========================================
    // 2. VQA INTERACTIVE SEARCH & QA LOGIC
    // ==========================================
    async function performVqaSearch() {
        const context = inputVqaContext.value.trim();
        const question = inputVqaQuestion.value.trim();

        if (!context && !question) {
            alert('Vui lòng nhập Bối cảnh sự kiện hoặc Câu hỏi chi tiết!');
            return;
        }

        const topK = selectVqaTopK ? parseInt(selectVqaTopK.value, 10) : 20;
        const autoAiSuggest = checkVqaAiSuggest ? checkVqaAiSuggest.checked : true;

        showLoading(true, 'Đang quét phân cảnh & phân tích VQA...');

        try {
            const startTime = performance.now();
            const response = await fetch('/api/search/vqa', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    context,
                    question,
                    top_k: topK,
                    auto_ai_suggest: autoAiSuggest
                })
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'Lỗi khi tìm kiếm VQA!');
            }

            const data = await response.json();
            currentResults = data.results || [];
            const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);

            renderVqaResults(currentResults, data.count, question, context, elapsed);
        } catch (err) {
            alert(`Lỗi VQA: ${err.message}`);
            console.error(err);
        } finally {
            showLoading(false);
        }
    }

    if (btnSearchVqa) {
        btnSearchVqa.addEventListener('click', performVqaSearch);
    }
    if (inputVqaQuestion) {
        inputVqaQuestion.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') performVqaSearch();
        });
    }
    if (inputVqaContext) {
        inputVqaContext.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') performVqaSearch();
        });
    }

    function renderVqaResults(results, totalCount, question, context, elapsed) {
        keyframeGrid.innerHTML = '';
        resultsCount.innerHTML = `${results.length} khung hình ứng viên <span style="margin-left: 10px; font-size: 0.85rem; color: #ec4899; font-weight: 600;">✨ VQA Ready (${elapsed}s)</span>`;
        btnExportSub.disabled = results.length === 0;

        results.forEach((item, index) => {
            const rank = index + 1;
            const card = document.createElement('div');
            card.className = 'keyframe-card vqa-card pop-in';
            card.style.animationDelay = `${index * 20}ms`;

            const imgSrc = `${HF_CDN_URL}/${item.video_id}/${item.frame_filename}`;
            const initialAnswer = item.suggested_answer || "";
            item.answer = initialAnswer; // Sync state

            card.innerHTML = `
                <div class="card-img-wrapper shimmer" style="cursor: pointer;">
                    <img src="${imgSrc}" alt="${item.video_id} - frame ${item.frame_idx}" loading="lazy" decoding="async"
                        onload="this.classList.add('loaded'); this.parentElement.classList.remove('shimmer');"
                        onerror="this.parentElement.classList.remove('shimmer');">
                    <div class="card-badge vqa-badge">#${rank} • ${(item.score).toFixed(3)}</div>
                </div>
                <div class="card-content" style="padding-bottom: 0.4rem;">
                    <div class="card-title">${item.video_id}</div>
                    <div class="card-sub">Frame ${item.frame_filename} (${item.pts_time ? item.pts_time.toFixed(1) + 's' : item.frame_idx})</div>
                </div>
                <div class="vqa-answer-container">
                    <div class="vqa-answer-label">
                        <span>✍️ Câu trả lời (Answer):</span>
                        ${initialAnswer ? '<span style="color: #34d399; font-size: 0.72rem;">🤖 AI gợi ý</span>' : ''}
                    </div>
                    <div class="vqa-input-wrapper">
                        <input type="text" class="vqa-answer-input" value="${initialAnswer}" 
                            placeholder="Nhập câu trả lời..." data-idx="${index}">
                        <button class="btn-vqa-ask" data-idx="${index}" title="Hỏi lại Gemini cho riêng khung hình này">
                            🤖 Hỏi AI
                        </button>
                    </div>
                    <div class="vqa-select-row">
                        <label class="vqa-checkbox-label">
                            <input type="checkbox" class="vqa-checkbox" data-idx="${index}" checked>
                            <span>Bao gồm trong CSV nộp bài</span>
                        </label>
                    </div>
                </div>
            `;

            // Click image to zoom in modal
            const imgWrapper = card.querySelector('.card-img-wrapper');
            imgWrapper.addEventListener('click', () => {
                openModal(item, imgSrc, rank);
            });

            // Handle manual answer input change
            const answerInput = card.querySelector('.vqa-answer-input');
            answerInput.addEventListener('input', (e) => {
                item.answer = e.target.value.trim();
            });

            // Handle per-frame Ask AI button
            const btnAsk = card.querySelector('.btn-vqa-ask');
            btnAsk.addEventListener('click', async (e) => {
                e.stopPropagation();
                btnAsk.disabled = true;
                btnAsk.textContent = '⏳ Đang hỏi...';
                try {
                    const resp = await fetch('/api/vqa/ask_frame', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            video_id: item.video_id,
                            frame_filename: item.frame_filename,
                            question: question || inputVqaQuestion.value.trim(),
                            context: context || inputVqaContext.value.trim()
                        })
                    });
                    if (resp.ok) {
                        const resData = await resp.json();
                        if (resData.answer) {
                            answerInput.value = resData.answer;
                            item.answer = resData.answer;
                        }
                    }
                } catch (err) {
                    console.error('Ask AI error:', err);
                } finally {
                    btnAsk.disabled = false;
                    btnAsk.textContent = '🤖 Hỏi AI';
                }
            });

            // Handle checkbox include toggle
            const checkbox = card.querySelector('.vqa-checkbox');
            checkbox.addEventListener('change', (e) => {
                item.include_in_submission = e.target.checked;
            });
            item.include_in_submission = true;

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

    // ==========================================
    // 3. SUBMISSION EXPORT CSV (KIS & VQA & TRAKE)
    // ==========================================
    btnExportSub.addEventListener('click', async () => {
        if (!currentResults || currentResults.length === 0) return;

        let exportPayload = [];
        let queryType = activeTab === "vqa" ? "vqa" : "kis";

        if (activeTab === "vqa") {
            // Task 2: <video_id>, <frame_idx>, <answer>
            exportPayload = currentResults
                .filter(item => item.include_in_submission !== false)
                .map(item => ({
                    video_id: item.video_id,
                    frame_idx: item.frame_idx,
                    answer: item.answer || "Không xác định"
                }));
        } else {
            // Task 1: <video_id>, <frame_idx>
            exportPayload = currentResults.map(item => ({
                video_id: item.video_id,
                frame_idx: item.frame_idx
            }));
        }

        try {
            const response = await fetch('/api/export_submission', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query_id: `${queryType}_query`,
                    predictions: exportPayload
                })
            });

            if (!response.ok) throw new Error('Không thể xuất file nộp bài!');

            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = `submission_${queryType}.csv`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
        } catch (err) {
            alert(`Lỗi xuất CSV: ${err.message}`);
        }
    });

    function showLoading(isLoading, msg = 'Đang tìm kiếm...') {
        if (isLoading) {
            loadingSpinner.innerHTML = `<div class="spinner"></div> ${msg}`;
            loadingSpinner.classList.remove('hidden');
            if (btnSearchKis) btnSearchKis.disabled = true;
            if (btnSearchVqa) btnSearchVqa.disabled = true;
        } else {
            loadingSpinner.classList.add('hidden');
            if (btnSearchKis) btnSearchKis.disabled = false;
            if (btnSearchVqa) btnSearchVqa.disabled = false;
        }
    }
});

