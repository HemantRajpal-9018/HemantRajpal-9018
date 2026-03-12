/**
 * Spine AI — Research Agent Platform
 * Frontend JavaScript for real-time workflow visualization.
 */

(function () {
    'use strict';

    // DOM Elements
    const researchInput = document.getElementById('researchInput');
    const workflowSection = document.getElementById('workflowSection');
    const sessionsSection = document.getElementById('sessionsSection');
    const queryInput = document.getElementById('queryInput');
    const startResearchBtn = document.getElementById('startResearch');
    const backBtn = document.getElementById('backBtn');
    const workflowQuery = document.getElementById('workflowQuery');
    const stepsContainer = document.getElementById('stepsContainer');
    const feedContainer = document.getElementById('feedContainer');
    const sourcesGrid = document.getElementById('sourcesGrid');
    const sourceCount = document.getElementById('sourceCount');
    const reportSection = document.getElementById('reportSection');
    const reportContent = document.getElementById('reportContent');
    const sessionsList = document.getElementById('sessionsList');
    const navBtns = document.querySelectorAll('.nav-btn');
    const exampleBtns = document.querySelectorAll('.example-btn');

    // State
    let currentSessionId = null;
    let sourceCounter = 0;

    // Step type icons
    const STEP_ICONS = {
        decompose: '🔍',
        search: '🌐',
        analyze: '🧠',
        synthesize: '⚡',
        report: '📄',
    };

    // ===== Navigation =====
    navBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            navBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const view = btn.dataset.view;
            researchInput.classList.add('hidden');
            workflowSection.classList.add('hidden');
            sessionsSection.classList.add('hidden');
            if (view === 'research') {
                if (currentSessionId) {
                    workflowSection.classList.remove('hidden');
                } else {
                    researchInput.classList.remove('hidden');
                }
            } else if (view === 'sessions') {
                sessionsSection.classList.remove('hidden');
                loadSessions();
            }
        });
    });

    // ===== Example Queries =====
    exampleBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            queryInput.value = btn.dataset.query;
            queryInput.focus();
        });
    });

    // ===== Start Research =====
    startResearchBtn.addEventListener('click', startResearch);
    queryInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') startResearch();
    });

    async function startResearch() {
        const query = queryInput.value.trim();
        if (!query || query.length < 3) return;

        startResearchBtn.disabled = true;

        try {
            const response = await fetch('/api/research', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query }),
            });

            if (!response.ok) {
                throw new Error('Failed to start research');
            }

            const session = await response.json();
            currentSessionId = session.id;

            // Switch to workflow view
            showWorkflowView(query);

            // Start streaming events
            streamEvents(session.id);
        } catch (err) {
            console.error('Error starting research:', err);
            addFeedItem('Error starting research. Please try again.', 'progress');
        } finally {
            startResearchBtn.disabled = false;
        }
    }

    function showWorkflowView(query) {
        researchInput.classList.add('hidden');
        sessionsSection.classList.add('hidden');
        workflowSection.classList.remove('hidden');
        workflowQuery.textContent = query;
        stepsContainer.innerHTML = '';
        feedContainer.innerHTML = '';
        sourcesGrid.innerHTML = '';
        sourceCounter = 0;
        sourceCount.textContent = '0';
        reportSection.classList.add('hidden');
        reportContent.innerHTML = '';
    }

    // ===== Back Button =====
    backBtn.addEventListener('click', () => {
        currentSessionId = null;
        workflowSection.classList.add('hidden');
        researchInput.classList.remove('hidden');
        queryInput.value = '';
        queryInput.focus();
    });

    // ===== SSE Event Streaming =====
    function streamEvents(sessionId) {
        const eventSource = new EventSource(`/api/research/${sessionId}/stream`);

        eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                handleEvent(data);

                if (data.type === 'done' || data.type === 'session_complete') {
                    eventSource.close();
                }
            } catch (err) {
                console.error('Error parsing event:', err);
            }
        };

        eventSource.onerror = () => {
            eventSource.close();
            addFeedItem('Connection closed', 'complete');
        };
    }

    // ===== Event Handlers =====
    function handleEvent(data) {
        switch (data.type) {
            case 'session_update':
                addFeedItem(`Research session started`, 'progress');
                break;

            case 'step_update':
                updateStep(data.step);
                if (data.step.status === 'running') {
                    addFeedItem(`Starting: ${data.step.title}`, 'progress');
                } else if (data.step.status === 'completed') {
                    addFeedItem(`✓ Completed: ${data.step.title}`, 'complete');
                }
                break;

            case 'step_progress':
                updateStepProgress(data.step_id, data.progress);
                if (data.message) {
                    addFeedItem(data.message, 'progress');
                }
                break;

            case 'source_found':
                addSource(data.source);
                updateStepProgress(data.step_id, data.progress);
                addFeedItem(`📎 ${data.message}`, 'source');
                break;

            case 'finding':
                addFeedItem(`💡 ${data.finding}`, 'finding');
                updateStepProgress(data.step_id, data.progress);
                break;

            case 'report_ready':
                showReport(data.report);
                addFeedItem('📄 Research report generated!', 'complete');
                break;

            case 'session_complete':
                addFeedItem('✅ Research complete!', 'complete');
                break;

            case 'error':
                addFeedItem(`❌ ${data.message}`, 'progress');
                break;

            case 'done':
                break;
        }
    }

    // ===== Step UI =====
    function updateStep(stepData) {
        let card = document.getElementById(`step-${stepData.id}`);

        if (!card) {
            card = document.createElement('div');
            card.id = `step-${stepData.id}`;
            card.className = 'step-card';
            card.innerHTML = `
                <div class="step-card-header">
                    <div class="step-icon pending">${STEP_ICONS[stepData.step_type] || '⚙️'}</div>
                    <div class="step-info">
                        <div class="step-title">${escapeHtml(stepData.title)}</div>
                        <div class="step-description">${escapeHtml(stepData.description)}</div>
                    </div>
                </div>
                <div class="step-progress-bar">
                    <div class="step-progress-fill" style="width: 0%"></div>
                </div>
                <div class="step-result-area"></div>
            `;
            stepsContainer.appendChild(card);
        }

        // Update status
        card.className = `step-card ${stepData.status}`;
        const icon = card.querySelector('.step-icon');
        icon.className = `step-icon ${stepData.status}`;

        // Update progress
        const progressFill = card.querySelector('.step-progress-fill');
        progressFill.style.width = `${(stepData.progress * 100).toFixed(0)}%`;

        // Show result if completed
        if (stepData.status === 'completed' && stepData.result) {
            const resultArea = card.querySelector('.step-result-area');
            resultArea.innerHTML = `<div class="step-result">✓ ${escapeHtml(stepData.result)}</div>`;
        }

        // Show sub-queries if present
        if (stepData.sub_queries && stepData.sub_queries.length > 0) {
            let subArea = card.querySelector('.step-sub-queries');
            if (!subArea) {
                subArea = document.createElement('div');
                subArea.className = 'step-sub-queries';
                card.appendChild(subArea);
            }
            subArea.innerHTML = stepData.sub_queries
                .map(sq => `<div class="sub-query">→ ${escapeHtml(sq)}</div>`)
                .join('');
        }
    }

    function updateStepProgress(stepId, progress) {
        const card = document.getElementById(`step-${stepId}`);
        if (card) {
            const progressFill = card.querySelector('.step-progress-fill');
            if (progressFill) {
                progressFill.style.width = `${(progress * 100).toFixed(0)}%`;
            }
        }
    }

    // ===== Live Feed =====
    function addFeedItem(message, type) {
        const item = document.createElement('div');
        item.className = 'feed-item';

        const now = new Date();
        const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

        item.innerHTML = `
            <div class="feed-dot ${type}"></div>
            <div class="feed-text">${escapeHtml(message)}</div>
            <div class="feed-time">${timeStr}</div>
        `;

        feedContainer.insertBefore(item, feedContainer.firstChild);

        // Keep only last 50 items
        while (feedContainer.children.length > 50) {
            feedContainer.removeChild(feedContainer.lastChild);
        }
    }

    // ===== Sources =====
    function addSource(source) {
        sourceCounter++;
        sourceCount.textContent = sourceCounter;

        const card = document.createElement('div');
        card.className = 'source-card';
        card.innerHTML = `
            <div class="source-title">${escapeHtml(source.title)}</div>
            <div class="source-url">${escapeHtml(source.url)}</div>
            <div class="source-snippet">${escapeHtml(source.snippet)}</div>
            <span class="source-relevance">${(source.relevance_score * 100).toFixed(0)}% relevant</span>
        `;

        sourcesGrid.appendChild(card);
    }

    // ===== Report =====
    function showReport(markdown) {
        reportSection.classList.remove('hidden');
        reportContent.innerHTML = markdownToHtml(markdown);
    }

    // ===== Sessions History =====
    async function loadSessions() {
        try {
            const response = await fetch('/api/sessions');
            const sessions = await response.json();
            renderSessions(sessions);
        } catch (err) {
            console.error('Error loading sessions:', err);
        }
    }

    function renderSessions(sessions) {
        if (sessions.length === 0) {
            sessionsList.innerHTML = '<p style="color: var(--text-muted); text-align: center; padding: 40px;">No research sessions yet. Start your first research!</p>';
            return;
        }

        sessionsList.innerHTML = sessions.map(s => {
            const date = new Date(s.created_at * 1000);
            const timeStr = date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            return `
                <div class="session-card" data-id="${s.id}">
                    <div class="session-query">${escapeHtml(s.query)}</div>
                    <div class="session-meta">
                        <span class="session-status ${s.status}">${s.status}</span>
                        <span class="session-time">${timeStr}</span>
                    </div>
                </div>
            `;
        }).join('');
    }

    // ===== Utilities =====
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function markdownToHtml(md) {
        return md
            .replace(/^### (.*$)/gm, '<h3>$1</h3>')
            .replace(/^## (.*$)/gm, '<h2>$1</h2>')
            .replace(/^# (.*$)/gm, '<h1>$1</h1>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
            .replace(/^  (\d+)\. (.*)$/gm, '<li>$2</li>')
            .replace(/^  - (.*)$/gm, '<li>$1</li>')
            .replace(/^---$/gm, '<hr>')
            .replace(/\n\n/g, '</p><p>')
            .replace(/^(.+)$/gm, function(match) {
                if (match.startsWith('<')) return match;
                return match;
            });
    }
})();
