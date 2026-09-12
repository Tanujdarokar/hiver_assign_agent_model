async function handleChatSubmit(event) {
    event.preventDefault();
    const input = document.getElementById('userInput');
    const message = input.value.trim();
    if (!message) return;

    input.value = '';
    hideWelcomeCard();
    appendUserMessage(message);
    showTypingIndicator();

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message })
        });
        const data = await response.json();
        removeTypingIndicator();
        appendAgentResponse(data);
    } catch (error) {
        removeTypingIndicator();
        appendErrorMessage("Failed to connect to agent backend. Please ensure app.py server is running.");
    }
}

function sendSample(text) {
    document.getElementById('userInput').value = text;
    document.getElementById('chatForm').dispatchEvent(new Event('submit'));
}

function hideWelcomeCard() {
    const welcome = document.getElementById('welcomeCard');
    if (welcome) welcome.style.display = 'none';
}

function clearFeed() {
    const feed = document.getElementById('chatFeed');
    feed.innerHTML = `
        <div class="welcome-card" id="welcomeCard">
            <div class="welcome-badge"> Official Support Test Bench</div>
            <h2>Welcome to AppleSupport AI Agent</h2>
            <p>Enter any customer service inquiry below or pick a preset scenario from the sidebar to test intent classification confidence, risk escalation boundaries, vector similarity retrieval, and dynamic grounded responses.</p>
            <div class="preset-chips">
                <span class="chip" onclick="sendSample('How do I update to iOS 11 safely?')">📲 How to update iOS?</span>
                <span class="chip" onclick="sendSample('My WiFi keeps disconnecting on my iPad')">📶 WiFi dropping</span>
                <span class="chip" onclick="sendSample('How do I buy more iCloud storage?')">☁️ iCloud storage full</span>
            </div>
        </div>
    `;
}

function appendUserMessage(text) {
    const feed = document.getElementById('chatFeed');
    const container = document.createElement('div');
    container.className = 'user-query-container';
    container.innerHTML = `<div class="user-query-bubble">${escapeHtml(text)}</div>`;
    feed.appendChild(container);
    feed.scrollTop = feed.scrollHeight;
}

function showTypingIndicator() {
    const feed = document.getElementById('chatFeed');
    const indicator = document.createElement('div');
    indicator.id = 'typingIndicator';
    indicator.className = 'agent-card';
    indicator.innerHTML = '<div class="reply-text-box"> AppleSupport AI is processing query & retrieving historical evidence...</div>';
    feed.appendChild(indicator);
    feed.scrollTop = feed.scrollHeight;
}

function removeTypingIndicator() {
    const el = document.getElementById('typingIndicator');
    if (el) el.remove();
}

function appendAgentResponse(data) {
    const feed = document.getElementById('chatFeed');
    const card = document.createElement('div');
    card.className = 'agent-card';

    const confidencePct = (data.confidence * 100).toFixed(1);
    const decisionClass = data.decision === 'AUTO' ? 'AUTO' : 'ESCALATE';

    let evidenceHtml = '';
    if (data.retrieved_examples && data.retrieved_examples.length > 0) {
        evidenceHtml = `
            <div class="evidence-box">
                <h4>Grounding Evidence (Top Historical Resolution Cases)</h4>
                <div class="evidence-list">
                    ${data.retrieved_examples.map(e => `
                        <div class="evidence-item">
                            <div class="evidence-top">
                                <span>Conv ID: ${e.conversation_id}</span>
                                <span class="similarity-val">Similarity: ${(e.similarity_score * 100).toFixed(1)}%</span>
                            </div>
                            <div class="evidence-cust"><strong>Customer:</strong> ${escapeHtml(e.customer_message)}</div>
                            <div class="evidence-sup"><strong>Agent:</strong> ${escapeHtml(e.support_response)}</div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    card.innerHTML = `
        <div class="card-top-bar">
            <div class="intent-group">
                <span class="intent-pill">${data.intent}</span>
                <span class="conf-pill">Confidence: ${confidencePct}%</span>
            </div>
            <span class="decision-badge ${decisionClass}">${data.decision}</span>
        </div>
        <div class="reply-text-box">${escapeHtml(data.reply)}</div>
        <div class="reason-callout ${decisionClass}">
            <strong>Decision Reason:</strong> ${escapeHtml(data.reason)}
        </div>
        ${evidenceHtml}
    `;

    feed.appendChild(card);
    feed.scrollTop = feed.scrollHeight;
}

function appendErrorMessage(msg) {
    const feed = document.getElementById('chatFeed');
    const err = document.createElement('div');
    err.className = 'agent-card';
    err.innerHTML = `<div class="reason-callout ESCALATE">${escapeHtml(msg)}</div>`;
    feed.appendChild(err);
    feed.scrollTop = feed.scrollHeight;
}

function escapeHtml(text) {
    if (!text) return '';
    return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
