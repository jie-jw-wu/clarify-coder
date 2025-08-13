const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');
const chatMessages = document.getElementById('chatMessages');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const saveButton = document.getElementById('saveButton');

let chatHistory = [];

function addMessage(content, sender, isUser = false) {
    // Add to chat history
    chatHistory.push({
        content: content,
        sender: sender,
        isUser: isUser,
        timestamp: new Date().toISOString()
    });
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'llm-message'}`;
    
    const senderDiv = document.createElement('div');
    senderDiv.className = 'message-sender';
    senderDiv.textContent = sender;
    
    const contentDiv = document.createElement('div');
    
    if (isUser) {
        // For user messages, just display as plain text
        contentDiv.textContent = content;
    } else {
        // For LLM messages, render markdown
        contentDiv.innerHTML = marked.parse(content);
    }
    
    messageDiv.appendChild(senderDiv);
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function updateStatus(isOnline) {
    if (isOnline) {
        statusDot.classList.remove('offline');
        statusText.textContent = 'Online';
    } else {
        statusDot.classList.add('offline');
        statusText.textContent = 'Offline';
    }
}

function saveChat() {
    let markdown = '# ClarifyCoder Chat Session\n\n';
    markdown += `*Saved on: ${new Date().toLocaleString()}*\n\n`;
    
    chatHistory.forEach(msg => {
        if (msg.sender === 'ClarifyCoder' && msg.content === 'Hello! I\'m ready to help you with your coding questions. What would you like to know?') {
            return; // Skip initial message
        }
        
        markdown += `## ${msg.sender}\n\n`;
        markdown += `${msg.content}\n\n`;
        markdown += '---\n\n';
    });
    
    const blob = new Blob([markdown], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `clarify-coder-chat-${new Date().toISOString().split('T')[0]}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Check server status periodically
function checkServerStatus() {
    fetch('/send_message', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: 'ping' })
    })
    .then(response => {
        updateStatus(response.ok);
    })
    .catch(() => {
        updateStatus(false);
    });
}

function sendMessage() {
    const message = messageInput.value.trim();
    if (!message) return;

    // Add user message to chat
    addMessage(message, 'You', true);
    
    // Clear input and disable button
    messageInput.value = '';
    sendButton.disabled = true;
    sendButton.textContent = 'Sending...';
    
    // Add loading message
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message llm-message loading';
    loadingDiv.innerHTML = '<div class="message-sender">ClarifyCoder</div><div>Thinking...</div>';
    chatMessages.appendChild(loadingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    // Send message to server
    fetch('/send_message', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: message })
    })
    .then(response => response.json())
    .then(data => {
        // Remove loading message
        chatMessages.removeChild(loadingDiv);
        
        if (data.error) {
            addMessage('Error: ' + data.error, 'System');
        } else {
            addMessage(data.llm_response, data.llm_name);
        }
    })
    .catch(error => {
        // Remove loading message
        chatMessages.removeChild(loadingDiv);
        addMessage('Error: Failed to send message', 'System');
    })
    .finally(() => {
        sendButton.disabled = false;
        sendButton.textContent = 'Send';
        messageInput.focus();
    });
}

// Event listeners
sendButton.addEventListener('click', sendMessage);
saveButton.addEventListener('click', saveChat);

messageInput.addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

// Check server status every 30 seconds
setInterval(checkServerStatus, 30000);

// Initial status check
checkServerStatus();

// Focus on input when page loads
messageInput.focus();