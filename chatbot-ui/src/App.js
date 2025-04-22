import React, { useState, useEffect, useCallback, Suspense, lazy } from 'react';
import io from 'socket.io-client';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import './App.css';

// Lazy load LoginForm to improve initial load
const LoginForm = lazy(() => import('./LoginForm'));

const App = () => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [userMessage, setUserMessage] = useState('');
  const [messages, setMessages] = useState([]);
  const [moodData, setMoodData] = useState([]);
  const [chatHistory, setChatHistory] = useState([]);
  const [currentChatId, setCurrentChatId] = useState(null);
  const [isTyping, setIsTyping] = useState(false);
  const [socket, setSocket] = useState(null);

  // Auto-login from token
  useEffect(() => {
    if (localStorage.getItem("token")) {
      setIsAuthenticated(true);
    }
  }, []);

  // Setup WebSocket
  useEffect(() => {
    if (!isAuthenticated) return;

    const newSocket = io('http://localhost:8000', {
      path: '/socket.io/',
      transports: ['websocket', 'polling'],
      auth: { token: localStorage.getItem("token") }
    });

    setSocket(newSocket);

    newSocket.on('connect', fetchMoodHistory);
    newSocket.on('response', (data) => {
      setIsTyping(false);
      setMessages(prev => [...prev, { text: data.message, isUser: false }]);
      fetchMoodHistory();
    });

    return () => newSocket.disconnect();
  }, [isAuthenticated]);

  const fetchMoodHistory = useCallback(async () => {
    try {
      const res = await fetch("http://localhost:8000/mood-history", {
        headers: { Authorization: `Bearer ${localStorage.getItem("token")}` }
      });

      const data = await res.json();
      const moodToScore = { positive: 1, neutral: 0, negative: -1 };

      const transformed = data.moods.map(entry => ({
        ...entry,
        moodScore: moodToScore[entry.mood]
      }));

      setMoodData(transformed);
    } catch (error) {
      console.error("Failed to fetch mood history:", error);
    }
  }, []);

  const handleSend = () => {
    if (!userMessage.trim()) return;

    setMessages(prev => [...prev, { text: userMessage, isUser: true }]);
    setIsTyping(true);

    socket?.emit('chat_message', {
      text: userMessage,
      timestamp: new Date().toISOString(),
      token: localStorage.getItem("token")
    });

    setUserMessage('');
  };

  const handleLogout = () => {
    localStorage.removeItem("token");
    setIsAuthenticated(false);
    socket?.disconnect();
    setMessages([]);
    setMoodData([]);
    setChatHistory([]);
  };

  const startNewChat = () => {
    const newChat = {
      id: Date.now(),
      name: `Chat ${chatHistory.length + 1}`
    };
    setMessages([]);
    setChatHistory(prev => [...prev, newChat]);
    setCurrentChatId(newChat.id);
  };

  const loadChat = (id) => {
    setMessages([{ text: `Loaded chat ${id}`, isUser: false }]);
    setCurrentChatId(id);
  };

  if (!isAuthenticated) {
    return (
      <Suspense fallback={<div>Loading login...</div>}>
        <LoginForm onLogin={() => setIsAuthenticated(true)} />
      </Suspense>
    );
  }

  return (
    <div className="app-container">
      {/* Sidebar */}
      <aside className="sidebar">
        <div>
          <button className="new-chat-button" onClick={startNewChat}>+ New Chat</button>
          <div className="chat-history">
            {chatHistory.length === 0 ? (
              <p style={{ color: '#9ca3af', fontSize: '0.85rem' }}>No past chats</p>
            ) : (
              chatHistory.map(chat => (
                <button key={chat.id} className="chat-history-item" onClick={() => loadChat(chat.id)}>
                  {chat.name}
                </button>
              ))
            )}
          </div>
        </div>
        <button className="logout-button" onClick={handleLogout}>Logout</button>
      </aside>

      {/* Chat Main */}
      <main className="chat-main">
        <header className="header">Mental Health Support</header>

        <div className="message-area">
          {messages.map((msg, i) => (
            <div key={i} className={`message ${msg.isUser ? 'user' : 'bot'}`}>
              {msg.text}
            </div>
          ))}
          {isTyping && <div className="typing-indicator">Bot is typing...</div>}
        </div>

        <div className="input-area">
          <input
            value={userMessage}
            onChange={(e) => setUserMessage(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Type your message..."
          />
          <button onClick={handleSend} disabled={!userMessage.trim()}>Send</button>
        </div>

        <section className="mood-chart">
          <h2>Mood Tracking</h2>
          <p><em>Mood Score: -1 = Negative | 0 = Neutral | 1 = Positive</em></p>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={moodData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="timestamp"
                tickFormatter={(str) =>
                  new Date(str).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              />
              <YAxis domain={[-1, 1]} ticks={[-1, 0, 1]} />
              <Tooltip formatter={(value, name, props) => props.payload.mood} />
              <Legend />
              <Line type="monotone" dataKey="moodScore" stroke="#4F46E5" strokeWidth={2} dot={{ r: 4 }} />
            </LineChart>
          </ResponsiveContainer>

          <div className="mood-list">
            {moodData.map((entry, idx) => (
              <div key={idx}>
                <strong>{new Date(entry.timestamp).toLocaleString()}:</strong> {entry.mood}
              </div>
            ))}
          </div>
        </section>
      </main>
    </div>
  );
};

export default App;
