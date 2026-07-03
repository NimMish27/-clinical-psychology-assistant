import { useState, useRef, useEffect } from "react";
import { sendChatMessage, type ChatResponse } from "../api/client";

interface Message {
  role: "user" | "assistant";
  text: string;
  sources?: ChatResponse["sources"];
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      text: "Hello! I'm your clinical psychology assistant. Ask me anything about psychological assessments, diagnoses, treatment approaches, or upload documents to the Knowledge Base for me to reference.",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const q = input.trim();
    if (!q || loading) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", text: q }]);
    setLoading(true);

    try {
      const res = await sendChatMessage(q);
      setMessages((m) => [
        ...m,
        { role: "assistant", text: res.answer, sources: res.sources },
      ]);
    } catch (err) {
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          text: `Sorry, I encountered an error: ${err instanceof Error ? err.message : "Unknown error"}`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="chat-container">
      <div className="messages">
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            {msg.role === "assistant" ? (
              <>
                {msg.text.split("\n").map((p, j) => (
                  <p key={j}>{p}</p>
                ))}
                {msg.sources && msg.sources.length > 0 && (
                  <div className="sources">
                    <details>
                      <summary>
                        Sources ({msg.sources.length})
                      </summary>
                      <div className="sources-list">
                        {msg.sources.map((s, k) => (
                          <div key={k} className="source-item">
                            <strong>{s.source}</strong>
                            {s.page ? ` — p. ${s.page}` : ""} &middot; score{" "}
                            {s.score.toFixed(2)}
                            <br />
                            {s.text.slice(0, 200)}
                            {s.text.length > 200 ? "…" : ""}
                          </div>
                        ))}
                      </div>
                    </details>
                  </div>
                )}
              </>
            ) : (
              <p>{msg.text}</p>
            )}
          </div>
        ))}
        {loading && (
          <div className="message typing">
            <div className="typing-dot" />
            <div className="typing-dot" />
            <div className="typing-dot" />
          </div>
        )}
        <div ref={endRef} />
      </div>

      <form className="chat-input-form" onSubmit={handleSubmit}>
        <input
          className="chat-input"
          type="text"
          placeholder="Ask a clinical question…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={loading}
        />
        <button className="send-btn" type="submit" disabled={loading || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  );
}
