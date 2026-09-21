import { useState, useRef, useEffect } from "react";
import {
  sendChatMessage,
  type ChatResponse,
} from "../api/client";
import { FRAMEWORK_CARDS, type FrameworkCardData } from "../lib/frameworks";
import { useNav } from "../lib/nav";
import {
  BrainIcon,
  CheckIcon,
  LayersIcon,
  MessageSquareIcon,
  RefreshIcon,
  SparklesIcon,
} from "../ui/icons";

interface Message {
  id: string;
  role: "user" | "assistant";
  text: string;
  caseStudy?: string;
  diagnosis?: string;
  sources?: ChatResponse["sources"];
  // Interactive options for 9 theories
  theoriesOptions?: FrameworkCardData[];
  selectedTheory?: FrameworkCardData;
}

export default function ChatPage() {
  const { openFramework } = useNav();
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "assistant",
      text: "Welcome to the Multi-Theory Clinical Assistant. Please enter the **Client Case Study** and **Diagnosis** below. I will analyze the case across all 9 theoretical frameworks, present tailored evidence and next steps for each theory, and allow you to select a primary theory to proceed with.",
    },
  ]);

  const [caseStudy, setCaseStudy] = useState("");
  const [diagnosis, setDiagnosis] = useState("");
  const [followUp, setFollowUp] = useState("");
  const [loading, setLoading] = useState(false);
  const [includeDocs, setIncludeDocs] = useState(true);

  // Active chosen theory to proceed with across conversation
  const [activeTheory, setActiveTheory] = useState<FrameworkCardData | null>(null);

  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // Handler for initial Case + Diagnosis submission
  async function handleInitialSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!caseStudy.trim() || loading) return;

    const userText = `**Case Study:**\n${caseStudy.trim()}\n\n**Diagnosis:**\n${diagnosis.trim() || "Unspecified / In Evaluation"}`;
    const userMsgId = `user-${Date.now()}`;
    const assistantMsgId = `assistant-${Date.now()}`;

    const currentCase = caseStudy.trim();
    const currentDiag = diagnosis.trim();

    setMessages((m) => [
      ...m,
      {
        id: userMsgId,
        role: "user",
        text: userText,
        caseStudy: currentCase,
        diagnosis: currentDiag,
      },
    ]);

    setCaseStudy("");
    setDiagnosis("");
    setLoading(true);

    try {
      // Prompt query for LLM multi-theory overview
      const rawPrompt = `Analyze this clinical case study and diagnosis across theoretical frameworks:\nCase: ${currentCase}\nDiagnosis: ${currentDiag}`;
      const queryPrompt = rawPrompt.length > 48000 ? rawPrompt.slice(0, 48000) : rawPrompt;
      
      const res = await sendChatMessage(
        queryPrompt,
        5,
        null,
        includeDocs,
      );

      setMessages((m) => [
        ...m,
        {
          id: assistantMsgId,
          role: "assistant",
          text: res.answer || "Here is the multi-theory analysis for your case:",
          sources: res.sources,
          theoriesOptions: FRAMEWORK_CARDS, // All 9 framework cards
        },
      ]);
    } catch (err) {
      setMessages((m) => [
        ...m,
        {
          id: assistantMsgId,
          role: "assistant",
          text: `Error analyzing case: ${err instanceof Error ? err.message : String(err)}`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  // Handler when user selects one of the 9 theories to proceed
  async function handleSelectTheory(msgId: string, theory: FrameworkCardData) {
    setActiveTheory(theory);

    // Update message to reflect selection
    setMessages((prev) =>
      prev.map((m) => (m.id === msgId ? { ...m, selectedTheory: theory } : m))
    );

    const userChoiceMsgId = `user-select-${Date.now()}`;
    const assistantChoiceMsgId = `assistant-select-${Date.now()}`;

    setMessages((prev) => [
      ...prev,
      {
        id: userChoiceMsgId,
        role: "user",
        text: `Proceeding with **${theory.name}** (${theory.tagline}).`,
      },
    ]);

    setLoading(true);

    try {
      const apiFramework = theory.id === "cbt" ? "cbt" : theory.id === "act" ? "act" : theory.id === "humanistic" ? "humanistic" : null;
      const res = await sendChatMessage(
        `Focusing strictly on ${theory.name} (${theory.tagline}), provide a detailed session plan, concrete interventions, and homework assignments for this case.`,
        5,
        apiFramework,
        includeDocs
      );

      setMessages((prev) => [
        ...prev,
        {
          id: assistantChoiceMsgId,
          role: "assistant",
          text: res.answer,
          sources: res.sources,
          selectedTheory: theory,
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: assistantChoiceMsgId,
          role: "assistant",
          text: `Selected ${theory.name}. You can now ask follow-up questions tailored to this framework!`,
          selectedTheory: theory,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  // Handler for ongoing follow-up questions under chosen theory
  async function handleFollowUpSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!followUp.trim() || loading) return;

    const q = followUp.trim();
    setFollowUp("");

    const userMsgId = `user-fup-${Date.now()}`;
    const assistantMsgId = `assistant-fup-${Date.now()}`;

    setMessages((m) => [...m, { id: userMsgId, role: "user", text: q }]);
    setLoading(true);

    try {
      const apiFw = activeTheory?.id === "cbt" ? "cbt" : activeTheory?.id === "act" ? "act" : activeTheory?.id === "humanistic" ? "humanistic" : null;
      const res = await sendChatMessage(q, 5, apiFw, includeDocs);

      setMessages((m) => [
        ...m,
        {
          id: assistantMsgId,
          role: "assistant",
          text: res.answer,
          sources: res.sources,
          selectedTheory: activeTheory ?? undefined,
        },
      ]);
    } catch (err) {
      setMessages((m) => [
        ...m,
        {
          id: assistantMsgId,
          role: "assistant",
          text: `Error: ${err instanceof Error ? err.message : String(err)}`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="relative min-h-[calc(100vh-60px)] pb-12">
      {/* Top Header */}
      <header className="sticky top-0 z-30 flex flex-wrap items-center justify-between gap-4 border-b border-stone-200/80 bg-white/80 px-8 py-4 backdrop-blur-xl">
        <div className="flex items-center gap-3">
          <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-600 text-white shadow-md">
            <MessageSquareIcon className="h-5 w-5" />
          </span>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-stone-900">Clinical Assistant Chat</h1>
            <p className="text-xs text-stone-500">Multi-Theory Analysis · 9 Theoretical Frameworks · RAG Grounded</p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          {activeTheory && (
            <div className="flex items-center gap-2 rounded-full border border-stone-200 bg-stone-50 px-3.5 py-1.5 text-xs font-semibold text-stone-700">
              <span className="h-2 w-2 rounded-full" style={{ background: activeTheory.color }} />
              Active Theory: {activeTheory.name}
            </div>
          )}
          <label className="flex items-center gap-2 text-xs font-medium text-stone-600">
            <input
              type="checkbox"
              checked={includeDocs}
              onChange={(e) => setIncludeDocs(e.target.checked)}
              className="rounded border-stone-300 text-indigo-600 focus:ring-indigo-500"
            />
            Include My Documents
          </label>
        </div>
      </header>

      {/* Main Content Layout with Chat Stream & Right Sidebar */}
      <div className="mx-auto max-w-7xl px-4 py-8 flex gap-8">
        {/* Messages Stream */}
        <div className="flex-1 flex flex-col gap-8">
          {messages.map((msg) => (
            <div key={msg.id} className={`flex flex-col gap-3 ${msg.role === "user" ? "items-end" : "items-start"}`}>
              
              {/* Avatar / Sender */}
              <div className="flex items-center gap-2">
                <span
                  className={`flex h-7 w-7 items-center justify-center rounded-lg text-xs font-bold text-white shadow-sm ${
                    msg.role === "user" ? "bg-stone-800" : "bg-indigo-600"
                  }`}
                >
                  {msg.role === "user" ? "Dr" : <BrainIcon className="h-4 w-4" />}
                </span>
                <span className="text-xs font-semibold text-stone-500">
                  {msg.role === "user" ? "Clinician" : "Clinical Assistant"}
                </span>
              </div>

              {/* Bubble */}
              <div
                className={`max-w-3xl rounded-3xl p-6 text-sm leading-relaxed shadow-sm ${
                  msg.role === "user"
                    ? "bg-stone-900 text-white"
                    : "border border-white/80 bg-white/90 text-stone-800 shadow-md backdrop-blur-xl"
                }`}
              >
                {msg.text.split("\n\n").map((para, i) => (
                  <p key={i} className={i > 0 ? "mt-3" : ""}>
                    {para}
                  </p>
                ))}

                {/* RAG Sources list */}
                {msg.sources && msg.sources.length > 0 && (
                  <div className="mt-4 border-t border-stone-100 pt-3">
                    <details className="text-xs text-stone-500">
                      <summary className="cursor-pointer font-semibold text-indigo-600 hover:underline">
                        View Grounded Literature Sources ({msg.sources.length})
                      </summary>
                      <div className="mt-2 flex flex-col gap-2">
                        {msg.sources.map((s, idx) => (
                          <div key={idx} className="rounded-lg bg-stone-50 p-2.5 ring-1 ring-stone-200/60">
                            <span className="font-bold text-stone-700">{s.source}</span> (p. {s.page}) · Score: {(s.score * 100).toFixed(0)}%
                            <p className="mt-1 line-clamp-2 italic text-stone-600">"{s.text}"</p>
                          </div>
                        ))}
                      </div>
                    </details>
                  </div>
                )}
              </div>

              {/* Case Presentation Banner (Matching Stitch e4ac1e5583464f018109e114d79ed233) */}
              <div className="w-full rounded-2xl border border-stone-800 bg-[#0f172a] p-6 text-stone-100 shadow-xl">
                <div className="flex items-center justify-between border-b border-stone-800 pb-3">
                  <div className="flex items-center gap-2">
                    <span className="rounded-md bg-indigo-600 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider text-white">
                      Case Presentation
                    </span>
                    <h3 className="text-sm font-bold text-white">Active Case Analysis</h3>
                  </div>
                  <span className="text-xs text-stone-400">Intake Dossier · HIPAA Enforced</span>
                </div>

                <div className="mt-4 flex flex-col gap-3 text-xs leading-relaxed text-stone-300">
                  <div>
                    <strong className="text-stone-100 font-semibold">Chief Complaint & Sleep Profile:</strong> "Reports feeling persistently fatigued with night appetite shifts and bedtime rumination."
                  </div>
                  <div>
                    <strong className="text-stone-100 font-semibold">Psychosocial Stressors:</strong> Recent romantic relationship breakup, social isolation out of fear of appearing weak.
                  </div>
                  <div className="rounded-xl border border-amber-900/60 bg-amber-950/30 p-3 text-amber-200">
                    <strong className="font-semibold text-amber-100">Safety Protocol Level 2 (Passive):</strong> Passive suicidal ideation statement noted ("easier if I could disappear"). No active intent or plan.
                  </div>
                </div>
              </div>

              {/* 9 Theoretical Framework Options Cards Grid */}
              {msg.theoriesOptions && (
                <div className="w-full mt-4 flex flex-col gap-4 rounded-3xl border border-stone-200 bg-stone-50/60 p-6 backdrop-blur-xl">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-base font-bold text-stone-900">Choose a Theoretical Framework to Proceed</h3>
                      <p className="text-xs text-stone-500">
                        Below are tailored evidence summaries and next steps for all 9 core theories. Click any theory to proceed with its targeted formulation.
                      </p>
                    </div>
                    {msg.selectedTheory && (
                      <span className="rounded-full bg-emerald-100 px-3 py-1 text-xs font-bold text-emerald-800">
                        Selected: {msg.selectedTheory.name}
                      </span>
                    )}
                  </div>

                  <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
                    {msg.theoriesOptions.map((fw) => {
                      const isSelected = msg.selectedTheory?.id === fw.id;
                      return (
                        <div
                          key={fw.id}
                          className={`group relative flex flex-col justify-between rounded-2xl border bg-white p-5 shadow-sm transition-all duration-300 hover:-translate-y-1 hover:shadow-xl ${
                            isSelected ? "ring-2 ring-indigo-600 border-indigo-600" : "border-stone-200/80"
                          }`}
                        >
                          <div>
                            <div className="flex items-center justify-between">
                              <span
                                className="flex h-8 w-8 items-center justify-center rounded-xl text-xs font-bold text-white shadow-sm"
                                style={{ background: fw.color }}
                              >
                                {fw.name.slice(0, 2).toUpperCase()}
                              </span>
                              <span className="rounded-full bg-stone-100 px-2.5 py-0.5 text-[10px] font-bold text-stone-600">
                                {Math.round(fw.confidence * 100)}% Match
                              </span>
                            </div>

                            <h4 className="mt-3 text-base font-bold text-stone-900">{fw.name}</h4>
                            <p className="text-xs text-stone-500 font-medium">{fw.tagline}</p>

                            <p className="mt-3 text-xs leading-relaxed text-stone-600 line-clamp-3">
                              {fw.description}
                            </p>

                            {/* Key Interventions preview */}
                            <div className="mt-4 flex flex-col gap-1.5">
                              <span className="text-[10px] font-bold uppercase tracking-wider text-stone-400">
                                Next Steps / Interventions
                              </span>
                              {fw.workspace.suggestedInterventions.slice(0, 2).map((inv, idx) => (
                                <div key={idx} className="flex items-center gap-1.5 text-xs text-stone-700">
                                  <span className="h-1.5 w-1.5 rounded-full" style={{ background: fw.color }} />
                                  <span className="font-semibold">{inv.name}</span>
                                </div>
                              ))}
                            </div>
                          </div>

                          <div className="mt-5 flex gap-2">
                            <button
                              onClick={() => handleSelectTheory(msg.id, fw)}
                              disabled={loading}
                              className="flex-1 inline-flex items-center justify-center gap-1.5 rounded-xl px-3 py-2 text-xs font-bold text-white shadow-md transition hover:opacity-90 disabled:opacity-50"
                              style={{ background: fw.color }}
                            >
                              <CheckIcon className="h-3.5 w-3.5" />
                              {isSelected ? "Selected" : "Select & Proceed"}
                            </button>
                            <button
                              onClick={() => openFramework(fw.id)}
                              className="inline-flex items-center justify-center rounded-xl border border-stone-200 bg-stone-50 px-2.5 py-2 text-xs font-semibold text-stone-600 transition hover:bg-stone-100"
                              title="View full framework workspace"
                            >
                              <LayersIcon className="h-3.5 w-3.5" />
                            </button>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          ))}

          {loading && (
            <div className="flex items-center gap-3 text-sm font-semibold text-stone-500">
              <span className="flex h-8 w-8 items-center justify-center rounded-xl bg-indigo-50 text-indigo-600">
                <RefreshIcon className="h-4 w-4 animate-spin" />
              </span>
              Analyzing case across 9 theoretical frameworks & literature…
            </div>
          )}

          <div ref={endRef} />
        </div>

        {/* Right Sidebar — Patient Case Context (Stitch e4ac1e5583464f018109e114d79ed233) */}
        <aside className="w-80 shrink-0 hidden lg:flex flex-col gap-6">
          <div className="rounded-2xl border border-stone-200 bg-white p-5 shadow-sm">
            <div className="flex items-center justify-between border-b border-stone-100 pb-3">
              <h4 className="text-xs font-bold uppercase tracking-wider text-stone-500">Patient Case Context</h4>
              <span className="rounded-full bg-emerald-50 px-2 py-0.5 text-[10px] font-bold text-emerald-600">Synced</span>
            </div>

            <div className="mt-4 flex flex-col gap-3">
              <div>
                <h5 className="text-sm font-bold text-stone-900">Aarav Patel</h5>
                <p className="text-xs text-stone-400">28 yrs · Tech / Software Engineer</p>
              </div>

              <div className="grid grid-cols-2 gap-2 text-xs border-t border-stone-100 pt-3">
                <span className="text-stone-400">Presentation:</span>
                <span className="font-semibold text-stone-700 text-right">Intake / Evaluation</span>
                <span className="text-stone-400">Prior Psych Therapy:</span>
                <span className="font-semibold text-stone-700 text-right">None reported</span>
                <span className="text-stone-400">Family History:</span>
                <span className="font-semibold text-stone-700 text-right">Maternal Depression</span>
              </div>
            </div>
          </div>

          <div className="rounded-2xl border border-stone-200 bg-white p-5 shadow-sm">
            <h4 className="text-xs font-bold uppercase tracking-wider text-stone-500 border-b border-stone-100 pb-3">Active Theoretical Dial</h4>
            <div className="mt-4 flex flex-col gap-3 text-xs">
              <div className="flex items-center justify-between">
                <span className="font-semibold text-indigo-600">CBT (Cognitive Restructuring)</span>
                <span className="font-bold text-stone-700">45%</span>
              </div>
              <div className="h-1.5 w-full rounded-full bg-stone-100 overflow-hidden">
                <div className="h-full bg-indigo-600" style={{ width: "45%" }} />
              </div>

              <div className="flex items-center justify-between mt-2">
                <span className="font-semibold text-sky-600">ACT (Acceptance & Defusion)</span>
                <span className="font-bold text-stone-700">30%</span>
              </div>
              <div className="h-1.5 w-full rounded-full bg-stone-100 overflow-hidden">
                <div className="h-full bg-sky-600" style={{ width: "30%" }} />
              </div>

              <div className="flex items-center justify-between mt-2">
                <span className="font-semibold text-purple-600">Psychodynamic / Attachment</span>
                <span className="font-bold text-stone-700">25%</span>
              </div>
              <div className="h-1.5 w-full rounded-full bg-stone-100 overflow-hidden">
                <div className="h-full bg-purple-600" style={{ width: "25%" }} />
              </div>
            </div>
          </div>
        </aside>
      </div>

      {/* Input Box Bottom Bar */}
      <div className="fixed bottom-0 left-0 right-0 z-20 border-t border-stone-200/80 bg-white/80 p-4 backdrop-blur-xl sm:left-[280px]">
        <div className="mx-auto max-w-4xl">
          {messages.length <= 1 ? (
            /* Initial Form: Case Study + Diagnosis */
            <form onSubmit={handleInitialSubmit} className="flex flex-col gap-3 rounded-2xl border border-stone-200 bg-white p-4 shadow-lg">
              <div className="flex items-center justify-between text-xs font-bold uppercase tracking-wider text-stone-500">
                <span>Enter Case Details</span>
                <span>Stitch Multi-Theory Clinical Assistant</span>
              </div>
              <textarea
                value={caseStudy}
                onChange={(e) => setCaseStudy(e.target.value)}
                placeholder="Enter client case study (e.g. 32-year-old presenting with severe panic attacks, avoidance of social gatherings, and rumination)..."
                rows={3}
                className="w-full resize-none rounded-xl border border-stone-200 p-3 text-sm text-stone-800 focus:border-indigo-500 focus:outline-none"
              />
              <div className="flex gap-3">
                <input
                  type="text"
                  value={diagnosis}
                  onChange={(e) => setDiagnosis(e.target.value)}
                  placeholder="Diagnosis (e.g. Panic Disorder with Agoraphobia)"
                  className="flex-1 rounded-xl border border-stone-200 px-3.5 py-2.5 text-sm text-stone-800 focus:border-indigo-500 focus:outline-none"
                />
                <button
                  type="submit"
                  disabled={loading || !caseStudy.trim()}
                  className="inline-flex items-center justify-center gap-2 rounded-xl bg-indigo-600 px-6 py-2.5 text-sm font-bold text-white shadow-md transition hover:bg-indigo-700 disabled:opacity-50"
                >
                  <SparklesIcon className="h-4 w-4" /> Synthesize 9 Theories
                </button>
              </div>
            </form>
          ) : (
            /* Follow-up Form under chosen theory */
            <form onSubmit={handleFollowUpSubmit} className="flex gap-3">
              <input
                type="text"
                value={followUp}
                onChange={(e) => setFollowUp(e.target.value)}
                placeholder={
                  activeTheory
                    ? `Ask a question or request session notes under ${activeTheory.name}...`
                    : "Ask follow-up question or enter a new case..."
                }
                className="flex-1 rounded-2xl border border-stone-200 bg-white px-5 py-3.5 text-sm text-stone-800 shadow-sm focus:border-indigo-500 focus:outline-none"
              />
              <button
                type="submit"
                disabled={loading || !followUp.trim()}
                className="inline-flex items-center justify-center gap-2 rounded-2xl bg-stone-900 px-6 py-3.5 text-sm font-bold text-white shadow-md transition hover:bg-stone-800 disabled:opacity-50"
              >
                Synthesize →
              </button>
            </form>
          )}
        </div>
      </div>
    </div>
  );
}
