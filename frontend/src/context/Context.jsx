import { createContext, useState, useRef, useEffect } from "react";
import runChat from "../config/Gemini";
import {
	addChatMessage,
	createChat,
	getChat,
	getChatResults,
	listChats,
	deleteChat
} from "../services/chat";

export const Context = createContext();

const EMPTY_RESULTS_TABLE = { columns: [], rows: [] };

const ContextProvider = (props) => {
	const [input, setInput] = useState("");
	const [recentPrompt, setRecentPrompt] = useState("");
	const [prevPrompts, setPrevPrompts] = useState([]);
	const [showResults, setShowResults] = useState(false);
	const [loading, setLoading] = useState(false);
	const [resultData, setResultData] = useState("");
	const [agentData, setAgentData] = useState("");
	const [graphData, setGraphData] = useState();
	const [socket, setSocket] = useState(null);
	const [evenData, setEvenData] = useState();
	const [downloadData, setDownloadData] = useState(false);
	const [chatNo, setChatNo] = useState(0);
	const displayedCharsRef = useRef(0); // Use a ref to track displayed characters count
	const totalCharsRef = useRef(0); // Use a ref for total characters
	const [fileHistory, setFileHistory] = useState([]);		// State to store the file history
	const [uploadNotice, setUploadNotice] = useState(null);
	const [totalDisplayedCharsRef, setTotalDisplayedCharsRef] = useState(0); // State to track total displayed chars
	const [prevResults, setPrevResults] = useState([]);
	const pendingDataRef = useRef([]);
	const resp = useRef(false);
	const renderTokenRef = useRef(0);
	const uploadNoticeTimerRef = useRef(null);
	const [resultsTablesByThread, setResultsTablesByThread] = useState({});
	const [resultsUpdatedAtByThread, setResultsUpdatedAtByThread] = useState({});
	const [threads, setThreads] = useState([]);
	const [activeThreadId, setActiveThreadId] = useState(null);
	const [chatMessages, setChatMessages] = useState([]);
	const [chatHydrated, setChatHydrated] = useState(false);



	// Reveal text incrementally using a single requestAnimationFrame loop.
	// charsPerFrame is scaled so the animation completes in ~2 s regardless of length.
	const _animateText = (text, setter, onDone) => {
		const currentToken = renderTokenRef.current;
		const totalLen = text.length;
		totalCharsRef.current = totalLen;
		displayedCharsRef.current = 0;

		if (totalLen === 0) {
			onDone && onDone();
			return;
		}

		// Target ~2 s at 60 fps; minimum 4 chars/frame so short responses are instant.
		const TARGET_FRAMES = 120; // 2 s × 60 fps
		const charsPerFrame = Math.max(4, Math.ceil(totalLen / TARGET_FRAMES));

		let pos = 0;
		const tick = () => {
			if (renderTokenRef.current !== currentToken) return; // render cancelled
			pos = Math.min(pos + charsPerFrame, totalLen);
			displayedCharsRef.current = pos;
			setter(text.slice(0, pos));
			if (pos < totalLen) {
				requestAnimationFrame(tick);
			} else {
				onDone && onDone();
			}
		};
		requestAnimationFrame(tick);
	};

	// Function to reset the states for a new chat
	const newChat = () => {
		setLoading(false);
		setShowResults(false);
		setDownloadData(false);
		setChatNo(0);
		displayedCharsRef.current = 0; // Reset ref for displayed characters
		setResultData("");
		setAgentData("");
		setChatMessages([]);
		setActiveThreadId(null);
		setRecentPrompt("");
	};

	const getThreadKey = (threadId) => {
		if (threadId) {
			return String(threadId);
		}
		return activeThreadId ? String(activeThreadId) : "global";
	};

	const resultsTable = resultsTablesByThread[getThreadKey()] || EMPTY_RESULTS_TABLE;
	const resultsUpdatedAt = resultsUpdatedAtByThread[getThreadKey()] || null;

	const setResultsTable = (table, threadId = null) => {
		const key = getThreadKey(threadId);
		setResultsTablesByThread((prev) => ({
			...prev,
			[key]: table,
		}));
	};

	const setResultsUpdatedAt = (timestamp, threadId = null) => {
		const key = getThreadKey(threadId);
		setResultsUpdatedAtByThread((prev) => ({
			...prev,
			[key]: timestamp,
		}));
	};

	const startRenderCycle = () => {
		renderTokenRef.current += 1;
		displayedCharsRef.current = 0;
		totalCharsRef.current = 0;
		pendingDataRef.current = [];
		return renderTokenRef.current;
	};

	const cancelRenderCycle = () => {
		renderTokenRef.current += 1;
		displayedCharsRef.current = 0;
		totalCharsRef.current = 0;
		pendingDataRef.current = [];
	};

	const loadThreadResults = async (threadId) => {
		if (!threadId) {
			return;
		}
		try {
			const data = await getChatResults(threadId);
			const columns = Array.isArray(data?.columns) ? data.columns.filter(Boolean) : [];
			const rows = Array.isArray(data?.rows) ? data.rows : [];
			setResultsTable({ columns, rows }, threadId);
			if (Object.prototype.hasOwnProperty.call(data || {}, "updated_at")) {
				setResultsUpdatedAt(data.updated_at, threadId);
			}
		} catch (error) {
			console.error("Failed to load results table:", error);
		}
	};

	const selectThread = async (threadId, { persist = true } = {}) => {
		try {
			const data = await getChat(threadId);
			setActiveThreadId(data.thread.id);
			setChatMessages(data.messages || []);
			setShowResults((data.messages || []).length > 0);
			setDownloadData(false);
			setResultData("");
			setAgentData("");
			await loadThreadResults(data.thread.id);
			if (persist) {
				localStorage.setItem("ui3gpp_active_thread", String(data.thread.id));
			}
		} catch (error) {
			console.error("Failed to load chat thread:", error);
		}
	};

	const refreshThreads = async (selectedId) => {
		try {
			const forceNewChat = sessionStorage.getItem("ui3gpp_force_new_chat") === "1";
			const data = await listChats();
			const nextThreads = data.threads || [];
			setThreads(nextThreads);
			if (forceNewChat) {
				sessionStorage.removeItem("ui3gpp_force_new_chat");
				localStorage.removeItem("ui3gpp_active_thread");
				newChat();
				setChatHydrated(true);
				return;
			}
			if (selectedId) {
				const exists = nextThreads.find((thread) => String(thread.id) === String(selectedId));
				if (exists) {
					await selectThread(exists.id, { persist: true });
				}
			}
			if (!selectedId && nextThreads.length > 0) {
				const fallbackId = localStorage.getItem("ui3gpp_active_thread");
				if (fallbackId && nextThreads.find((t) => String(t.id) === String(fallbackId))) {
					await selectThread(fallbackId, { persist: false });
				} else {
					await selectThread(nextThreads[0].id, { persist: true });
				}
			}
			setChatHydrated(true);
		} catch (error) {
			console.error("Failed to load chats:", error);
			setChatHydrated(true);
		}
	};

	const createThread = async (title) => {
		try {
			const data = await createChat(title);
			const newThread = {
				id: data.id,
				title: data.title,
				last_message: "",
				created_at: data.created_at,
				updated_at: data.updated_at,
			};
			setThreads((prev) => [newThread, ...prev]);
			setActiveThreadId(data.id);
			setChatMessages([]);
			setShowResults(false);
			localStorage.setItem("ui3gpp_active_thread", String(data.id));
			return data.id;
		} catch (error) {
			console.error("Failed to create chat:", error);
			return null;
		}
	};

	const deleteThread = async (threadId) => {
		if (!threadId) {
			return;
		}
		try {
			await deleteChat(threadId);
			setThreads((prev) =>
				prev.filter((thread) => String(thread.id) !== String(threadId))
			);
			setResultsTablesByThread((prev) => {
				const next = { ...prev };
				delete next[String(threadId)];
				return next;
			});
			setResultsUpdatedAtByThread((prev) => {
				const next = { ...prev };
				delete next[String(threadId)];
				return next;
			});
			if (String(activeThreadId) === String(threadId)) {
				localStorage.removeItem("ui3gpp_active_thread");
				newChat();
			}
			if (socket && socket.readyState === WebSocket.OPEN) {
				socket.send(JSON.stringify({ type: "delete_thread", thread_id: threadId }));
			}
		} catch (error) {
			console.error("Failed to delete chat:", error);
		}
	};

	const appendMessage = (threadId, message) => {
		setChatMessages((prev) =>
			!activeThreadId || String(activeThreadId) === String(threadId)
				? [...prev, message]
				: prev
		);
		setThreads((prev) => {
			const updated = prev.map((thread) =>
				String(thread.id) === String(threadId)
					? {
						...thread,
						last_message: message.content.slice(0, 200),
						updated_at: message.created_at || new Date().toISOString(),
					}
					: thread
			);
			return [...updated].sort(
				(a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime()
			);
		});
	};

	const persistMessage = async (threadId, message) => {
		try {
			await addChatMessage(threadId, message);
		} catch (error) {
			console.error("Failed to save message:", error);
		}
	};

	useEffect(() => {
		refreshThreads();
	}, []);

	useEffect(() => {
		return () => {
			if (uploadNoticeTimerRef.current) {
				clearTimeout(uploadNoticeTimerRef.current);
			}
		};
	}, []);

	const pushUploadNotice = (notice) => {
		if (uploadNoticeTimerRef.current) {
			clearTimeout(uploadNoticeTimerRef.current);
		}
		if (!notice) {
			setUploadNotice(null);
			return;
		}
		setUploadNotice({ id: Date.now(), ...notice });
		uploadNoticeTimerRef.current = setTimeout(() => {
			setUploadNotice(null);
		}, 2600);
	};

	const renderBatch = () => {
		const newChunk = pendingDataRef.current.shift();
		if (!newChunk) return;
		// Capture prefix length before this chunk so we can animate only the new portion
		setAgentData((current) => {
			const base = current;
			const currentToken = renderTokenRef.current;
			const charsPerFrame = Math.max(4, Math.ceil(newChunk.length / 120));
			let pos = 0;
			const tick = () => {
				if (renderTokenRef.current !== currentToken) return;
				pos = Math.min(pos + charsPerFrame, newChunk.length);
				setAgentData(base + newChunk.slice(0, pos));
				if (pos < newChunk.length) {
					requestAnimationFrame(tick);
				} else if (pendingDataRef.current.length > 0) {
					renderBatch();
				}
			};
			requestAnimationFrame(tick);
			return base; // initial render; RAF will update it
		});
	};

	// Function to handle sending the prompt
	const onSent = async (prompt, isAgent) => {
		setResultData("");
		setLoading(true);
		setShowResults(true);
		startRenderCycle();
		let response;
		if (isAgent === undefined) {
			if (prompt !== undefined) {
				response = await runChat(prompt);
				setRecentPrompt(prompt);
			} else {
				setPrevPrompts((prev) => [...prev, input]);
				setRecentPrompt(input);
				response = await runChat(input);
			}
		}

		try {
			const text = String(response || "");
			_animateText(text, setResultData, () => {
				setDownloadData(true);
				resp.current = false;
			});
		} catch (error) {
			console.error("Error while running chat:", error);
		} finally {
			setLoading(false);
			setInput("");
		}
	};

	// Animate the full response text using RAF — no thousands of timers.
	// Passes raw markdown directly; ReactMarkdown renders it correctly.
	const onRender = (data) => {
		setResultData("");
		setLoading(false);
		setInput("");
		const text = String(data || "");
		startRenderCycle();
		_animateText(
			text,
			setResultData,
			() => {
				setDownloadData(true);
				resp.current = false;
			}
		);
	};

	const onRenderAgent = (data) => {
		setLoading(false);
		setInput("");
		const chunk = String(data || "");
		if (!chunk) return;
		// Queue chunk and trigger batch render
		pendingDataRef.current.push(chunk);
		if (displayedCharsRef.current === totalCharsRef.current) {
			renderBatch();
		}
	};

	const contextValue = {
		prevPrompts,
		setPrevPrompts,
		onSent,
		setShowResults,
		setRecentPrompt,
		setResultData,
		recentPrompt,
		input,
		onRender,
		setInput,
		showResults,
		prevResults,
    	setPrevResults,
		setLoading,
		loading,
		resultData,
		newChat,
		graphData,
		setGraphData,
		evenData,
		setEvenData,
		socket,
		setSocket,
		downloadData,
		setDownloadData,
		onRenderAgent,
		agentData,
		setAgentData,
		chatNo,
		setChatNo,
		fileHistory,
		setFileHistory,
		resp,
		uploadNotice,
		pushUploadNotice,
		totalDisplayedCharsRef,
		setTotalDisplayedCharsRef,
		resultsTable,
		setResultsTable,
		resultsUpdatedAt,
		setResultsUpdatedAt,
		threads,
		deleteThread,
		setThreads,
		activeThreadId,
		setActiveThreadId,
		chatMessages,
		setChatMessages,
		chatHydrated,
		selectThread,
		createThread,
		appendMessage,
		persistMessage,
		refreshThreads,
		startRenderCycle,
		cancelRenderCycle
	};

	return <Context.Provider value={contextValue}>{props.children}</Context.Provider>;
};

export default ContextProvider;
