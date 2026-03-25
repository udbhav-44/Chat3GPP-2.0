import { useContext } from "react";
import { assets } from "../../assets/assets";
import "./main.css";
import { Context } from "../../context/Context";
import React, { useState, useEffect, useRef, useMemo } from "react";
import Dropdown from "../dropdown/Dropdown";
import { TypeAnimation } from 'react-type-animation';
import ReactMarkdown from "react-markdown";
import rehypeRaw from "rehype-raw";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { FaCloudUploadAlt, FaGoogleDrive } from "react-icons/fa";
import { getApiBaseUrl } from "../../services/auth";
import { getToken } from "../../services/authToken";
import { listUploads, uploadFiles } from "../../services/uploads";
import {
	uploadChatImages,
	deleteChatImage,
	getChatImageUrl,
} from "../../services/chatImages";

const Main = ({ user }) => {
	const {
		onSent,
		onRender,
		newChat,
		recentPrompt,
		showResults,
		setRecentPrompt,
		setShowResults,
		setResultData,
		setLoading,
		loading,
		resultData,
		setInput,
		input,
		evenData,
		setEvenData,
		graphData,
		setGraphData,
		downloadData,
		setDownloadData,
		socket,
		setSocket,
		agentData,
		setAgentData,
		setPrevResults,
		prevPrompts,
		setPrevPrompts,
		setResultsTable,
		setResultsUpdatedAt,
		chatNo,
		setChatNo,
		setFileHistory,
		resp,
		pushUploadNotice,
		activeThreadId,
		chatMessages,
		createThread,
		appendMessage,
		persistMessage,
		startRenderCycle,
		cancelRenderCycle,
	} = useContext(Context);
	const [socket1, setSocket1] = useState(null);

	const resultDataRef = useRef(null); // Reference to the result-data container for auto scrolling
	const agentDataRef = useRef(null);
	const agent = useRef(true);
	const pendingThreadIdRef = useRef(null);
	const isProcessingRef = useRef(false);
	const activeThreadIdRef = useRef(activeThreadId);
	const prevThreadIdRef = useRef(activeThreadId);
	const agentSocketRef = useRef(null);
	const mainSocketRef = useRef(null);

	const [markdownContent, setMarkdownContent] = useState('');
	const [isDocsEnabled, setIsDocsEnabled] = useState(false);
	const [isWebToolsEnabled, setIsWebToolsEnabled] = useState(false);
	const [isDropdownOpen, setIsDropdownOpen] = useState(false);
	const [mainWsStatus, setMainWsStatus] = useState("disconnected");
	const [agentWsStatus, setAgentWsStatus] = useState("disconnected");
	const [feedbackChoice, setFeedbackChoice] = useState(null);
	const [feedbackStatus, setFeedbackStatus] = useState("idle");
	const [feedbackComment, setFeedbackComment] = useState("");
	const responseIdRef = useRef(null);
	const [isProcessing, setIsProcessing] = useState(false);
	const [pendingImages, setPendingImages] = useState([]);
	const verboseLineRef = useRef(null);
	const verboseContentRef = useRef(null);
	const [verboseExpanded, setVerboseExpanded] = useState(false);
	const [copiedMessageId, setCopiedMessageId] = useState(null);
	const copiedMessageTimerRef = useRef(null);
	const requestStartRef = useRef(0);
	const firstTokenReceivedRef = useRef(false);
	const [ttftMs, setTtftMs] = useState(null);
	const [ttftLiveMs, setTtftLiveMs] = useState(null);
	const MAX_CHAT_IMAGES = 6;
	const MAX_CHAT_IMAGE_BYTES = 10 * 1024 * 1024;
	const CHAT_IMAGE_TYPES = new Set(["image/png", "image/jpeg", "image/webp", "image/jpg"]);
	const MODEL_OPTIONS = {
		openai: [
			"gpt-4o-mini",
			"gpt-4o",
			"gpt-4.1",
			"gpt-4.1-mini",
			"gpt-4.1-nano",
		],
		deepseek: [
			"deepseek-chat",
			"deepseek-reasoner",
		],
	};

	const [selectedProvider, setSelectedProvider] = useState("openai");
	const [selectedModel, setSelectedModel] = useState(MODEL_OPTIONS.openai[0]);

	useEffect(() => {
		const options = MODEL_OPTIONS[selectedProvider] || [];
		if (!options.includes(selectedModel)) {
			setSelectedModel(options[0] || "");
		}
	}, [selectedProvider]);

	const nowMs = () =>
		(typeof performance !== "undefined" && typeof performance.now === "function"
			? performance.now()
			: Date.now());
	const formatLatency = (value) => {
		if (!Number.isFinite(value) || value < 0) {
			return "--";
		}
		if (value < 1000) {
			return `${Math.round(value)} ms`;
		}
		return `${(value / 1000).toFixed(2)} s`;
	};
	const resetTtftState = () => {
		requestStartRef.current = 0;
		firstTokenReceivedRef.current = false;
		setTtftMs(null);
		setTtftLiveMs(null);
	};
	const beginTtftTracking = () => {
		requestStartRef.current = nowMs();
		firstTokenReceivedRef.current = false;
		setTtftMs(null);
		setTtftLiveMs(0);
	};
	const captureFirstToken = () => {
		if (firstTokenReceivedRef.current || !requestStartRef.current) {
			return;
		}
		firstTokenReceivedRef.current = true;
		const elapsed = Math.max(0, Math.round(nowMs() - requestStartRef.current));
		setTtftMs(elapsed);
		setTtftLiveMs(elapsed);
	};

	const ToggleSwitch = ({ label, checked, onToggle }) => {
		return (
			<div className="container">
				{label}{" "}
				<div className="toggle-switch">
					<input
						type="checkbox"
						className="checkbox"
						name={label}
						id={label}
						checked={checked}
						onChange={onToggle}
					/>
					<label className="label" htmlFor={label}>
						<span className="inner" />
						<span className="switch" />
					</label>
				</div>
			</div>
		);
	};

	const handleMarkdownChange = (e) => {
		setMarkdownContent(e.target.value);
	};

	const handleDocsToggle = () => {
		const next = !isDocsEnabled;
		setIsDocsEnabled(next);
		if (socket && socket.readyState === WebSocket.OPEN) {
			socket.send(JSON.stringify({ type: 'toggleRag', query: next }));
		}
	};

	const handleWebToolsToggle = () => {
		const next = !isWebToolsEnabled;
		setIsWebToolsEnabled(next);
		if (socket && socket.readyState === WebSocket.OPEN) {
			socket.send(JSON.stringify({ type: 'toggleWebTools', query: next }));
		}
	};

	const textAreaRef = useRef(null);
	const dropdownRef = useRef(null);
	const buttonContainerRef = useRef(null);
	const apiBaseUrl = getApiBaseUrl() || (typeof window !== "undefined" ? window.location.origin : "");
	const normalizeWsBase = (url) => url.replace(/^http/, "ws").replace(/\/$/, "");
	const rawWsBaseUrl = import.meta.env.VITE_WS_BASE_URL || apiBaseUrl || "ws://localhost";
	const wsBaseUrl = normalizeWsBase(rawWsBaseUrl);
	const wsRootBaseUrl = wsBaseUrl.endsWith("/ws") ? wsBaseUrl.slice(0, -3) : wsBaseUrl;
	const agentBaseUrl = normalizeWsBase(import.meta.env.VITE_AGENT_WS_BASE_URL || wsRootBaseUrl);
	const mainWsUrl = wsBaseUrl.endsWith("/ws") ? wsBaseUrl : `${wsBaseUrl}/ws`;
	const agentWsUrl = agentBaseUrl.endsWith("/agent-ws")
		? agentBaseUrl
		: `${agentBaseUrl}/agent-ws`;
	const buildAgentSubscriptionPayload = (threadId) => {
		const payload = threadId
			? { type: "subscribe", thread_id: String(threadId) }
			: { type: "unsubscribe" };
		const token = getToken();
		if (token) {
			payload.auth_token = token;
		}
		payload.user_id = user?.id || user?.email || null;
		return payload;
	};
	const sendAgentSubscription = (threadId) => {
		const ws = agentSocketRef.current;
		if (!ws || ws.readyState !== WebSocket.OPEN) {
			return;
		}
		ws.send(JSON.stringify(buildAgentSubscriptionPayload(threadId)));
	};

	const generatePDF = () => {
		// Send the raw Markdown content to the backend
		fetch(`${apiBaseUrl}/convert`, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
			},
			body: JSON.stringify({ content: markdownContent }),
		})
			.then(response => {
				if (!response.ok) {
					throw new Error('Failed to send data to the backend');
				}
				return response.json(); // Expecting a JSON response
			})
			.then(data => {
				console.log('Markdown content sent successfully to backend:', data.message);

				// Now fetch the generated HTML from the backend after it's processed
				return fetch(`${apiBaseUrl}/download-pdf`, {
					method: 'GET',
				});
			})
			.then(response => {
				if (!response.ok) {
					throw new Error('Failed to fetch the generated HTML');
				}
				return response.text(); // Convert the response to plain text (HTML content)
			})
			.then(htmlContent => {
				// Open the HTML content in a new tab
				const newTab = window.open();
				if (newTab) {
					newTab.document.open();
					newTab.document.write(htmlContent);
					newTab.document.close();
				} else {
					console.error('Failed to open a new Tab')
				}

			})
			.catch(error => {
				console.error('Error during the process:', error);
			});
	};


	// Auto-scrolling effect when resultData changes
	useEffect(() => {
		if (resultDataRef.current) {
			resultDataRef.current.scrollTop = resultDataRef.current.scrollHeight;
		}
	}, [resultData, agentData, chatMessages, loading]);
	useEffect(() => {
		if (agentDataRef.current) {
			agentDataRef.current.scrollTop = agentDataRef.current.scrollHeight;
		}
	}, [agentData]);

	useEffect(() => {
		setFeedbackChoice(null);
		setFeedbackStatus("idle");
		setCopiedMessageId(null);
		resetTtftState();
	}, [activeThreadId]);

	useEffect(() => {
		if (prevThreadIdRef.current && prevThreadIdRef.current !== activeThreadId) {
			clearPendingImages();
		}
		prevThreadIdRef.current = activeThreadId;
	}, [activeThreadId]);

	useEffect(() => {
		if (!activeThreadId) {
			isProcessingRef.current = false;
			setIsProcessing(false);
			setLoading(false);
			cancelRenderCycle();
		}
	}, [activeThreadId, cancelRenderCycle, setLoading]);

	useEffect(() => {
		activeThreadIdRef.current = activeThreadId;
	}, [activeThreadId]);

	useEffect(() => {
		sendAgentSubscription(activeThreadId);
	}, [activeThreadId, user]);

	useEffect(() => {
		if (!downloadData || !pendingThreadIdRef.current) {
			return;
		}
		const threadId = pendingThreadIdRef.current;
		const answerText = agent.current ? agentData : (markdownContent || resultData);
		if (!answerText) {
			return;
		}
		const assistantMessage = {
			role: "assistant",
			content: answerText,
			source: agent.current ? "agent" : "main",
			response_id: responseIdRef.current,
			created_at: new Date().toISOString(),
		};
		if (Number.isFinite(ttftMs) && ttftMs >= 0) {
			assistantMessage.ttft_ms = ttftMs;
		}
		appendMessage(threadId, assistantMessage);
		persistMessage(threadId, assistantMessage);
		pendingThreadIdRef.current = null;
		setResultData("");
		setAgentData("");
	}, [downloadData, markdownContent, agentData, resultData, appendMessage, persistMessage, setResultData, setAgentData, ttftMs]);


	const handleCardClick = (promptText) => {
		setInput(promptText);
	};

	const displayName = user?.name || user?.email || "Researcher";
	const shortName = displayName.split(" ")[0];
	const displayEmail = user?.email || "";
	const lastAssistantIndex = useMemo(() => {
		for (let i = chatMessages.length - 1; i >= 0; i -= 1) {
			const role = (chatMessages[i]?.role || "assistant").toLowerCase();
			if (role === "assistant") {
				return i;
			}
		}
		return -1;
	}, [chatMessages]);
	const sanitizeLogLine = (line) => {
		return line
			.replace(/\[(.*?)\]\((.*?)\)/g, "$1 ($2)")
			.replace(/`{1,3}/g, "")
			.replace(/\*\*(.*?)\*\*/g, "$1")
			.replace(/_(.*?)_/g, "$1")
			.replace(/^\s*#{1,6}\s*/, "")
			.replace(/^\s*[-*+]\s+/, "")
			.trim();
	};
	const sanitizedLogLines = useMemo(() => {
		if (!agentData) {
			return [];
		}
		return agentData
			.replace(/\r\n/g, "\n")
			.split("\n")
			.map((line) => sanitizeLogLine(line))
			.filter((line, index, arr) => line || (index > 0 && arr[index - 1]));
	}, [agentData]);
	const latestLogLine = useMemo(() => {
		if (!sanitizedLogLines.length) {
			return "";
		}
		return sanitizedLogLines[sanitizedLogLines.length - 1];
	}, [sanitizedLogLines]);
	const verboseText = useMemo(() => sanitizedLogLines.join("\n"), [sanitizedLogLines]);
	const lastAssistantMessage = useMemo(() => {
		for (let i = chatMessages.length - 1; i >= 0; i -= 1) {
			const role = (chatMessages[i]?.role || "assistant").toLowerCase();
			if (role === "assistant") {
				return chatMessages[i];
			}
		}
		return null;
	}, [chatMessages]);
	const feedbackAnswerText =
		(agent.current ? agentData : (markdownContent || resultData)) ||
		lastAssistantMessage?.content ||
		"";
	const feedbackSource =
		(agent.current && (agentData || markdownContent || resultData))
			? "agent"
			: (lastAssistantMessage?.source || "main");
	const canShowFeedback = showResults && !loading && Boolean(feedbackAnswerText);
	const ttftValue = ttftMs ?? ttftLiveMs;
	const ttftLabel = Number.isFinite(ttftValue) ? formatLatency(ttftValue) : null;
	const loadingTimingLabel = ttftLabel ? `Thinking for ${ttftLabel}` : null;

	const handleClick = async () => {
		const trimmedInput = input.trim();
		if ((!trimmedInput && pendingImages.length === 0) || isProcessingRef.current) {
			return;
		}
		const userContent = trimmedInput || "Shared an image.";
		const queryText = trimmedInput || "Analyze the attached image(s) and respond.";
		const historySnapshot = chatMessages.slice(-6).map((message) => ({
			role: message.role,
			content: message.content,
		}));
		startRenderCycle();
		isProcessingRef.current = true;
		setIsProcessing(true);
		let threadId = activeThreadId;
		if (!threadId) {
			const title = trimmedInput ? trimmedInput.slice(0, 60) : "Image chat";
			threadId = await createThread(title);
		}
		if (!threadId) {
			isProcessingRef.current = false;
			setIsProcessing(false);
			return;
		}

		setInput("");
		setResultData("");
		setShowResults(true);
		setLoading(true);
		setDownloadData(false);
		setFeedbackChoice(null);
		setFeedbackStatus("idle");
		setFeedbackComment("");
		responseIdRef.current = `resp_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
		resp.current = true;
		setRecentPrompt(userContent);
		if (chatNo == 0) {
			setPrevPrompts(prev => [...prev, userContent]);
		}
		setChatNo(chatNo + 1);

		const attachments = pendingImages.map((image) => ({
			filename: image.filename,
			mime_type: image.mime_type,
			size: image.size,
		}));
		const userMessage = {
			role: "user",
			content: userContent,
			attachments,
			created_at: new Date().toISOString(),
		};
		appendMessage(threadId, userMessage);
		persistMessage(threadId, userMessage);
		pendingThreadIdRef.current = threadId;
		clearPendingImages();

		let query = queryText;
		if (socket && socket.readyState === WebSocket.OPEN) {
			beginTtftTracking();
			socket.send(JSON.stringify({
				type: 'query',
				query,
				thread_id: threadId,
				history: historySnapshot,
				response_id: responseIdRef.current,
				model: selectedModel,
				provider: selectedProvider,
				web_tools: isWebToolsEnabled,
				auth_token: getToken(),
				user_id: user?.id || user?.email || null,
				images: attachments,
			}));
		}
	}

	const handleAbort = () => {
		if (!isProcessingRef.current) {
			return;
		}
		isProcessingRef.current = false;
		setIsProcessing(false);
		resp.current = false;
		setLoading(false);
		setDownloadData(false);
		cancelRenderCycle();
		resetTtftState();

		const abortPayload = JSON.stringify({
			type: "abort",
			thread_id: pendingThreadIdRef.current,
			response_id: responseIdRef.current,
		});
		if (socket && socket.readyState === WebSocket.OPEN) {
			socket.send(abortPayload);
		}
		if (socket1 && socket1.readyState === WebSocket.OPEN) {
			socket1.send(abortPayload);
		}
		pendingThreadIdRef.current = null;
	};

	useEffect(() => {
		if (!downloadData) {
			return;
		}
		isProcessingRef.current = false;
		setIsProcessing(false);
	}, [downloadData]);

	useEffect(() => {
		if (!isProcessing || ttftMs !== null || !requestStartRef.current) {
			return;
		}
		const interval = setInterval(() => {
			setTtftLiveMs(Math.max(0, Math.round(nowMs() - requestStartRef.current)));
		}, 100);
		return () => clearInterval(interval);
	}, [isProcessing, ttftMs]);

	useEffect(() => {
		if (verboseLineRef.current) {
			verboseLineRef.current.scrollLeft = verboseLineRef.current.scrollWidth;
		}
	}, [latestLogLine]);

	useEffect(() => {
		if (!verboseExpanded || !verboseContentRef.current) {
			return;
		}
		verboseContentRef.current.scrollTop = verboseContentRef.current.scrollHeight;
	}, [verboseText, verboseExpanded]);

	useEffect(() => {
		return () => {
			if (copiedMessageTimerRef.current) {
				clearTimeout(copiedMessageTimerRef.current);
			}
		};
	}, []);

	const handleCopyText = async (text) => {
		if (!text) {
			return false;
		}
		try {
			if (navigator?.clipboard?.writeText) {
				await navigator.clipboard.writeText(text);
				return true;
			}
			const textarea = document.createElement("textarea");
			textarea.value = text;
			textarea.setAttribute("readonly", "");
			textarea.style.position = "fixed";
			textarea.style.top = "-1000px";
			document.body.appendChild(textarea);
			textarea.focus();
			textarea.select();
			const ok = document.execCommand("copy");
			document.body.removeChild(textarea);
			return ok;
		} catch (error) {
			console.error("Failed to copy content:", error);
			return false;
		}
	};

	const handleCopyMessage = async (text, id) => {
		const ok = await handleCopyText(text);
		if (!ok) {
			return;
		}
		setCopiedMessageId(id);
		if (copiedMessageTimerRef.current) {
			clearTimeout(copiedMessageTimerRef.current);
		}
		copiedMessageTimerRef.current = setTimeout(() => {
			setCopiedMessageId(null);
		}, 1200);
	};

	const handleFeedback = (rating) => {
		if (!canShowFeedback || feedbackStatus === "sending") {
			return;
		}
		setFeedbackChoice(rating);
		setFeedbackStatus("idle");
	};

	const handleFeedbackSubmit = async () => {
		if (!canShowFeedback || feedbackStatus === "sending" || !feedbackChoice) {
			return;
		}

		const token = getToken();
		const answerText = feedbackAnswerText;

		setFeedbackStatus("sending");

		try {
			const response = await fetch(`${apiBaseUrl}/api/feedback`, {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
					...(token ? { Authorization: `Bearer ${token}` } : {}),
				},
				body: JSON.stringify({
					rating: feedbackChoice,
					comment: feedbackComment.trim(),
					prompt: recentPrompt,
					answer: answerText,
					source: feedbackSource,
					response_id: responseIdRef.current,
					user: {
						name: user?.name || "",
						email: user?.email || "",
					},
				}),
			});

			if (!response.ok) {
				throw new Error("Feedback request failed");
			}
			setFeedbackStatus("sent");
		} catch (error) {
			console.error("Error submitting feedback:", error);
			setFeedbackStatus("error");
		}
	};

	// Adjust textarea height dynamically
	const adjustHeight = () => {
		const textArea = textAreaRef.current;

		// Reset the height to auto to shrink it before resizing
		textArea.style.height = 'auto';

		// Adjust the height based on scrollHeight
		textArea.style.height = `${textArea.scrollHeight}px`;

		// Move the textarea upwards by adjusting margin-top dynamically
		const diff = textArea.scrollHeight - textArea.clientHeight;
	};

	// Adjust height when input changes
	useEffect(() => {
		adjustHeight();
	}, [input]);

	const isImageFile = (file) => {
		if (!file) {
			return false;
		}
		if (CHAT_IMAGE_TYPES.has(file.type)) {
			return true;
		}
		const name = String(file.name || "").toLowerCase();
		return name.endsWith(".png") || name.endsWith(".jpg") || name.endsWith(".jpeg") || name.endsWith(".webp");
	};

	const ensureThreadForUpload = async () => {
		if (activeThreadId) {
			return activeThreadId;
		}
		const createdId = await createThread("New chat");
		return createdId;
	};

	const clearPendingImages = () => {
		setPendingImages((prev) => {
			prev.forEach((item) => {
				if (item?.previewUrl) {
					URL.revokeObjectURL(item.previewUrl);
				}
			});
			return [];
		});
	};

	const handleImageFiles = async (files) => {
		const entries = Array.from(files || []);
		if (!entries.length) {
			return;
		}
		const remainingSlots = Math.max(0, MAX_CHAT_IMAGES - pendingImages.length);
		if (remainingSlots === 0) {
			pushUploadNotice({
				type: "warning",
				title: "Image limit reached",
				message: `You can attach up to ${MAX_CHAT_IMAGES} images.`,
			});
			return;
		}
		const imageEntries = entries.filter(isImageFile);
		if (!imageEntries.length) {
			return;
		}
		const validImages = [];
		const rejected = [];
		for (const file of imageEntries) {
			if (file.size > MAX_CHAT_IMAGE_BYTES) {
				rejected.push(file);
				continue;
			}
			validImages.push(file);
		}
		if (!validImages.length) {
			pushUploadNotice({
				type: "error",
				title: "Images too large",
				message: "Each image must be 10MB or smaller.",
			});
			return;
		}
		const limitedImages = validImages.slice(0, remainingSlots);
		const threadId = await ensureThreadForUpload();
		if (!threadId) {
			pushUploadNotice({
				type: "error",
				title: "Upload failed",
				message: "We could not create a chat thread.",
			});
			return;
		}
		const results = await uploadChatImages(limitedImages, threadId);
		const successes = [];
		const failures = [];
		results.forEach((entry) => {
			if (entry.ok) {
				const previewUrl = URL.createObjectURL(entry.file);
				successes.push({
					id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
					filename: entry.payload?.filename || entry.file.name,
					mime_type: entry.file.type || entry.payload?.content_type || "",
					size: entry.file.size,
					previewUrl,
				});
			} else {
				failures.push(entry.file);
			}
		});
		if (successes.length) {
			setPendingImages((prev) => [...prev, ...successes]);
		}
		const failCount = failures.length + Math.max(0, validImages.length - limitedImages.length) + rejected.length;
		if (successes.length && !failCount) {
			pushUploadNotice({
				type: "success",
				title: "Images attached",
				message: `${successes.length} image${successes.length === 1 ? "" : "s"} ready.`,
			});
		} else if (successes.length) {
			pushUploadNotice({
				type: "warning",
				title: "Partial image upload",
				message: `${successes.length} added, ${failCount} skipped.`,
			});
		} else {
			pushUploadNotice({
				type: "error",
				title: "Image upload failed",
				message: "We could not upload those images.",
			});
		}
	};

	const handlePaste = async (event) => {
		if (!event.clipboardData) {
			return;
		}
		const items = Array.from(event.clipboardData.items || []);
		const imageFiles = [];
		for (const item of items) {
			if (item.type && item.type.startsWith("image/")) {
				const blob = item.getAsFile();
				if (!blob) {
					continue;
				}
				const ext = blob.type === "image/png"
					? "png"
					: blob.type === "image/webp"
					? "webp"
					: "jpg";
				const name = `paste-${Date.now()}.${ext}`;
				const file = new File([blob], name, { type: blob.type });
				imageFiles.push(file);
			}
		}
		if (imageFiles.length) {
			event.preventDefault();
			await handleImageFiles(imageFiles);
		}
	};

	const removePendingImage = async (image) => {
		if (!image) {
			return;
		}
		if (image.previewUrl) {
			URL.revokeObjectURL(image.previewUrl);
		}
		setPendingImages((prev) => prev.filter((item) => item.id !== image.id));
		if (activeThreadId && image.filename) {
			try {
				await deleteChatImage(activeThreadId, image.filename);
			} catch (error) {
				console.error("Failed to delete image:", error);
			}
		}
	};

	const renderAttachments = (attachments) => {
		if (!attachments || attachments.length === 0) {
			return null;
		}
		return (
			<div className="chat-attachments">
				{attachments.map((attachment, index) => {
					const filename = attachment?.filename;
					const src = filename ? getChatImageUrl(activeThreadId, filename) : "";
					if (!src) {
						return null;
					}
					return (
						<a
							key={`${filename}-${index}`}
							className="chat-attachment"
							href={src}
							target="_blank"
							rel="noopener noreferrer"
						>
							<img src={src} alt={filename} loading="lazy" />
						</a>
					);
				})}
			</div>
		);
	};

	const handleFileChange = async (event) => {
		const files = event.target.files;
		setEvenData(event);

		if (files && files.length > 0) {
			try {
				const entries = Array.from(files);
				const imageFiles = entries.filter(isImageFile);
				const docFiles = entries.filter((file) => !isImageFile(file));
				if (imageFiles.length) {
					await handleImageFiles(imageFiles);
				}
				if (!docFiles.length) {
					setIsDropdownOpen(false);
					event.target.value = "";
					return;
				}
				const results = await uploadFiles(docFiles);
				const successCount = results.filter((entry) => entry.ok).length;
				const failCount = results.length - successCount;

				if (successCount > 0) {
					try {
						const uploaded = await listUploads();
						setFileHistory(uploaded);
					} catch (error) {
						console.error("Failed to refresh uploads:", error);
					}
				}

				if (successCount > 0 && failCount === 0) {
					pushUploadNotice({
						type: "success",
						title: "Upload complete",
						message: `${successCount} file${successCount === 1 ? "" : "s"} added.`,
					});
				} else if (successCount > 0) {
					pushUploadNotice({
						type: "warning",
						title: "Partial upload",
						message: `${successCount} succeeded, ${failCount} failed.`,
					});
				} else {
					pushUploadNotice({
						type: "error",
						title: "Upload failed",
						message: "We could not upload those files. Try again.",
					});
				}
			} catch (error) {
				console.error("Error during upload:", error);
				pushUploadNotice({
					type: "error",
					title: "Upload failed",
					message: "We could not upload those files. Try again.",
				});
			}
		};
		setIsDropdownOpen(false);
		event.target.value = "";
	}

	const triggerFileInput = () => {

		document.getElementById('hiddenFileInput').click(); // Programmatically trigger click on hidden input
	};

	// Toggle the dropdown visibility
	const toggleDropdown = () => {
		setIsDropdownOpen(!isDropdownOpen);
	};

	// Close the dropdown
	const closeDropdown = () => {
		setIsDropdownOpen(false);
	};

	useEffect(() => {
		if (!isDropdownOpen) {
			return;
		}
		const handleClickOutside = (event) => {
			const dropdown = dropdownRef.current;
			const buttonContainer = buttonContainerRef.current;
			if (
				dropdown &&
				buttonContainer &&
				!dropdown.contains(event.target) &&
				!buttonContainer.contains(event.target)
			) {
				setIsDropdownOpen(false);
			}
		};
		document.addEventListener("mousedown", handleClickOutside);
		return () => {
			document.removeEventListener("mousedown", handleClickOutside);
		};
	}, [isDropdownOpen]);

	useEffect(() => {
		let ws = null;
		let reconnectTimer = null;
		let retryCount = 0;
		let shouldReconnect = true;

		const scheduleReconnect = () => {
			if (!shouldReconnect || reconnectTimer) {
				return;
			}
			const delay = Math.min(10000, 500 * 2 ** retryCount);
			reconnectTimer = setTimeout(() => {
				reconnectTimer = null;
				if (shouldReconnect) {
					connect();
				}
			}, delay);
			retryCount += 1;
		};

		const connect = () => {
			ws = new WebSocket(agentWsUrl);
			agentSocketRef.current = ws;
			setSocket1(ws);

			ws.onopen = () => {
				console.log('WebSocket connected to agent server');
				retryCount = 0;
				setAgentWsStatus("connected");
				sendAgentSubscription(activeThreadIdRef.current);
			};
			ws.onmessage = (event) => {
				try {
					const data = JSON.parse(event.data);
					const currentThreadId = activeThreadIdRef.current;
					const incomingThreadId = data.thread_id;
					if (incomingThreadId && currentThreadId && String(incomingThreadId) !== String(currentThreadId)) {
						return;
					}

					// ✅ HANDLE VERBOSE LOGS
					if (data.type === 'logs') {
						console.log("Agent logs:", data.response);
						agent.current = true;
						if (!isProcessingRef.current) {
							return;
						}
						if (data.response && String(data.response).trim()) {
							captureFirstToken();
						}
						setAgentData(prev => prev + data.response);  // ensure UI updates
						setMarkdownContent(prev => prev + data.response);
					}

					// ✅ HANDLE SIDEBAR RESULTS (CSV)
					else if (data.type === 'results') {
						if (!isProcessingRef.current) {
							return;
						}
						const columns = Array.isArray(data.columns) ? data.columns.filter(Boolean) : [];
						const rows = Array.isArray(data.rows) ? data.rows : [];
						setResultsTable({ columns, rows }, data.thread_id);
						setResultsUpdatedAt(Date.now(), data.thread_id);
					}

				} catch (error) {
					console.error('Error parsing WebSocket message:', error);
				}
			};

			ws.onclose = () => {
				console.log('WebSocket disconnected');
				setAgentWsStatus("disconnected");
				scheduleReconnect();
			};
			ws.onerror = () => {
				setAgentWsStatus("error");
			};
		};

		connect();

		return () => {
			shouldReconnect = false;
			if (reconnectTimer) {
				clearTimeout(reconnectTimer);
			}
			if (ws) {
				ws.close();
			}
		};
	}, [agentWsUrl]);

	useEffect(() => {
		let ws = null;
		let reconnectTimer = null;
		let retryCount = 0;
		let shouldReconnect = true;

		const scheduleReconnect = () => {
			if (!shouldReconnect || reconnectTimer) {
				return;
			}
			const delay = Math.min(10000, 500 * 2 ** retryCount);
			reconnectTimer = setTimeout(() => {
				reconnectTimer = null;
				if (shouldReconnect) {
					connect();
				}
			}, delay);
			retryCount += 1;
		};

		const connect = () => {
			ws = new WebSocket(mainWsUrl);
			mainSocketRef.current = ws;
			setSocket(ws);

			ws.onopen = () => {
				console.log('WebSocket connected');
				retryCount = 0;
				setMainWsStatus("connected");
			};
			ws.onmessage = (event) => {
				try {
					const data = JSON.parse(event.data);
					// console.log(data);
					const currentThreadId = activeThreadIdRef.current;
                    if (data.thread_id && currentThreadId && String(data.thread_id) !== String(currentThreadId)) {
                        return;
                    }
                    if (data.response_id && responseIdRef.current && data.response_id !== responseIdRef.current) {
                        return;
                    }
                    if (data.type === 'graph') {
                        if (!isProcessingRef.current) {
                            return;
                        }

                        const graph = JSON.parse(data.response);
                        console.log(graph);
                        setGraphData(graph);

                    } else if (data.type === 'response') {
                        if (!isProcessingRef.current) {
                            return;
                        }
                        captureFirstToken();
                        agent.current = false;
                        onRender(data.response);
                        console.log(data.response);
                        setMarkdownContent(data.response);
                    }
				} catch (error) {
					console.error('Error parsing WebSocket message:', error);
				}
			};
			ws.onclose = () => {
				console.log('WebSocket disconnected');
				setMainWsStatus("disconnected");
				scheduleReconnect();
			};
			ws.onerror = () => {
				setMainWsStatus("error");
			};
		};

		connect();

		return () => {
			shouldReconnect = false;
			if (reconnectTimer) {
				clearTimeout(reconnectTimer);
			}
			if (ws) {
				ws.close();
			}
		};
	}, [mainWsUrl]);

	return (

		<div
			className={`main ${showResults ? "main--results" : "main--home"}`}
			tabIndex="0"
			onKeyDown={(e) => {
			if (e.key === 'Enter' && !isProcessingRef.current) {
				e.preventDefault();
				handleClick();
			}
		}}
		>
			<div className="nav">
				<div className="nav-inner">
					<img src={assets.main_logo} className="pway" alt="" />
					<div className="rightside">
						<div className="status-stack">
							<span className={`status-pill ${mainWsStatus}`}>Main WS</span>
							<span className={`status-pill ${agentWsStatus}`}>Agent WS</span>
						</div>
						<Dropdown
							selectedProvider={selectedProvider}
							selectedModel={selectedModel}
							onProviderChange={setSelectedProvider}
							onModelChange={setSelectedModel}
							modelOptions={MODEL_OPTIONS}
						/>
						<ToggleSwitch label={"Docs"} checked={isDocsEnabled} onToggle={handleDocsToggle} />
						<ToggleSwitch label={"Web"} checked={isWebToolsEnabled} onToggle={handleWebToolsToggle} />
						<div className="user-meta">
							<p className="user-name">{shortName}</p>
							{displayEmail && <p className="user-email">{displayEmail}</p>}
						</div>
						<img src={assets.user} className="user" alt="" />
					</div>
				</div>
			</div>
			<div className="main-content">
				<div className="main-container" >
					{!showResults ? (
						<>
							<div className="contain">
								<div className="greet">
									<TypeAnimation
										sequence={[
											`Hello, ${shortName}!`,
										]}
										speed={{ type: 'keyStrokeDelayInMs', value: 100 }}
										style={{ fontSize: '1em' }}
									/>
									<p style={{ fontSize: '0.75em' }}>
										What would you like to explore today?
									</p>
								</div>
								<div className="cards">
									<div
										className="card"
										onClick={() =>
											handleCardClick(
												"Summarize the latest RAN meeting outcomes for Release 18."
											)
										}
									>
										<p style={{ textAlign: "justify" }}>
											Summarize the latest RAN meeting outcomes for Release 18.
										</p>
										{/* <img src={assets.compass_icon} alt="" /> */}
									</div>
									<div
										className="card"
										onClick={() => {
											handleCardClick(
												"Find documents about sidelink enhancements and cite the top sources."
											);
										}}
									>
										<p style={{ textAlign: "justify" }}>
											Find documents about sidelink enhancements and cite the top sources.
										</p>
									</div>
									<div
										className="card"
										onClick={() =>
											handleCardClick(
												"Compare TS 38.331 vs TS 38.321 and list key differences."
											)
										}
									>
										<p style={{ textAlign: "justify" }}>
											Compare TS 38.331 vs TS 38.321 and list key differences.</p>
										{/* <img src={assets.message_icon} alt="" /> */}
									</div>
									<div
										className="card"
										onClick={() =>
											handleCardClick(
												"Show me the most relevant documents for NR positioning accuracy."
											)
										}
									>
										<p style={{ textAlign: "justify" }}>
											Show me the most relevant documents for NR positioning accuracy.
										</p>
									</div>
									{/* Your card elements here */}
								</div>
							</div>

						</>
					) : (
						<div className="result" ref={resultDataRef}>
							<div className="chat-thread">
								{chatMessages.map((message, index) => {
									const role = (message.role || "assistant").toLowerCase();
									const messageKey = message.id || `${role}-${index}`;
									const isAssistantCopied = copiedMessageId === messageKey;
									const isLastAssistant = role === "assistant" && index === lastAssistantIndex;
									const messageTtft = Number(message.ttft_ms);
									const hasMessageTtft = Number.isFinite(messageTtft) && messageTtft >= 0;
									return (
										<React.Fragment key={messageKey}>
											<div className={`chat-row ${role}`}>
												<img
													src={role === "user" ? assets.user : assets.pway_icon}
													className="chat-avatar"
													alt=""
												/>
												<div className={`chat-bubble ${role}`}>
													{role === "user" ? (
														<>
															<p className="chat-bubble__text">{message.content}</p>
															{renderAttachments(message.attachments)}
														</>
													) : (
														<ReactMarkdown
															rehypePlugins={[rehypeRaw, [rehypeKatex, { throwOnError: false, strict: "ignore" }]]}
															remarkPlugins={[remarkGfm, remarkMath]}
															components={{
																a: ({ href, children }) => (
																	<a href={href} target="_blank" rel="noopener noreferrer">
																		{children}
																	</a>
																)
															}}
														>
															{message.content}
														</ReactMarkdown>
													)}
													{role !== "user" && renderAttachments(message.attachments)}
												</div>
											</div>
												{role === "assistant" && (
													<>
														{hasMessageTtft && (
															<div className="chat-row chat-timing-row">
																<span className="chat-timing" title="Time to first response token">
																	Thought for {formatLatency(messageTtft)}
																</span>
															</div>
														)}
														<div className="chat-row chat-actions-row">
															<div className="chat-actions">
																<button
																	type="button"
																	className="feedback-button copy"
																	onClick={() => handleCopyMessage(message.content, messageKey)}
																	title="Copy answer"
																	aria-label="Copy answer"
																>
																	{isAssistantCopied ? (
																		<svg viewBox="0 0 24 24" aria-hidden="true">
																			<path
																				fill="currentColor"
																				d="M9 16.17l-3.88-3.88L3.7 13.7 9 19l12-12-1.41-1.41z"
																			/>
																		</svg>
																	) : (
																		<svg viewBox="0 0 24 24" aria-hidden="true">
																			<path
																				fill="currentColor"
																				d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"
																			/>
																		</svg>
																	)}
																</button>
																{canShowFeedback && isLastAssistant && (
																	<>
																		<button
																			type="button"
																			className={`feedback-button up ${feedbackChoice === "up" ? "active" : ""}`}
																			onClick={() => handleFeedback("up")}
																			disabled={feedbackStatus === "sending"}
																			title="Thumbs up"
																			aria-label="Thumbs up"
																		>
																			<svg viewBox="0 0 24 24" aria-hidden="true">
																				<path
																					fill="currentColor"
																					d="M2 21h4V9H2v12zm20-11c0-1.1-.9-2-2-2h-6.3l1-4.6.03-.32c0-.41-.17-.79-.44-1.06L13 1 7.6 6.4C7.22 6.78 7 7.3 7 7.83V19c0 1.1.9 2 2 2h7c.82 0 1.54-.5 1.84-1.22l3-7.05c.1-.23.16-.48.16-.73V10z"
																				/>
																			</svg>
																		</button>
																		<button
																			type="button"
																			className={`feedback-button down ${feedbackChoice === "down" ? "active" : ""}`}
																			onClick={() => handleFeedback("down")}
																			disabled={feedbackStatus === "sending"}
																			title="Thumbs down"
																			aria-label="Thumbs down"
																		>
																			<svg viewBox="0 0 24 24" aria-hidden="true">
																				<path
																					fill="currentColor"
																					d="M22 3h-4v12h4V3zM2 14c0 1.1.9 2 2 2h6.3l-1 4.6-.03.32c0 .41.17.79.44 1.06L11 23l5.4-5.4c.38-.38.6-.9.6-1.43V5c0-1.1-.9-2-2-2H8c-.82 0-1.54.5-1.84 1.22l-3 7.05c-.1.23-.16.48-.16.73V14z"
																				/>
																			</svg>
																		</button>
																	</>
																)}
															</div>
														</div>
													</>
												)}
										</React.Fragment>
									);
								})}
								{(loading || resultData || agentData) && (
									<div className="chat-row assistant">
										<img src={assets.pway_icon} className="chat-avatar" alt="" />
										<div className={`chat-bubble assistant ${loading ? "loading" : ""}`}>
											{loading ? (
												agentData ? (
													<div className="assistant-verbose">
														<div className="assistant-verbose__header">
															<span className="assistant-verbose__badge">Processing log</span>
															<div className="assistant-verbose__actions">
																{loadingTimingLabel && (
																	<span className="assistant-loading__timing">{loadingTimingLabel}</span>
																)}
																<div className="assistant-loading__dots" aria-hidden="true">
																	<span />
																	<span />
																	<span />
																</div>
																<button
																	type="button"
																	className="assistant-verbose__toggle"
																	onClick={() => setVerboseExpanded((prev) => !prev)}
																	aria-expanded={verboseExpanded}
																	aria-label={verboseExpanded ? "Collapse processing log" : "Expand processing log"}
																	title={verboseExpanded ? "Collapse" : "Expand"}
																>
																	<span className="assistant-verbose__chevron" aria-hidden="true" />
																</button>
															</div>
														</div>
														<div
															className="assistant-verbose__line"
															ref={verboseLineRef}
															title={latestLogLine}
															onClick={() => setVerboseExpanded(true)}
														>
															{latestLogLine || "Waiting for logs..."}
														</div>
														{verboseExpanded && (
															<pre className="assistant-verbose__full" ref={verboseContentRef}>
																{verboseText || "Waiting for logs..."}
															</pre>
														)}
													</div>
												) : (
													<div className="assistant-loading">
														<div className="assistant-loading__header">
															<span className="assistant-loading__badge">Processing</span>
															{loadingTimingLabel && (
																<span className="assistant-loading__timing">{loadingTimingLabel}</span>
															)}
															<div className="assistant-loading__dots" aria-hidden="true">
																<span />
																<span />
																<span />
															</div>
														</div>
														<div className="assistant-loading__lines">
															<span className="assistant-loading__line line-lg" />
															<span className="assistant-loading__line line-md" />
														</div>
													</div>
												)
											) : (
												<ReactMarkdown
													rehypePlugins={[rehypeRaw, [rehypeKatex, { throwOnError: false, strict: "ignore" }]]}
													remarkPlugins={[remarkGfm, remarkMath]}
													components={{
														a: ({ href, children }) => (
															<a href={href} target="_blank" rel="noopener noreferrer">
																{children}
															</a>
														)
													}}
												>
													{agent.current ? agentData : resultData}
												</ReactMarkdown>
											)}
										</div>
									</div>
								)}
							</div>
							{canShowFeedback &&
								<div className="result-data feedback-row" ref={agentDataRef} style={{ overflow: 'auto' }}>
									{downloadData && (
										<button
											type="button"
											className="download-button"
											onClick={generatePDF}
											title="Download report"
											aria-label="Download report"
										>
											<img src={assets.download_icon} alt="" />
										</button>
									)}
								</div>

							}
							{canShowFeedback && feedbackChoice && (
								feedbackStatus === "sent" ? (
									<div className="feedback-thanks" role="status">
										<span>Thanks for the feedback.</span>
									</div>
								) : (
									<div className="feedback-comment">
										<label className="feedback-comment__label" htmlFor="feedback-comment">
											Add a comment (optional)
										</label>
										<textarea
											id="feedback-comment"
											className="feedback-comment__input"
											value={feedbackComment}
											onChange={(e) => setFeedbackComment(e.target.value)}
											placeholder="Share what worked well or what was missing..."
											rows={3}
										/>
										<div className="feedback-comment__actions">
											<span className="feedback-comment__hint">
												{feedbackStatus === "error"
													? "Could not submit. Try again."
													: "Submit to send your feedback."}
											</span>
											<button
												type="button"
												className="feedback-comment__submit"
												onClick={handleFeedbackSubmit}
												disabled={feedbackStatus === "sending" || !feedbackChoice}
											>
												{feedbackStatus === "sending" ? "Submitting..." : "Submit"}
											</button>
										</div>
									</div>
								)
							)}

						</div>
					)}
				</div>
				<div className="main-bottom">
					<div className="search-box">
						{pendingImages.length > 0 && (
							<div className="pending-images">
								{pendingImages.map((image) => (
									<div className="pending-image" key={image.id}>
										<img src={image.previewUrl} alt={image.filename} />
										<button
											type="button"
											className="pending-image__remove"
											onClick={() => removePendingImage(image)}
											aria-label="Remove image"
										>
											×
										</button>
									</div>
								))}
								<div className="pending-image__meta">
									{pendingImages.length}/{MAX_CHAT_IMAGES} images
								</div>
							</div>
						)}
						<div className="search-input-row">
							<textarea
								ref={textAreaRef}
								onChange={(e) => setInput(e.target.value)}
								onPaste={handlePaste}
								value={input}
								placeholder="Ask about meetings, releases, specs, or upload docs..."
								rows={1} // Start with 1 row
								className="search-input"
							/>
							<div className="button-container" ref={buttonContainerRef}>
								<img
									src={assets.attach_icon}
									className="upload"
									onClick={!isProcessing ? toggleDropdown : null}
									alt=""
								/>
								{isProcessing ? (
									<button
										type="button"
										className="stop-button"
										onClick={handleAbort}
										aria-label="Stop generating"
									>
										<span className="stop-icon" aria-hidden="true" />
									</button>
								) : (
									<img
										src={assets.send_icon}
										alt=""
										onClick={!isProcessing ? handleClick : null}
									/>
								)}
							</div>
						</div>
					</div>
					<div className="bottom-info">
						<p></p>
					</div>
				</div>
				{/* Overlay and Dropdown */}
				{isDropdownOpen && (
					<>
						{/* Overlay */}
						<div className="overlay" onClick={closeDropdown}></div>

						{/* Dropdown Content */}
						<div
							ref={dropdownRef}
							className="upload-menu"
						>
							<div className="upload-menu__header">
								<p>Upload sources</p>
								<span>PDF, DOCX, TXT, images</span>
							</div>
							<button
								type="button"
								className="upload-menu__item"
								onClick={triggerFileInput}
							>
								<span className="upload-menu__icon">
									<FaCloudUploadAlt />
								</span>
								<span className="upload-menu__text">
									<span>From computer</span>
									<small>Fast local upload</small>
								</span>
								<span className="upload-menu__badge">Local</span>
							</button>
							<input
								multiple
								id="hiddenFileInput"
								type="file"
								accept=".pdf,.doc,.docx,.txt,.png,.jpg,.jpeg,.webp"
								style={{ display: "none" }}
								onChange={handleFileChange}
							/>
							<a
								href="https://drive.google.com/drive/folders/1bmB1oKZ3J8_Onbd-pQKbhiBDLi8AGls9"
								target="_blank"
								rel="noopener noreferrer"
								className="upload-menu__item link"
								onClick={closeDropdown}
							>
								<span className="upload-menu__icon">
									<FaGoogleDrive />
								</span>
								<span className="upload-menu__text">
									<span>Google Drive</span>
									<small>Send docs to the shared folder</small>
								</span>
								<span className="upload-menu__badge">Cloud</span>
							</a>
						</div>
					</>
				)}
			</div>
		</div>
	);
};

export default Main;
