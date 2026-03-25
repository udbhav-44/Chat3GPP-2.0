import "./sidebar.css";
import { assets } from "../../assets/assets";
import { useContext, useState } from "react";
import { FaCheckCircle, FaExclamationTriangle, FaTimes } from "react-icons/fa";
import { Context } from "../../context/Context";
import Viewer from "../file/fileSystem";

const Sidebar = ({ onLogout }) => {
  const [extended, setExtended] = useState(true);

  const {
    threads,
    fileHistory,
    activeThreadId,
    selectThread,
    newChat,
    deleteThread,
    uploadNotice,
    pushUploadNotice
  } = useContext(Context);

  const [isFilePopupVisible, setFilePopupVisible] = useState(false);

  // -------------------------
  // Popup controls
  // -------------------------
  const closePopup = () => {
    setFilePopupVisible(false);
  };

  const openFilePopup = () => setFilePopupVisible(true);

  return (
    <>
      <div className={`sidebar ${extended ? "extended" : "collapsed"}`}>
        <div className="sidebar-top">
          <div className="sidebar-header">
            <button
              type="button"
              className="icon-button ghost"
              title="Toggle Sidebar"
              onClick={() => setExtended((p) => !p)}
            >
              <img src={assets.menu_icon} alt="" />
            </button>
            <div className="brand">
              <div className="brand-logo">
                <img src={assets.main_logo} alt="3GPP" />
              </div>
              {extended && (
                <div className="brand-text">
                  <span>3GPP</span>
                  <small>Research Console</small>
                </div>
              )}
            </div>
          </div>
          <div className="sidebar-actions">
            <button
              type="button"
              className="primary-action"
              title="New Chat"
              onClick={newChat}
            >
              <img src={assets.edit_icon} alt="" />
              {extended && <span>New chat</span>}
            </button>
            {extended && (
              <div className="sidebar-metrics">
                <div className="metric-card">
                  <span>Threads</span>
                  <strong>{threads.length}</strong>
                </div>
                <div className="metric-card">
                  <span>Files</span>
                  <strong>{fileHistory.length}</strong>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* ---------------- RECENT ---------------- */}
        <div className="recent">
          <div className="recent-header">
            <p className="recent-title">Recent</p>
            {extended && <span className="recent-count">{threads.length}</span>}
          </div>
          {threads.length === 0 && (
            <div className="recent-empty">
              <p>No chats yet.</p>
              {extended && (
                <button type="button" onClick={newChat}>
                  Start a new chat
                </button>
              )}
            </div>
          )}
          {threads.map((thread) => (
            <div
              key={thread.id}
              className={`recent-entry ${String(activeThreadId) === String(thread.id) ? "active" : ""}`}
              onClick={() => selectThread(thread.id)}
            >
              <div className="thread-icon">
                <img src={assets.message_icon} alt="" />
              </div>
              <div className="thread-body">
                <p className="thread-title">
                  {(thread.title || thread.last_message || "New chat").slice(0, 28)}
                </p>
                {extended && (
                  <span className="thread-meta">
                    {thread.updated_at ? new Date(thread.updated_at).toLocaleDateString() : "Just now"}
                  </span>
                )}
              </div>
              {extended && (
                <button
                  type="button"
                  className="thread-delete"
                  onClick={(event) => {
                    event.stopPropagation();
                    deleteThread(thread.id);
                  }}
                  aria-label="Delete chat"
                  title="Delete chat"
                >
                  x
                </button>
              )}
            </div>
          ))}
        </div>

        {/* ---------------- BOTTOM ---------------- */}
        <div className="bottom">
          <button type="button" className="bottom-item" onClick={openFilePopup}>
            <span className="bottom-icon">
              <img src={assets.history_icon} alt="" />
            </span>
            {extended && (
              <span className="bottom-text">
                Files
                <small>Manage uploads</small>
              </span>
            )}
          </button>

          <button
            type="button"
            className="bottom-item"
            onClick={() => onLogout && onLogout()}
          >
            <span className="bottom-icon avatar">
              <img src={assets.user} alt="" />
            </span>
            {extended && (
              <span className="bottom-text">
                Logout
                <small>End session</small>
              </span>
            )}
          </button>
        </div>
      </div>

      {/* ---------------- FILE VIEWER ---------------- */}
      {isFilePopupVisible && (
        <div className="popup">
          <div className="popup-overlay" onClick={closePopup}></div>
          <div className="popup-form">
            <Viewer />
          </div>
        </div>
      )}

      {uploadNotice && (
        <div className={`upload-toast ${uploadNotice.type || "info"}`}>
          <div className="upload-toast__icon" aria-hidden="true">
            {uploadNotice.type === "success" && <FaCheckCircle />}
            {uploadNotice.type === "warning" && <FaExclamationTriangle />}
            {uploadNotice.type === "error" && <FaExclamationTriangle />}
            {!uploadNotice.type && <FaCheckCircle />}
          </div>
          <div className="upload-toast__content">
            <p className="upload-toast__title">{uploadNotice.title || "Upload update"}</p>
            <p className="upload-toast__message">{uploadNotice.message}</p>
          </div>
          <button
            type="button"
            className="upload-toast__close"
            onClick={() => pushUploadNotice(null)}
            aria-label="Dismiss notification"
          >
            <FaTimes />
          </button>
        </div>
      )}
    </>
  );
};

export default Sidebar;
