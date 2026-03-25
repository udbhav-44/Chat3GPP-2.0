import React, { useContext, useEffect, useMemo, useRef, useState } from "react";
import {
  FaCloudUploadAlt,
  FaExclamationTriangle,
  FaFileAlt,
  FaFileImage,
  FaFilePdf,
  FaFolder,
  FaSearch,
  FaSyncAlt,
  FaTrashAlt,
} from "react-icons/fa";
import { Context } from "../../context/Context";
import { deleteUpload, listUploads, uploadFiles } from "../../services/uploads";
import "./fileSystem.css";

// Helper function to get the correct icon based on the file type
const getFileIcon = (fileName) => {
  const fileExtension = fileName.split(".").pop().toLowerCase();
  
  switch (fileExtension) {
    case "pdf":
      return <FaFilePdf />;
    case "txt":
      return <FaFileAlt />;
    case "jpg":
    case "jpeg":
    case "png":
      return <FaFileImage />;
    case "folder":
      return <FaFolder />;
    default:
      return <FaFileAlt />;
  }
};

const Viewer = () => {
  const { fileHistory, setFileHistory, pushUploadNotice } = useContext(Context);
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState(null);
  const [deleteConfirm, setDeleteConfirm] = useState(null);
  const [searchQuery, setSearchQuery] = useState("");
  const fileInputRef = useRef(null);

  const fetchUploads = async () => {
    setIsLoading(true);
    try {
      const files = await listUploads();
      setFileHistory(files);
    } catch (error) {
      console.error("Failed to load uploads:", error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchUploads();
  }, []);

  const totalBytes = useMemo(() => {
    return fileHistory.reduce((acc, file) => acc + (file.size || 0), 0);
  }, [fileHistory]);

  const filteredFiles = useMemo(() => {
    const query = searchQuery.trim().toLowerCase();
    if (!query) {
      return fileHistory;
    }
    return fileHistory.filter((file) =>
      file.name.toLowerCase().includes(query)
    );
  }, [fileHistory, searchQuery]);

  const formatBytes = (bytes) => {
    if (!bytes && bytes !== 0) {
      return "-";
    }
    if (bytes < 1024) {
      return `${bytes} B`;
    }
    const kb = bytes / 1024;
    if (kb < 1024) {
      return `${kb.toFixed(1)} KB`;
    }
    const mb = kb / 1024;
    if (mb < 1024) {
      return `${mb.toFixed(1)} MB`;
    }
    const gb = mb / 1024;
    return `${gb.toFixed(1)} GB`;
  };

  const formatDate = (value) => {
    if (!value) {
      return "-";
    }
    try {
      return new Date(value).toLocaleString();
    } catch (error) {
      return value;
    }
  };

  const handleUploadClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
      fileInputRef.current.click();
    }
  };

  const handleUploadChange = async (event) => {
    const files = event.target.files;
    if (!files || files.length === 0) {
      return;
    }
    setIsUploading(true);
    const results = await uploadFiles(files);
    const successCount = results.filter((entry) => entry.ok).length;
    const failCount = results.length - successCount;

    if (successCount > 0) {
      await fetchUploads();
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
    setIsUploading(false);
  };

  const handleDelete = async (fileName) => {
    setDeleteTarget(fileName);
    try {
      await deleteUpload(fileName);
      await fetchUploads();
      pushUploadNotice({
        type: "success",
        title: "File removed",
        message: `${fileName} deleted.`,
      });
    } catch (error) {
      console.error("Failed to delete file:", error);
      pushUploadNotice({
        type: "error",
        title: "Delete failed",
        message: `Unable to delete ${fileName}.`,
      });
    } finally {
      setDeleteTarget(null);
      setDeleteConfirm(null);
    }
  };

  const totalLabel = `${fileHistory.length} file${fileHistory.length === 1 ? "" : "s"}`;

  return (
    <div className="file-explorer">
      <div className="file-explorer__header">
        <div>
          <h2>File Explorer</h2>
          <p className="file-explorer__subtitle">
            {totalLabel} · {formatBytes(totalBytes)}
          </p>
        </div>
        <div className="file-explorer__actions">
          <input
            ref={fileInputRef}
            type="file"
            multiple
            className="file-explorer__input"
            onChange={handleUploadChange}
          />
          <button
            type="button"
            className="file-explorer__button primary"
            onClick={handleUploadClick}
            disabled={isUploading}
          >
            <FaCloudUploadAlt />
            {isUploading ? "Uploading..." : "Upload"}
          </button>
          <button
            type="button"
            className="file-explorer__button ghost"
            onClick={fetchUploads}
            disabled={isLoading}
          >
            <FaSyncAlt className={isLoading ? "spin" : ""} />
            Refresh
          </button>
        </div>
      </div>

      <div className="file-explorer__toolbar">
        <div className="file-explorer__search">
          <FaSearch />
          <input
            type="text"
            placeholder="Search uploads"
            value={searchQuery}
            onChange={(event) => setSearchQuery(event.target.value)}
          />
        </div>
      </div>

      <div className="file-explorer__list">
        {isLoading && (
          <div className="file-explorer__state">Loading files...</div>
        )}

        {!isLoading && filteredFiles.length === 0 && (
          <div className="file-explorer__empty">
            <div className="file-explorer__empty-icon">
              <FaFolder />
            </div>
            <p className="file-explorer__empty-title">No uploads yet</p>
            <p className="file-explorer__empty-subtitle">
              Upload PDFs, notes, or images to keep them ready for search.
            </p>
            <button
              type="button"
              className="file-explorer__button primary"
              onClick={handleUploadClick}
            >
              <FaCloudUploadAlt />
              Upload files
            </button>
          </div>
        )}

        {!isLoading &&
          filteredFiles.map((file) => (
            <div key={file.name} className="file-row">
              <div className="file-row__name">
                <span className="file-row__icon">{getFileIcon(file.name)}</span>
                <div>
                  <p className="file-row__title">{file.name}</p>
                  <p className="file-row__meta">
                    {formatBytes(file.size)} · {formatDate(file.modified_at)}
                  </p>
                </div>
              </div>

              <div className="file-row__actions">
                {deleteConfirm === file.name ? (
                  <div className="file-row__confirm">
                    <button
                      type="button"
                      className="file-explorer__button ghost"
                      onClick={() => setDeleteConfirm(null)}
                    >
                      Cancel
                    </button>
                    <button
                      type="button"
                      className="file-explorer__button danger"
                      onClick={() => handleDelete(file.name)}
                      disabled={deleteTarget === file.name}
                    >
                      {deleteTarget === file.name ? "Removing..." : "Delete"}
                    </button>
                  </div>
                ) : (
                  <button
                    type="button"
                    className="file-explorer__button danger ghost"
                    onClick={() => setDeleteConfirm(file.name)}
                  >
                    <FaTrashAlt />
                    Remove
                  </button>
                )}
              </div>
            </div>
          ))}
      </div>

      {!isLoading && filteredFiles.length > 0 && (
        <div className="file-explorer__hint">
          <FaExclamationTriangle />
          Deleting a file removes it from the document store immediately.
        </div>
      )}
    </div>
  );
};

export default Viewer;
