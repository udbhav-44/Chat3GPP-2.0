import { useEffect, useState } from "react";
import {
  forgotPassword,
  getApiBaseUrl,
  login,
  resetPassword,
  signup,
} from "../../services/auth";
import { setToken } from "../../services/authToken";
import "./login.css";

const Login = ({ onAuthSuccess, checking = false }) => {
  const [mode, setMode] = useState("login");
  const [status, setStatus] = useState({ type: "", message: "" });
  const [resetLink, setResetLink] = useState("");
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [form, setForm] = useState({
    name: "",
    email: "",
    password: "",
    confirmPassword: "",
    resetToken: "",
  });

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const resetToken = params.get("reset_token");
    const resetEmail = params.get("email");
    const oauthError = params.get("oauth_error");

    if (resetToken) {
      setMode("reset");
      setForm((prev) => ({
        ...prev,
        resetToken,
        email: resetEmail || prev.email,
      }));
    }

    if (oauthError) {
      setStatus({
        type: "error",
        message: "Google sign-in failed. Please try again.",
      });
    }
  }, []);

  const handleFieldChange = (event) => {
    const { name, value } = event.target;
    setForm((prev) => ({ ...prev, [name]: value }));
  };

  const switchMode = (nextMode) => {
    setMode(nextMode);
    setStatus({ type: "", message: "" });
    setResetLink("");
  };

  const handleLogin = async (event) => {
    event.preventDefault();
    setLoading(true);
    setStatus({ type: "", message: "" });
    try {
      const data = await login({
        email: form.email,
        password: form.password,
      });
      setToken(data.token);
      sessionStorage.setItem("ui3gpp_force_new_chat", "1");
      localStorage.removeItem("ui3gpp_active_thread");
      if (onAuthSuccess) {
        onAuthSuccess(data.user);
      }
    } catch (error) {
      setStatus({ type: "error", message: error.message });
    } finally {
      setLoading(false);
    }
  };

  const handleSignup = async (event) => {
    event.preventDefault();
    setLoading(true);
    setStatus({ type: "", message: "" });
    if (form.password !== form.confirmPassword) {
      setStatus({ type: "error", message: "Passwords do not match." });
      setLoading(false);
      return;
    }
    try {
      const data = await signup({
        name: form.name,
        email: form.email,
        password: form.password,
      });
      setToken(data.token);
      sessionStorage.setItem("ui3gpp_force_new_chat", "1");
      localStorage.removeItem("ui3gpp_active_thread");
      if (onAuthSuccess) {
        onAuthSuccess(data.user);
      }
    } catch (error) {
      setStatus({ type: "error", message: error.message });
    } finally {
      setLoading(false);
    }
  };

  const handleForgot = async (event) => {
    event.preventDefault();
    setLoading(true);
    setStatus({ type: "", message: "" });
    setResetLink("");
    try {
      const data = await forgotPassword({ email: form.email });
      setStatus({
        type: "success",
        message: data.message || "Reset link sent if the account exists.",
      });
      if (data.reset_url) {
        setResetLink(data.reset_url);
      }
    } catch (error) {
      setStatus({ type: "error", message: error.message });
    } finally {
      setLoading(false);
    }
  };

  const handleReset = async (event) => {
    event.preventDefault();
    setLoading(true);
    setStatus({ type: "", message: "" });
    if (form.password !== form.confirmPassword) {
      setStatus({ type: "error", message: "Passwords do not match." });
      setLoading(false);
      return;
    }
    try {
      const data = await resetPassword({
        email: form.email,
        token: form.resetToken,
        password: form.password,
      });
      setStatus({
        type: "success",
        message: data.message || "Password updated. You can sign in now.",
      });
      setMode("login");
    } catch (error) {
      setStatus({ type: "error", message: error.message });
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleLogin = () => {
    const apiBaseUrl = getApiBaseUrl();
    const base = apiBaseUrl ? apiBaseUrl.replace(/\/$/, "") : "";
    window.location.href = `${base}/api/auth/google/login`;
  };

  const isBusy = loading || checking;

  return (
    <div className="login-page">
      <div className="login-shell">
        <section className="login-brand">
          <span className="login-badge">Wisdom Lab IITK </span>
          <h1>Welcome back.</h1>
          <p>
            Secure your research workflows, manage long-running pipelines, and
            keep every result in one place.
          </p>
          <div className="login-metrics">
            <div>
              <span className="metric-number">12k</span>
              <span className="metric-label">Spec pages indexed</span>
            </div>
            <div>
              <span className="metric-number">20.1s</span>
              <span className="metric-label">Average query latency</span>
            </div>
          </div>
        </section>
        <section className="login-panel">
          <div className="login-panel-header">
            <h2>
              {mode === "signup"
                ? "Create account"
                : mode === "forgot"
                ? "Reset access"
                : mode === "reset"
                ? "Choose a new password"
                : "Sign in"}
            </h2>
            <p>
              {mode === "signup"
                ? "Set up your workspace in under a minute."
                : mode === "forgot"
                ? "We will email you a reset link."
                : mode === "reset"
                ? "Use your reset link to pick a new password."
                : "Use your IITK credentials to continue."}
            </p>
          </div>

          <div className="login-tabs">
            <button
              type="button"
              className={`login-tab ${mode === "login" ? "active" : ""}`}
              onClick={() => switchMode("login")}
              disabled={isBusy}
            >
              Sign in
            </button>
            <button
              type="button"
              className={`login-tab ${mode === "signup" ? "active" : ""}`}
              onClick={() => switchMode("signup")}
              disabled={isBusy}
            >
              Sign up
            </button>
          </div>

          {status.message ? (
            <div className={`login-status ${status.type}`}>{status.message}</div>
          ) : null}
          {resetLink ? (
            <div className="login-status info">
              Dev reset link:{" "}
              <a href={resetLink} className="login-link">
                {resetLink}
              </a>
            </div>
          ) : null}

          {mode === "login" && (
            <form className="login-form" onSubmit={handleLogin}>
              <label className="login-field">
                <span>Email</span>
                <input
                  type="email"
                  name="email"
                  placeholder="cc-id@iitk.ac.in"
                  autoComplete="email"
                  value={form.email}
                  onChange={handleFieldChange}
                  required
                  disabled={isBusy}
                />
              </label>
              <label className="login-field">
                <span>Password</span>
                <div className="password-field">
                  <input
                    type={showPassword ? "text" : "password"}
                    name="password"
                    placeholder="********"
                    autoComplete="current-password"
                    value={form.password}
                    onChange={handleFieldChange}
                    required
                    disabled={isBusy}
                  />
                  <button
                    type="button"
                    className="password-toggle"
                    onClick={() => setShowPassword((prev) => !prev)}
                    disabled={isBusy}
                    aria-label={showPassword ? "Hide password" : "Show password"}
                  >
                    {showPassword ? (
                      <svg viewBox="0 0 24 24" aria-hidden="true">
                        <path
                          d="M4.2 4.2a1 1 0 0 1 1.4 0l14.2 14.2a1 1 0 1 1-1.4 1.4l-2.3-2.3A11.7 11.7 0 0 1 12 19C6 19 2.4 14.5 1.2 12a11.6 11.6 0 0 1 4-4.8L4.2 5.6a1 1 0 0 1 0-1.4Zm6 6a2 2 0 0 0 2.6 2.6l-2.6-2.6Zm3.1 6.5-1.7-1.7a4 4 0 0 1-5.3-5.3L4.8 8a9.4 9.4 0 0 0-2.4 4c1 2 4 6 9.6 6 1.2 0 2.3-.2 3.3-.5Zm-1.3-9.6 3.7 3.7a4 4 0 0 0-4-4c-.5 0-1 .1-1.4.3l1.7 1.7Z"
                          fill="currentColor"
                        />
                      </svg>
                    ) : (
                      <svg viewBox="0 0 24 24" aria-hidden="true">
                        <path
                          d="M12 5c6 0 9.6 4.5 10.8 7-1.2 2.5-4.8 7-10.8 7S2.4 14.5 1.2 12C2.4 9.5 6 5 12 5Zm0 2C7.5 7 4.6 10.3 3.4 12 4.6 13.7 7.5 17 12 17s7.4-3.3 8.6-5c-1.2-1.7-4.1-5-8.6-5Zm0 2.5a2.5 2.5 0 1 1 0 5 2.5 2.5 0 0 1 0-5Z"
                          fill="currentColor"
                        />
                      </svg>
                    )}
                  </button>
                </div>
              </label>
              <div className="login-actions">
                <button
                  type="button"
                  className="login-link-button"
                  onClick={() => switchMode("forgot")}
                  disabled={isBusy}
                >
                  Forgot password?
                </button>
              </div>
              <button className="login-submit" type="submit" disabled={isBusy}>
                {checking ? "Checking session..." : "Continue to workspace"}
              </button>
            </form>
          )}

          {mode === "signup" && (
            <form className="login-form" onSubmit={handleSignup}>
              <label className="login-field">
                <span>Name</span>
                <input
                  type="text"
                  name="name"
                  placeholder="Your name"
                  value={form.name}
                  onChange={handleFieldChange}
                  required
                  disabled={isBusy}
                />
              </label>
              <label className="login-field">
                <span>Email</span>
                <input
                  type="email"
                  name="email"
                  placeholder="cc-id@iitk.ac.in"
                  value={form.email}
                  onChange={handleFieldChange}
                  required
                  disabled={isBusy}
                />
              </label>
              <label className="login-field">
                <span>Password</span>
                <div className="password-field">
                  <input
                    type={showPassword ? "text" : "password"}
                    name="password"
                    placeholder="********"
                    value={form.password}
                    onChange={handleFieldChange}
                    required
                    disabled={isBusy}
                  />
                  <button
                    type="button"
                    className="password-toggle"
                    onClick={() => setShowPassword((prev) => !prev)}
                    disabled={isBusy}
                    aria-label={showPassword ? "Hide password" : "Show password"}
                  >
                    {showPassword ? (
                      <svg viewBox="0 0 24 24" aria-hidden="true">
                        <path
                          d="M4.2 4.2a1 1 0 0 1 1.4 0l14.2 14.2a1 1 0 1 1-1.4 1.4l-2.3-2.3A11.7 11.7 0 0 1 12 19C6 19 2.4 14.5 1.2 12a11.6 11.6 0 0 1 4-4.8L4.2 5.6a1 1 0 0 1 0-1.4Zm6 6a2 2 0 0 0 2.6 2.6l-2.6-2.6Zm3.1 6.5-1.7-1.7a4 4 0 0 1-5.3-5.3L4.8 8a9.4 9.4 0 0 0-2.4 4c1 2 4 6 9.6 6 1.2 0 2.3-.2 3.3-.5Zm-1.3-9.6 3.7 3.7a4 4 0 0 0-4-4c-.5 0-1 .1-1.4.3l1.7 1.7Z"
                          fill="currentColor"
                        />
                      </svg>
                    ) : (
                      <svg viewBox="0 0 24 24" aria-hidden="true">
                        <path
                          d="M12 5c6 0 9.6 4.5 10.8 7-1.2 2.5-4.8 7-10.8 7S2.4 14.5 1.2 12C2.4 9.5 6 5 12 5Zm0 2C7.5 7 4.6 10.3 3.4 12 4.6 13.7 7.5 17 12 17s7.4-3.3 8.6-5c-1.2-1.7-4.1-5-8.6-5Zm0 2.5a2.5 2.5 0 1 1 0 5 2.5 2.5 0 0 1 0-5Z"
                          fill="currentColor"
                        />
                      </svg>
                    )}
                  </button>
                </div>
              </label>
              <label className="login-field">
                <span>Confirm password</span>
                <div className="password-field">
                  <input
                    type={showConfirmPassword ? "text" : "password"}
                    name="confirmPassword"
                    placeholder="********"
                    value={form.confirmPassword}
                    onChange={handleFieldChange}
                    required
                    disabled={isBusy}
                  />
                  <button
                    type="button"
                    className="password-toggle"
                    onClick={() => setShowConfirmPassword((prev) => !prev)}
                    disabled={isBusy}
                    aria-label={showConfirmPassword ? "Hide password" : "Show password"}
                  >
                    {showConfirmPassword ? (
                      <svg viewBox="0 0 24 24" aria-hidden="true">
                        <path
                          d="M4.2 4.2a1 1 0 0 1 1.4 0l14.2 14.2a1 1 0 1 1-1.4 1.4l-2.3-2.3A11.7 11.7 0 0 1 12 19C6 19 2.4 14.5 1.2 12a11.6 11.6 0 0 1 4-4.8L4.2 5.6a1 1 0 0 1 0-1.4Zm6 6a2 2 0 0 0 2.6 2.6l-2.6-2.6Zm3.1 6.5-1.7-1.7a4 4 0 0 1-5.3-5.3L4.8 8a9.4 9.4 0 0 0-2.4 4c1 2 4 6 9.6 6 1.2 0 2.3-.2 3.3-.5Zm-1.3-9.6 3.7 3.7a4 4 0 0 0-4-4c-.5 0-1 .1-1.4.3l1.7 1.7Z"
                          fill="currentColor"
                        />
                      </svg>
                    ) : (
                      <svg viewBox="0 0 24 24" aria-hidden="true">
                        <path
                          d="M12 5c6 0 9.6 4.5 10.8 7-1.2 2.5-4.8 7-10.8 7S2.4 14.5 1.2 12C2.4 9.5 6 5 12 5Zm0 2C7.5 7 4.6 10.3 3.4 12 4.6 13.7 7.5 17 12 17s7.4-3.3 8.6-5c-1.2-1.7-4.1-5-8.6-5Zm0 2.5a2.5 2.5 0 1 1 0 5 2.5 2.5 0 0 1 0-5Z"
                          fill="currentColor"
                        />
                      </svg>
                    )}
                  </button>
                </div>
              </label>
              <button className="login-submit" type="submit" disabled={isBusy}>
                Create account
              </button>
            </form>
          )}

          {mode === "forgot" && (
            <form className="login-form" onSubmit={handleForgot}>
              <label className="login-field">
                <span>Email</span>
                <input
                  type="email"
                  name="email"
                  placeholder="cc-id@iitk.ac.in"
                  value={form.email}
                  onChange={handleFieldChange}
                  required
                  disabled={isBusy}
                />
              </label>
              <button className="login-submit" type="submit" disabled={isBusy}>
                Send reset link
              </button>
              <button
                type="button"
                className="login-link-button"
                onClick={() => switchMode("login")}
                disabled={isBusy}
              >
                Back to sign in
              </button>
            </form>
          )}

          {mode === "reset" && (
            <form className="login-form" onSubmit={handleReset}>
              <label className="login-field">
                <span>Email</span>
                <input
                  type="email"
                  name="email"
                  placeholder="cc-id@iitk.ac.in"
                  value={form.email}
                  onChange={handleFieldChange}
                  required
                  disabled={isBusy}
                />
              </label>
              <label className="login-field">
                <span>Reset token</span>
                <input
                  type="text"
                  name="resetToken"
                  placeholder="Paste your reset token"
                  value={form.resetToken}
                  onChange={handleFieldChange}
                  required
                  disabled={isBusy}
                />
              </label>
              <label className="login-field">
                <span>New password</span>
                <div className="password-field">
                  <input
                    type={showPassword ? "text" : "password"}
                    name="password"
                    placeholder="********"
                    value={form.password}
                    onChange={handleFieldChange}
                    required
                    disabled={isBusy}
                  />
                  <button
                    type="button"
                    className="password-toggle"
                    onClick={() => setShowPassword((prev) => !prev)}
                    disabled={isBusy}
                    aria-label={showPassword ? "Hide password" : "Show password"}
                  >
                    {showPassword ? (
                      <svg viewBox="0 0 24 24" aria-hidden="true">
                        <path
                          d="M4.2 4.2a1 1 0 0 1 1.4 0l14.2 14.2a1 1 0 1 1-1.4 1.4l-2.3-2.3A11.7 11.7 0 0 1 12 19C6 19 2.4 14.5 1.2 12a11.6 11.6 0 0 1 4-4.8L4.2 5.6a1 1 0 0 1 0-1.4Zm6 6a2 2 0 0 0 2.6 2.6l-2.6-2.6Zm3.1 6.5-1.7-1.7a4 4 0 0 1-5.3-5.3L4.8 8a9.4 9.4 0 0 0-2.4 4c1 2 4 6 9.6 6 1.2 0 2.3-.2 3.3-.5Zm-1.3-9.6 3.7 3.7a4 4 0 0 0-4-4c-.5 0-1 .1-1.4.3l1.7 1.7Z"
                          fill="currentColor"
                        />
                      </svg>
                    ) : (
                      <svg viewBox="0 0 24 24" aria-hidden="true">
                        <path
                          d="M12 5c6 0 9.6 4.5 10.8 7-1.2 2.5-4.8 7-10.8 7S2.4 14.5 1.2 12C2.4 9.5 6 5 12 5Zm0 2C7.5 7 4.6 10.3 3.4 12 4.6 13.7 7.5 17 12 17s7.4-3.3 8.6-5c-1.2-1.7-4.1-5-8.6-5Zm0 2.5a2.5 2.5 0 1 1 0 5 2.5 2.5 0 0 1 0-5Z"
                          fill="currentColor"
                        />
                      </svg>
                    )}
                  </button>
                </div>
              </label>
              <label className="login-field">
                <span>Confirm password</span>
                <div className="password-field">
                  <input
                    type={showConfirmPassword ? "text" : "password"}
                    name="confirmPassword"
                    placeholder="********"
                    value={form.confirmPassword}
                    onChange={handleFieldChange}
                    required
                    disabled={isBusy}
                  />
                  <button
                    type="button"
                    className="password-toggle"
                    onClick={() => setShowConfirmPassword((prev) => !prev)}
                    disabled={isBusy}
                    aria-label={showConfirmPassword ? "Hide password" : "Show password"}
                  >
                    {showConfirmPassword ? (
                      <svg viewBox="0 0 24 24" aria-hidden="true">
                        <path
                          d="M4.2 4.2a1 1 0 0 1 1.4 0l14.2 14.2a1 1 0 1 1-1.4 1.4l-2.3-2.3A11.7 11.7 0 0 1 12 19C6 19 2.4 14.5 1.2 12a11.6 11.6 0 0 1 4-4.8L4.2 5.6a1 1 0 0 1 0-1.4Zm6 6a2 2 0 0 0 2.6 2.6l-2.6-2.6Zm3.1 6.5-1.7-1.7a4 4 0 0 1-5.3-5.3L4.8 8a9.4 9.4 0 0 0-2.4 4c1 2 4 6 9.6 6 1.2 0 2.3-.2 3.3-.5Zm-1.3-9.6 3.7 3.7a4 4 0 0 0-4-4c-.5 0-1 .1-1.4.3l1.7 1.7Z"
                          fill="currentColor"
                        />
                      </svg>
                    ) : (
                      <svg viewBox="0 0 24 24" aria-hidden="true">
                        <path
                          d="M12 5c6 0 9.6 4.5 10.8 7-1.2 2.5-4.8 7-10.8 7S2.4 14.5 1.2 12C2.4 9.5 6 5 12 5Zm0 2C7.5 7 4.6 10.3 3.4 12 4.6 13.7 7.5 17 12 17s7.4-3.3 8.6-5c-1.2-1.7-4.1-5-8.6-5Zm0 2.5a2.5 2.5 0 1 1 0 5 2.5 2.5 0 0 1 0-5Z"
                          fill="currentColor"
                        />
                      </svg>
                    )}
                  </button>
                </div>
              </label>
              <button className="login-submit" type="submit" disabled={isBusy}>
                Update password
              </button>
              <button
                type="button"
                className="login-link-button"
                onClick={() => switchMode("login")}
                disabled={isBusy}
              >
                Back to sign in
              </button>
            </form>
          )}

          {(mode === "login" || mode === "signup") && (
            <>
              <div className="login-divider">
                <span>or continue with</span>
              </div>
              <button
                type="button"
                className="login-google"
                onClick={handleGoogleLogin}
                disabled={isBusy}
              >
                <span className="login-google-mark">G</span>
                Continue with Google
              </button>
            </>
          )}
        </section>
      </div>
    </div>
  );
};

export default Login;
