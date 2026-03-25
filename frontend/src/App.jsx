import { useCallback, useEffect, useState } from "react";
import Login from "./components/login/Login";
import Main from "./components/main/Main";
import Sidebar from "./components/sidebar/Sidebar";
import GraphBar from "./components/graphbar/Graphbar";
import ResultsBar from "./components/resultsbar/ResultsBar";
import ContextProvider from "./context/Context";
import { getCurrentUser } from "./services/auth";
import { clearToken, getToken, setToken } from "./services/authToken";

const App = () => {
  const [authState, setAuthState] = useState({
    user: null,
    checking: true,
  });
  const [isGraphOpen, setIsGraphOpen] = useState(false);
  const [isResultsOpen, setIsResultsOpen] = useState(false);
  const [graphDrawerWidth, setGraphDrawerWidth] = useState(420);
  const [resultsDrawerWidth, setResultsDrawerWidth] = useState(420);
  const [graphUserResized, setGraphUserResized] = useState(false);
  const [resultsUserResized, setResultsUserResized] = useState(false);
  const [isCompactLayout, setIsCompactLayout] = useState(false);

  const MIN_MAIN_WIDTH = 280;

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const oauthToken = params.get("token");
    if (oauthToken) {
      setToken(oauthToken);
      sessionStorage.setItem("ui3gpp_force_new_chat", "1");
      localStorage.removeItem("ui3gpp_active_thread");
      params.delete("token");
      params.delete("provider");
      const newQuery = params.toString();
      const newUrl = newQuery
        ? `${window.location.pathname}?${newQuery}`
        : window.location.pathname;
      window.history.replaceState({}, document.title, newUrl);
    }

    const existingToken = getToken();
    if (!existingToken) {
      setAuthState({ user: null, checking: false });
      return;
    }

    getCurrentUser()
      .then((user) => setAuthState({ user, checking: false }))
      .catch(() => {
        clearToken();
        setAuthState({ user: null, checking: false });
      });
  }, []);

  const clampWidth = (value, min, max) => Math.min(Math.max(value, min), max);

  const getDefaultDrawerWidth = (width) => {
    if (width < 640) {
      return width;
    }
    if (width < 900) {
      return 320;
    }
    if (width < 1200) {
      return 360;
    }
    return 420;
  };

  const getDrawerBounds = (width, otherOpenWidth) => {
    if (width < 640) {
      return { min: width, max: width };
    }
    const min = 320;
    const max = Math.max(min, width - MIN_MAIN_WIDTH - otherOpenWidth);
    return { min, max };
  };

  const updateLayout = useCallback(() => {
    const width = window.innerWidth;
    const compact = width < 1200;
    const defaultWidth = getDefaultDrawerWidth(width);
    const graphOtherWidth = !compact && isResultsOpen ? resultsDrawerWidth : 0;
    const resultsOtherWidth = !compact && isGraphOpen ? graphDrawerWidth : 0;
    const graphBounds = getDrawerBounds(width, graphOtherWidth);
    const resultsBounds = getDrawerBounds(width, resultsOtherWidth);

    setIsCompactLayout(compact);
    setGraphDrawerWidth((prev) =>
      clampWidth(graphUserResized ? prev : defaultWidth, graphBounds.min, graphBounds.max)
    );
    setResultsDrawerWidth((prev) =>
      clampWidth(resultsUserResized ? prev : defaultWidth, resultsBounds.min, resultsBounds.max)
    );
  }, [
    graphDrawerWidth,
    resultsDrawerWidth,
    graphUserResized,
    resultsUserResized,
    isGraphOpen,
    isResultsOpen,
  ]);

  useEffect(() => {
    updateLayout();
    window.addEventListener("resize", updateLayout);
    return () => window.removeEventListener("resize", updateLayout);
  }, [updateLayout]);

  const handleGraphResize = useCallback(
    (nextWidth) => {
      const width = window.innerWidth;
      const compact = width < 1200;
      const otherWidth = !compact && isResultsOpen ? resultsDrawerWidth : 0;
      const bounds = getDrawerBounds(width, otherWidth);
      setGraphDrawerWidth(clampWidth(nextWidth, bounds.min, bounds.max));
      setGraphUserResized(true);
    },
    [isResultsOpen, resultsDrawerWidth]
  );

  const handleResultsResize = useCallback(
    (nextWidth) => {
      const width = window.innerWidth;
      const compact = width < 1200;
      const otherWidth = !compact && isGraphOpen ? graphDrawerWidth : 0;
      const bounds = getDrawerBounds(width, otherWidth);
      setResultsDrawerWidth(clampWidth(nextWidth, bounds.min, bounds.max));
      setResultsUserResized(true);
    },
    [isGraphOpen, graphDrawerWidth]
  );

  if (authState.checking) {
    return <Login checking />;
  }

  if (!authState.user) {
    return <Login onAuthSuccess={(user) => setAuthState({ user, checking: false })} />;
  }

  const handleLogout = () => {
    clearToken();
    setAuthState({ user: null, checking: false });
  };

  return (
    <ContextProvider>
      <Sidebar onLogout={handleLogout} />
      <Main user={authState.user} />
      <ResultsBar
        isOpen={isResultsOpen}
        onToggle={setIsResultsOpen}
        offset={!isCompactLayout && isGraphOpen ? graphDrawerWidth : 0}
        drawerWidth={resultsDrawerWidth}
        onResize={handleResultsResize}
      />
      <GraphBar
        isOpen={isGraphOpen}
        onToggle={setIsGraphOpen}
        offset={0}
        drawerWidth={graphDrawerWidth}
        onResize={handleGraphResize}
      />
      {/* <Popup /> */}
    </ContextProvider>
  );
};

export default App;
