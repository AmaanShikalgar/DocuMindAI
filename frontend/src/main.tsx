import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import Chat from "./Chat";
import Dashboard from "./Dashboard";
import "@fontsource/ibm-plex-sans/400.css";
import "@fontsource/ibm-plex-sans/500.css";
import "@fontsource/ibm-plex-mono/400.css";
import "./styles.css";

function useHash() {
  const [hash, setHash] = useState(location.hash);
  useEffect(() => {
    const on = () => setHash(location.hash);
    addEventListener("hashchange", on);
    return () => removeEventListener("hashchange", on);
  }, []);
  return hash;
}

function App() {
  const page = useHash() === "#/dashboard" ? "dashboard" : "chat";
  return (
    <div className="app">
      <header className="topbar">
        <a className="brand" href="#/">
          DocuMind
        </a>
        <nav>
          <a href="#/" aria-current={page === "chat" ? "page" : undefined}>
            Chat
          </a>
          <a href="#/dashboard" aria-current={page === "dashboard" ? "page" : undefined}>
            Dashboard
          </a>
        </nav>
      </header>
      {page === "chat" ? <Chat /> : <Dashboard />}
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
