import React, { useState } from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import AuthPage from "./AuthPage";

function Root() {
  const [isLogged, setIsLogged] = useState(false); // ← luôn bắt đầu là false

  return isLogged ? (
    <App />
  ) : (
    <AuthPage onLogin={() => setIsLogged(true)} />
  );
}

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<Root />);