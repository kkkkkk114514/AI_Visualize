import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";

import "allotment/dist/style.css";
import "@xyflow/react/dist/style.css";
import "./i18n";
import "./styles/global.css";

import App from "./App";

const container = document.getElementById("root");
if (!container) {
  throw new Error("root container missing");
}

createRoot(container).render(
  <StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </StrictMode>,
);
