import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import { SpeedInsights } from "@vercel/speed-insights/react";
import App from "./App.tsx";
import { ThemeProvider } from "./ThemeContext.tsx";

createRoot(document.getElementById("root")!).render(
	<StrictMode>
		<ThemeProvider>
			<App />
			<SpeedInsights />
		</ThemeProvider>
	</StrictMode>
);
