import { MetaProvider } from "@solidjs/meta";
import { Router } from "@solidjs/router";
import { FileRoutes } from "@solidjs/start/router";
import { onMount, Suspense } from "solid-js";
import "./app.css";
import "@m3-components/solid/index.css";
import { setActiveTab, tabMap } from "./components/shell/Navigation";
import { minimatch } from "minimatch";

export const refreshTab = (path: string) => {
	for (const [key, value] of Object.entries(tabMap)) {
		if (!minimatch(path, key)) continue;
		setActiveTab(value);
		break;
	}
}

export default function App() {
	onMount(() => {
		refreshTab(location.pathname);
		window.addEventListener('popstate', (event) => {
			refreshTab(location.pathname);
		});
	});

	return (
		<Router
			root={(props) => (
				<MetaProvider>
					<Suspense>{props.children}</Suspense>
				</MetaProvider>
			)}
		>
			<FileRoutes />
		</Router>
	);
}
