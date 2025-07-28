import { MetaProvider } from "@solidjs/meta";
import { Router } from "@solidjs/router";
import { FileRoutes } from "@solidjs/start/router";
import { onMount, Suspense } from "solid-js";
import "./app.css";
import "@m3-components/solid/index.css";
import { setActiveTab } from "./components/shell/Navigation";
import { minimatch } from "minimatch";

const tabMap = {
	"/": 0,
	"/song*": 1,
	"/song/**/*": 1,
	"/albums": 2,
	"/album/**/*": 2
};

export default function App() {
	onMount(() => {
		const path = window.location.pathname;
		for (const [key, value] of Object.entries(tabMap)) {
			if (!minimatch(path, key)) continue;
			setActiveTab(value);
			break;
		}
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
