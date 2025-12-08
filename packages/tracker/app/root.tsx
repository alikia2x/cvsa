import {
	isRouteErrorResponse,
	Links,
	Meta,
	Outlet,
	Scripts,
	ScrollRestoration
} from "react-router";

import type { Route } from "./+types/root";
import "./app.css";
import { Toaster } from "@/components/ui/sonner";

export const links: Route.LinksFunction = () => [
	{ rel: "preconnect", href: "https://fonts.googleapis.com" },
	{
		rel: "preconnect",
		href: "https://fonts.gstatic.com",
		crossOrigin: "anonymous"
	},
	{
		rel: "stylesheet",
		href: "https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&display=swap"
	}
];

export function Layout({ children }: { children: React.ReactNode }) {
	return (
		<html lang="en">
			<head>
				<meta charSet="utf-8" />
				<meta name="viewport" content="width=device-width, initial-scale=1" />
				<Meta />
				<Links />
			</head>
			<body>
				{children}
				<ScrollRestoration />
				<Scripts />
				<Toaster />
			</body>
		</html>
	);
}

export default function App() {
	return <Outlet />;
}

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
	let message = "Oops!";
	let details = "An unexpected error occurred.";
	let stack: string | undefined;

	if (isRouteErrorResponse(error)) {
		message = error.status.toString();
		details = error.status === 404 ? "The requested page could not be found." : error.data;
	} else if (import.meta.env.DEV && error && error instanceof Error) {
		details = error.message;
		stack = error.stack;
	}

	return (
		<div className="w-full min-h-screen flex flex-col items-center justify-center">
			<div className="mb-8 text-center">
				<h1 className="text-6xl font-bold text-foreground mb-4">{message}</h1>
				<p className="md:text-xl text-muted-foreground mb-8">{details}</p>
				<div className="flex justify-center gap-4">
					<a
						href="/"
						className="inline-flex items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground shadow transition-colors hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
					>
						Go Home
					</a>
					<button
						onClick={() => window.location.reload()}
						className="inline-flex items-center justify-center rounded-md border border-input bg-background px-4 py-2 text-sm font-medium shadow-sm transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
					>
						Reload
					</button>
				</div>
			</div>
			{stack && (
				<div className="mt-8 text-left">
					<h3 className="text-lg font-semibold mb-2">Stack Trace:</h3>
					<pre className="w-full p-4 overflow-x-auto bg-muted rounded-md text-sm">
						<code>{stack}</code>
					</pre>
				</div>
			)}
		</div>
	);
}
