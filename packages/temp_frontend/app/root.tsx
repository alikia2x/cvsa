import {
	isRouteErrorResponse,
	Links,
	Meta,
	Outlet,
	Scripts,
	ScrollRestoration,
} from "react-router";
import { Toaster } from "@/components/ui/sonner";
import type { Route } from "./+types/root";
import "./app.css";
import { ErrorPage as ErrPage } from "./components/Error";

export const links: Route.LinksFunction = () => [
	{ href: "https://fonts.googleapis.com", rel: "preconnect" },
	{
		crossOrigin: "anonymous",
		href: "https://fonts.gstatic.com",
		rel: "preconnect",
	},
	{
		as: "style",
		href: "https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&display=swap",
		rel: "preload",
	},
	{
		href: "https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&display=swap",
		media: "print",
		onload: "this.media='all'",
		rel: "stylesheet",
	},
];

export function Layout({ children }: { children: React.ReactNode }) {
	return (
		<html lang="zh-CN">
			<head>
				<meta charSet="utf-8" />
				<meta name="viewport" content="width=device-width, initial-scale=1" />
				<link rel="icon" type="image/x-icon" href="/favicon.ico" />
				<link rel="manifest" href="/site.webmanifest" />
				<title>中V档案馆</title>
				<Meta />
				<Links />
			</head>
			<body className="overflow-x-hidden">
				{children}
				<ScrollRestoration />
				<Scripts />
				<Toaster position="top-center" />
			</body>
		</html>
	);
}

export default function App() {
	return <Outlet />;
}

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
	let status = 0;
	let details = "出错了！";

	if (isRouteErrorResponse(error)) {
		status = error.status;
		details = error.status === 404 ? "找不到页面" : error.statusText || details;
	} else if (import.meta.env.DEV && error && error instanceof Error) {
		details = error.message;
	}

	return (
		<ErrPage
			error={{
				status: status || 500,
				value: { message: details },
			}}
		/>
	);
}
