import type { Metadata } from "next";
import "./globals.css";
import React from "react";

export const metadata: Metadata = {
	title: "中V档案馆"
};

export default function RootLayout({
	children
}: Readonly<{
	children: React.ReactNode;
}>) {
	return (
		<html lang="zh-CN">
			<head>
				<meta charSet="UTF-8" />
				<meta name="viewport" content="width=device-width, initial-scale=1.0" />
				<title>中V档案馆</title>
			</head>
			<body className="min-h-screen flex flex-col">
				{children}
				<div id="portal-root"></div>
			</body>
		</html>
	);
}
