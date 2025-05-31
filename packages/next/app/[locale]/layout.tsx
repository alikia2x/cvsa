import type { Metadata } from "next";
import "./globals.css";
import React from "react";
import { routing } from "@/i18n/routing";
import { NextIntlClientProvider, hasLocale } from "next-intl";
import { notFound } from "next/navigation";

export const metadata: Metadata = {
	title: "中V档案馆"
};

export default async function RootLayout({
	children,
	params
}: Readonly<{
	children: React.ReactNode;
	params: Promise<{ locale: string }>;
}>) {
	const { locale } = await params;
	if (!hasLocale(routing.locales, locale)) {
		notFound();
	}
	return (
		<html lang="zh-CN">
			<head>
				<meta charSet="UTF-8" />
				<meta name="viewport" content="width=device-width, initial-scale=1.0" />
				<title>中V档案馆</title>
			</head>
			<body className="min-h-screen flex flex-col">
				<NextIntlClientProvider>
					{children}
					<div id="portal-root"></div>
				</NextIntlClientProvider>
			</body>
		</html>
	);
}
