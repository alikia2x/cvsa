import type { Metadata } from "next";
import "./global.css";
import React from "react";
import { routing } from "@/i18n/routing";
import { hasLocale } from "next-intl";
import { notFound } from "next/navigation";
import { Background } from "@/components/shell/Background";

export const metadata: Metadata = {
	title: "中 V 档案馆"
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
		<>
			<Background />
			{children}
		</>
	);
}
