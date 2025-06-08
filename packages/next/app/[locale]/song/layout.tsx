import HeaderServer from "@/components/shell/HeaderServer";
import React from "react";

export default async function RootLayout({
	children
}: Readonly<{
	children: React.ReactNode;
}>) {
	return (
		<>
			<HeaderServer />
			{children}
		</>
	);
}
