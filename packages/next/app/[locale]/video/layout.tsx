import { Header } from "@/components/shell/Header";
import { getCurrentUser } from "@/lib/userAuth";
import React from "react";

export default async function RootLayout({
	children
}: Readonly<{
	children: React.ReactNode;
}>) {
	const user = await getCurrentUser();
	return (
		<>
			<Header user={user} />
			{children}
		</>
	);
}
