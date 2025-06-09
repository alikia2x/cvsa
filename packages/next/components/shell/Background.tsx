"use client";

import { backgroundURLAtom } from "@/lib/state/background";
import { useAtom, useSetAtom } from "jotai";
import { useEffect } from "react";
import { usePathname } from "next/navigation";

export function Background() {
	const pathname = usePathname();
	const [url, setURL] = useAtom(backgroundURLAtom);
	useEffect(() => {
		setURL(null);
	}, [pathname]);
	if (!url) {
		return <></>;
	}
	return <img className="top-0 left-0 h-screen w-screen fixed object-cover" style={{ zIndex: -1 }} src={url} />;
}

export function BackgroundDelegate({ url }: { url: string }) {
	const setBG = useSetAtom(backgroundURLAtom);
	useEffect(() => {
		setBG(url);
	}, []);
	return <></>;
}
