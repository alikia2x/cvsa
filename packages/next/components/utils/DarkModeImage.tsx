"use client";

import { useState, useEffect } from "react";
import Image from "next/image";

interface Props {
	lightSrc: string;
	darkSrc: string;
	alt?: string;
	className?: string;
	width?: number;
	height?: number;
}

const DarkModeImage = ({ lightSrc, darkSrc, alt = "", className = "", width, height }: Props) => {
	const [isDarkMode, setIsDarkMode] = useState(false);
	const [currentSrc, setCurrentSrc] = useState(lightSrc);
	const [opacity, setOpacity] = useState(0);

	useEffect(() => {
		const handleDarkModeChange = (event: MediaQueryListEvent) => {
			setIsDarkMode(event.matches);
			setCurrentSrc(event.matches ? darkSrc : lightSrc);
			setOpacity(1);
		};

		const darkModeMediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
		setIsDarkMode(darkModeMediaQuery.matches);
		setCurrentSrc(darkModeMediaQuery.matches ? darkSrc : lightSrc);
		setOpacity(1);

		darkModeMediaQuery.addEventListener("change", handleDarkModeChange);

		return () => {
			darkModeMediaQuery.removeEventListener("change", handleDarkModeChange);
		};
	}, [darkSrc, lightSrc]);

	useEffect(() => {
		setCurrentSrc(isDarkMode ? darkSrc : lightSrc);
	}, [isDarkMode, darkSrc, lightSrc]);

	return (
		<Image
			src={currentSrc}
			alt={alt}
			className={className}
			style={{ opacity: opacity }}
			width={width}
			height={height}
		/>
	);
};

export default DarkModeImage;
