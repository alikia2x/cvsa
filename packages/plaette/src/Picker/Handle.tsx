import React, { useRef } from "react";
import { type Oklch } from "culori";

export const Handle = ({
	pos,
	color,
	onChange,
	maxValue
}: {
	pos: number;
	color: Omit<Oklch, "mode">;
	onChange: (value: number) => void;
	maxValue: number;
}) => {
	const handleRef = useRef<HTMLDivElement>(null);
	const isDragging = useRef(false);

	const handleMouseDown = (e: React.MouseEvent) => {
		e.preventDefault();
		isDragging.current = true;
		document.addEventListener("mousemove", handleMouseMove);
		document.addEventListener("mouseup", handleMouseUp);
	};

	const handleMouseMove = (e: MouseEvent) => {
		if (!isDragging.current || !handleRef.current) return;

		const sliderRect = handleRef.current.closest(".relative")?.getBoundingClientRect();
		if (!sliderRect) return;

		const x = e.clientX - sliderRect.left;
		const percentage = Math.max(0, Math.min(100, (x / sliderRect.width) * 100));
		const value = (percentage / 100) * maxValue;

		onChange(value);
	};

	const handleMouseUp = () => {
		isDragging.current = false;
		document.removeEventListener("mousemove", handleMouseMove);
		document.removeEventListener("mouseup", handleMouseUp);
	};

	return (
		<div
			ref={handleRef}
			className="absolute z-5 top-full left-0 size-7 border-white border-width-3 
             shadow-[0px_0px_7px_2px_rgba(0,0,0,0.35)] cursor-grab active:cursor-grabbing"
			style={{
				left: `${pos}%`,
				backgroundColor: `oklch(${color.l} ${color.c} ${color.h})`,
				transform: "translateY(-50%) translateX(-50%) rotate(45deg)"
			}}
			onMouseDown={handleMouseDown}
		/>
	);
};
