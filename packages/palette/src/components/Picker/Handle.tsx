import React, { useRef } from "react";
import { type Oklch } from "culori";

export const Handle = ({
	pos,
	color,
	onChange,
	maxValue,
	onTouchMove,
	onTouchStart
}: {
	pos: number;
	color: Omit<Oklch, "mode">;
	onChange: (value: number) => void;
	maxValue: number;
	onTouchMove?: (e: React.TouchEvent) => void;
	onTouchStart?: (e: React.TouchEvent) => void;
}) => {
	const handleRef = useRef<HTMLDivElement>(null);
	const isDragging = useRef(false);
	const isTouching = useRef(false);

	const getValueFromPosition = (clientX: number) => {
		const sliderRect = handleRef.current?.closest(".relative")?.getBoundingClientRect();
		if (!sliderRect) return 0;

		const x = clientX - sliderRect.left;
		const percentage = Math.max(0, Math.min(1, x / sliderRect.width));
		return percentage * maxValue;
	};

	const handleMouseDown = () => {
		isDragging.current = true;
		document.addEventListener("mousemove", handleMouseMove);
		document.addEventListener("mouseup", handleMouseUp);
	};

	const handleMouseMove = (e: MouseEvent) => {
		if (!isDragging.current) return;

		const value = getValueFromPosition(e.clientX);
		onChange(value);
	};

	const handleMouseUp = () => {
		isDragging.current = false;
		document.removeEventListener("mousemove", handleMouseMove);
		document.removeEventListener("mouseup", handleMouseUp);
	};

	const handleTouchStart = (e: React.TouchEvent) => {
		isTouching.current = true;

		const touch = e.touches[0];
		if (touch) {
			const value = getValueFromPosition(touch.clientX);
			onChange(value);
		}

		onTouchStart?.(e);
	};

	const handleTouchMove = (e: React.TouchEvent) => {
		if (!isTouching.current) return;

		const touch = e.touches[0];
		if (touch) {
			const value = getValueFromPosition(touch.clientX);
			onChange(value);
		}

		onTouchMove?.(e);
	};

	const handleTouchEnd = () => {
		isTouching.current = false;
	};

	return (
		<div
			ref={handleRef}
			className="absolute z-5 top-full left-0 size-7 border-white border-width-3 
             shadow-[0px_0px_7px_2px_rgba(0,0,0,0.35)] cursor-grab active:cursor-grabbing
             touch-none select-none"
			style={{
				left: `${pos}%`,
				backgroundColor: `oklch(${color.l} ${color.c} ${color.h})`,
				transform: "translateY(-50%) translateX(-50%) rotate(45deg)"
			}}
			onMouseDown={handleMouseDown}
			onTouchStart={handleTouchStart}
			onTouchMove={handleTouchMove}
			onTouchEnd={handleTouchEnd}
			onTouchCancel={handleTouchEnd}
		/>
	);
};
