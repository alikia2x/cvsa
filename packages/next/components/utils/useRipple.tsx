import "./ripple.css";

import { useCallback, useRef } from "react";

interface UseRippleOptions {
	ripple?: boolean;
}

const useRipple = ({ ripple = true }: UseRippleOptions = {}) => {
	const isTouchEventRef = useRef(false);

	const resetTouchFlag = useCallback(() => {
		isTouchEventRef.current = false;
	}, []);

	const handleMouseDown = useCallback(
		(event: React.MouseEvent<HTMLButtonElement>) => {
			if (!ripple) return;

			if (isTouchEventRef.current) {
				// Skip mouse event if this was a touch interaction
				return;
			}

			// Proceed with mouse ripple
			createRippleEffect(event);
		},
		[ripple]
	);

	const handleTouchStart = useCallback(
		(event: React.TouchEvent<HTMLButtonElement>) => {
			if (!ripple) return;

			isTouchEventRef.current = true;

			// Proceed with touch ripple
			createRippleEffect(event);
		},
		[ripple, resetTouchFlag]
	);

	const handleTouchEnd = useCallback(
		(_event: React.TouchEvent<HTMLButtonElement>) => {
			if (!ripple) return;

			isTouchEventRef.current = false;
		},
		[ripple, resetTouchFlag]
	);

	const createRippleEffect = (event: React.MouseEvent<HTMLButtonElement> | React.TouchEvent<HTMLButtonElement>) => {
		const target = event.currentTarget;
		const rect = target.getBoundingClientRect();

		let x = 0;
		let y = 0;

		if ("touches" in event) {
			x = event.touches[0].clientX - rect.left;
			y = event.touches[0].clientY - rect.top;
		} else {
			x = event.clientX - rect.left;
			y = event.clientY - rect.top;
		}

		const diameter = Math.max(rect.width, rect.height);
		const rippleElement = document.createElement("span");

		rippleElement.style.width = `${diameter}px`;
		rippleElement.style.height = `${diameter}px`;
		rippleElement.style.left = `${x - diameter / 2}px`;
		rippleElement.style.top = `${y - diameter / 2}px`;
		rippleElement.classList.add("ripple");

		target.appendChild(rippleElement);

		rippleElement.addEventListener("animationend", () => {
			rippleElement.remove();
		});
	};

	return {
		onMouseDown: ripple ? handleMouseDown : undefined,
		onTouchStart: ripple ? handleTouchStart : undefined,
		ontouchend: ripple ? handleTouchEnd : undefined
	};
};

export default useRipple;
