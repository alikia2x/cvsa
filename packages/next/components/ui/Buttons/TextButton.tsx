"use client";

import useRipple from "@/components/utils/useRipple";

interface TextButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
	size?: "xs" | "s" | "m" | "l" | "xl";
	shape?: "round" | "square";
	children?: React.ReactNode;
	ripple?: boolean;
	ref?: React.Ref<HTMLButtonElement>;
}

export const TextButton = ({
	children,
	size = "s",
	shape = "round",
	className = "",
	disabled,
	ref,
	ripple = true,
	...rest
}: TextButtonProps) => {
	let sizeClasses = "text-sm leading-5 h-10 px-4";
	let shapeClasses = "rounded-full";

	if (size === "m") {
		sizeClasses = "text-base leading-6 h-14 px-6";
		shapeClasses = shape === "round" ? "rounded-full" : "rounded-2xl";
	}

	const { onMouseDown, onTouchStart } = useRipple({ ripple });

	return (
		<button
			className={`text-primary dark:text-dark-primary duration-150 select-none
				flex items-center justify-center relative overflow-hidden
				disabled:text-on-surface/40 disabled:dark:text-dark-on-surface/40
				${sizeClasses} ${shapeClasses} ${className}`}
			{...rest}
			onMouseDown={onMouseDown}
			onTouchStart={onTouchStart}
			disabled={disabled}
			ref={ref}
		>
			<div
				className={`absolute w-full h-full enabled:hover:bg-primary/10 enabled:dark:hover:bg-dark-primary/10
				${disabled && "bg-on-surface/10 dark:bg-dark-on-surface/10"}
				left-0 top-0`}
			></div>
			{children}
		</button>
	);
};
