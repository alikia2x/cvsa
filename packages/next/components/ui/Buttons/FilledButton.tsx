import useRipple from "@/components/utils/useRipple";

interface FilledButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
	size?: "xs" | "s" | "m" | "l" | "xl";
	shape?: "round" | "square";
	children?: React.ReactNode;
	ripple?: boolean;
}

export const FilledButton = ({
	children,
	size = "s",
	shape = "round",
	className,
	ripple = true,
	...rest
}: FilledButtonProps) => {
	let sizeClasses = "text-sm leading-5 h-10 px-4";
	let shapeClasses = "rounded-full";

	if (size === "m") {
		sizeClasses = "text-base leading-6 h-14 px-6";
		shapeClasses = shape === "round" ? "rounded-full" : "rounded-2xl";
	}

	const { onMouseDown, onTouchStart } = useRipple({ ripple });

	return (
		<button
			className={`bg-primary dark:bg-dark-primary text-on-primary dark:text-dark-on-primary duration-150 select-none
				flex items-center justify-center relative overflow-hidden
				${sizeClasses} ${shapeClasses} ${className}`}
			{...rest}
			onMouseDown={onMouseDown}
			onTouchStart={onTouchStart}
		>
			<div className="absolute w-full h-full hover:bg-on-surface-variant/10"></div>
			{children}
		</button>
	);
};
