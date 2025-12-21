interface SwitchProps {
	checked: boolean;
	onChange: (checked: boolean) => void;
	disabled?: boolean;
	className?: string;
	label?: string;
}

export const Switch: React.FC<SwitchProps> = ({
	checked,
	onChange,
	disabled = false,
	className = "",
	label,
}) => {
	const handleToggle = () => {
		if (!disabled) {
			onChange(!checked);
		}
	};

	return (
		<div className={`flex items-center gap-3 ${className}`}>
			{/* Switch Container */}
			<button
				type="button"
				onClick={handleToggle}
				disabled={disabled}
				className={`relative flex items-center justify-center w-12 h-6 rounded-full transition-all duration-300
                     focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 ${
							disabled
								? "cursor-not-allowed opacity-50"
								: "cursor-pointer hover:scale-105"
						} ${checked ? "bg-green-500" : "bg-zinc-300 dark:bg-zinc-600"}`}
				aria-checked={checked}
				aria-disabled={disabled}
			>
				{/* Switch Thumb */}
				<span
					className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full transition-transform duration-300 transform ${
						checked ? "translate-x-6" : ""
					} ${disabled && "opacity-50"}`}
					aria-hidden="true"
				/>
			</button>

			{/* Optional Label */}
			{label && (
				<label
					className={`text-sm font-medium ${
						disabled
							? "text-gray-400 dark:text-gray-500"
							: "text-gray-800 dark:text-gray-200"
					}`}
					onClick={handleToggle}
				>
					{label}
				</label>
			)}
		</div>
	);
};
