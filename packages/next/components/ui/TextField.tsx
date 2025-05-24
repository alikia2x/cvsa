"use client";

import React, { useState } from "react";

interface InputProps extends React.HTMLAttributes<HTMLDivElement> {
	labelText?: string;
	type?: React.HTMLInputTypeAttribute;
	inputText?: string;
	onInputTextChange?: (value: string) => void;
	maxChar?: number;
	supportingText?: string;
}

const TextField: React.FC<InputProps> = ({
	labelText = "",
	type = "text",
	inputText: initialInputText = "",
	onInputTextChange,
	maxChar,
	supportingText,
	...rest
}) => {
	const [focus, setFocus] = useState(false);
	const [inputText, setInputText] = useState(initialInputText);

	const handleValueChange = (event: React.ChangeEvent<HTMLInputElement>) => {
		const { value } = event.target;
		setInputText(value);
		onInputTextChange?.(value);
	};

	return (
		<div {...rest}>
			<div className="relative h-14 px-4">
				<div className="absolute flex top-0 left-0 h-full w-full">
					<div
						className={`w-3 rounded-l-sm border-outline dark:border-dark-outline
				${focus ? "border-primary dark:border-dark-primary border-l-2 border-y-2" : "border-l-[1px] border-y-[1px] "}
			`}
					></div>

					<div
						className={`px-1 border-outline dark:border-dark-outline transition-none
				${!focus && !inputText ? "border-y-[1px]" : ""}
				${!focus && inputText ? "border-y-[1px] border-t-0" : ""}
				${focus ? "border-primary dark:border-dark-primary border-y-2 border-t-0" : ""}
			`}
					>
						<span
							className={`
					relative leading-6 text-base text-on-surface-variant dark:text-dark-on-surface-variant duration-150
					${focus || inputText ? "-top-3 text-xs leading-4" : "top-4"}
					${focus ? "text-primary dark:text-dark-primary" : ""}
				`}
						>
							{labelText}
						</span>
					</div>

					<div
						className={`flex-grow rounded-r-sm border-outline dark:border-dark-outline
					${focus ? "border-primary dark:border-dark-primary border-r-2 border-y-2" : "border-r-[1px] border-y-[1px] "}
			`}
					></div>
				</div>

				<input
					className="relative focus:outline-none h-full w-full"
					onFocus={() => setFocus(true)}
					onBlur={() => setFocus(false)}
					onChange={handleValueChange}
					type={type}
					value={inputText}
				/>
			</div>
			{(supportingText || maxChar) && (
				<div
					className="w-full relative mt-1 text-on-surface-variant dark:text-dark-on-surface-variant
		 	text-xs leading-4 h-4"
				>
					{supportingText && <span className="absolute left-4">{supportingText}</span>}
					{maxChar && (
						<span className="absolute right-4">
							{inputText.length}/{maxChar}
						</span>
					)}
				</div>
			)}
		</div>
	);
};

export default TextField;
