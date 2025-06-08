import React, { useState, useRef, useCallback } from "react";
import { SearchIcon } from "@/components/icons/SearchIcon";
import { CloseIcon } from "@/components/icons/CloseIcon";

interface SearchBoxProps {
	close?: () => void;
}

export const SearchBox: React.FC<SearchBoxProps> = ({ close = () => {} }) => {
	const [inputValue, setInputValue] = useState("");
	const inputElement = useRef<HTMLInputElement>(null);

	const search = useCallback((query: string) => {
		if (query.trim()) {
			window.location.href = `/song/${query.trim()}/info`;
		}
	}, []);

	const handleInputChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
		setInputValue(event.target.value);
	}, []);

	const handleKeyDown = useCallback(
		(event: React.KeyboardEvent<HTMLInputElement>) => {
			if (event.key === "Enter") {
				event.preventDefault();
				search(inputValue);
			}
		},
		[inputValue, search]
	);

	const handleClear = useCallback(() => {
		setInputValue("");
		close();
	}, [close]);

	return (
		<div
			className="absolute md:relative left-0 h-full mr-0 inline-flex items-center w-full px-4
			    md:px-0 md:w-full xl:max-w-[50rem] md:mx-4"
		>
			<div
				className="w-full h-10 lg:h-12 px-4 rounded-full bg-surface-container-high
				    dark:bg-dark-surface-container-high backdrop-blur-lg flex justify-between md:px-5"
			>
				<button className="w-6" onClick={() => search(inputValue)}>
					<SearchIcon
						className="h-full inline-flex items-center text-[1.5rem]
						    text-on-surface-variant dark:text-dark-on-surface-variant"
					/>
				</button>
				<div className="md:hidden flex-grow px-4 top-0 h-full">
					<input
						ref={inputElement}
						value={inputValue}
						onChange={handleInputChange}
						type="search"
						placeholder="搜索"
						autoComplete="off"
						autoCapitalize="none"
						autoCorrect="off"
						className="bg-transparent h-full w-full focus:outline-none"
						onKeyDown={handleKeyDown}
						autoFocus={true}
					/>
				</div>
				<div className="hidden md:block flex-grow px-4 top-0 h-full">
					<input
						ref={inputElement}
						value={inputValue}
						onChange={handleInputChange}
						type="search"
						placeholder="搜索"
						autoComplete="off"
						autoCapitalize="none"
						autoCorrect="off"
						className="bg-transparent h-full w-full focus:outline-none"
						onKeyDown={handleKeyDown}
					/>
				</div>

				<button
					className={`w-6 duration-100 ${inputValue ? "md:opacity-100" : "md:opacity-0"}`}
					onClick={handleClear}
				>
					<CloseIcon className="h-full w-6 inline-flex items-center text-[1.5rem] text-on-surface-variant dark:text-dark-on-surface-variant" />
				</button>
			</div>
		</div>
	);
};
