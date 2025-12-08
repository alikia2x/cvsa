import { buildColorTokens } from "../colorTokens";
import { useTheme } from "../ThemeContext";
import { formatHex, Oklch } from "culori";

const SearchBar = ({ baseColor }: { baseColor: Oklch }) => {
	const { theme } = useTheme();
	const tokens = buildColorTokens(baseColor)[theme];

	return (
		<div
			className="w-full h-18 flex items-center justify-center rounded-md"
			style={{ backgroundColor: formatHex(tokens.background) }}
		>
			<input
				type="text"
				placeholder="搜索"
				className="w-2/3 h-10 rounded-lg focus:outline-none text-center"
				style={{
					backgroundColor: formatHex(tokens["bg-elevated-1"]),
					border: `2px solid ${formatHex(tokens["border-var-1"])}`,
					color: formatHex(tokens["body-text"])
				}}
			/>
		</div>
	);
};

const Paragraph = ({ baseColor }: { baseColor: Oklch }) => {
	const { theme } = useTheme();
	const tokens = buildColorTokens(baseColor)[theme];

	return (
		<div
			className="w-full h-18 flex items-center justify-center rounded-md px-4"
			style={{ backgroundColor: formatHex(tokens.background) }}
		>
			<p style={{ color: formatHex(tokens["body-text"]) }}>
				《尘海绘仙缘》是洛凛于 2024 年 12 月 15 日投稿至哔哩哔哩的 Synthesizer
				V 中文原创歌曲, 由赤羽演唱。
			</p>
		</div>
	);
};

const Buttons = ({ baseColor }: { baseColor: Oklch }) => {
	const { theme } = useTheme();
	const tokens = buildColorTokens(baseColor)[theme];

	return (
		<div
			className="w-full py-4 grid [grid-template-columns:repeat(auto-fit,minmax(120px,1fr))] place-items-center 
				items-center justify-between rounded-md gap-4 px-10"
			style={{ backgroundColor: formatHex(tokens.background) }}
		>
			<button
				className="w-24 cursor-pointer font-medium py-1.5 px-4 rounded-lg border-2"
				style={{
					borderColor: formatHex(tokens["border-var-3"]),
					color: formatHex(tokens["on-bg-var-2"])
				}}
			>
				Cancel
			</button>
			<button
				className="w-24 cursor-pointer font-medium py-2 px-4 rounded-lg"
				style={{
					backgroundColor: formatHex(tokens.primary),
					color: formatHex(tokens["on-primary"])
				}}
			>
				Confirm
			</button>
			<button
				className="w-24 cursor-pointer font-medium py-2 px-4 rounded-lg"
				style={{
					backgroundColor: formatHex(tokens.error),
					color: formatHex(tokens["on-error"])
				}}
			>
				Delete
			</button>
		</div>
	);
};

export { SearchBar, Paragraph, Buttons };
