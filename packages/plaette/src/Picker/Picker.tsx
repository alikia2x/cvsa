import { useState, useEffect } from "react";
import { oklch, formatHex, inGamut, parse, type Oklch, rgb } from "culori";
import { Slider } from "./Slider";
import { displaySupportsP3, roundOklch } from "./utils";

export type LchChannel = "l" | "c" | "h";

const toOklchString = (color: Oklch) => {
	return `oklch(${color.l} ${color.c} ${color.h})`;
};

export type i18nKeys = LchChannel | "fallback" | "unsupported";

export type I18nProvider = (key: i18nKeys) => string;

interface PickerProps extends React.HTMLAttributes<HTMLDivElement> {
	selectedColor: Oklch;
	onColorChange: (value: Oklch) => void;
	useP3: boolean;
	i18n: I18nProvider;
}

const Preview = ({ color, i18n }: { color: Oklch; i18n: I18nProvider }) => {
	const supportsP3 = displaySupportsP3();
	const outOfRgb = !inGamut("rgb")(color);
	const outOfP3 = !inGamut("p3")(color);
	if (outOfP3 || (outOfRgb && !supportsP3)) {
		const rgbColor = rgb(color);
		const hex = formatHex(rgbColor);
		const fallbackColor = supportsP3 ? hex : toOklchString(color);
		return (
			<div className="flex gap-1">
				<div
					className="w-full h-20 mb-5 rounded-lg flex items-center justify-center 
					border border-dashed border-gray-300 dark:border-gray-500"
				>
					<span className="text-gray-700 dark:text-zinc-300 text-sm text-center">{i18n("unsupported")}</span>
				</div>
				<div
					className="w-full h-20 mb-5 rounded-lg flex items-end justify-center"
					style={{ backgroundColor: fallbackColor }}
				>
					<span className="text-sm mb-1 bg-black/70 text-white px-2 rounded-xl">{i18n("fallback")}</span>
				</div>
			</div>
		);
	} else if (outOfRgb && supportsP3) {
		const rgbColor = rgb(color);
		const hex = formatHex(rgbColor);
		return (
			<div className="flex gap-1">
				<div
					className="w-full h-20 mb-5 rounded-lg flex items-end justify-center"
					style={{ backgroundColor: toOklchString(color) }}
				>
					<span className="text-sm mb-1 bg-black/70 text-white px-2 rounded-xl">P3</span>
				</div>
				<div
					className="w-full h-20 mb-5 rounded-lg flex items-end justify-center"
					style={{ backgroundColor: hex }}
				>
					<span className="text-sm mb-1 bg-black/70 text-white px-2 rounded-xl">{i18n("fallback")}</span>
				</div>
			</div>
		);
	}
	return (
		<div
			className="w-full h-20 mb-5 rounded-lg flex items-center justify-center"
			style={{ backgroundColor: toOklchString(color) }}
		></div>
	);
};

export const Picker = ({ useP3, i18n, selectedColor, onColorChange, ...rest }: PickerProps) => {
	const [displayColor, setDisplayColor] = useState<Oklch>(selectedColor);
	const [hexText, setHexText] = useState(formatHex(selectedColor));
	const [oklchText, setOklchText] = useState(toOklchString(selectedColor));
	const colorGamut = useP3 ? "p3" : "rgb";

	useEffect(() => {
		try {
			setHexText(formatHex(selectedColor));
			setOklchText(toOklchString(selectedColor));
		} catch (error) {
			console.warn("Invalid color combination");
		}
	}, [selectedColor]);

	const handleChange = (channel: LchChannel, value: number) => {
		setDisplayColor((prev) => ({ ...prev, [channel]: value }));
		onColorChange({ ...selectedColor, [channel]: value });
	};

	const generateChangeHandler = (channel: LchChannel) => {
		return (value: number) => {
			handleChange(channel, value);
		};
	};

	const handleHexChange = (e: React.ChangeEvent<HTMLInputElement>) => {
		const hex = e.target.value;
		const color = parse(hex);
		const oklchColor = oklch(rgb(color));
		if (oklchColor) {
			setDisplayColor(roundOklch(oklchColor));
		}
		setHexText(hex);
	};

	const handleHexInput = (e: React.ChangeEvent<HTMLInputElement>) => {
		const hex = e.target.value;
		const color = parse(hex);
		const oklchColor = oklch(rgb(color));
		console.log("hey");
		if (oklchColor) {
			onColorChange(roundOklch(oklchColor));
			setHexText(hex);
		} else {
			setHexText(formatHex(selectedColor));
		}
	};

	const handleOklchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
		const oklchString = e.target.value;
		const color = parse(oklchString);
		const oklchColor = oklch(color);
		if (oklchColor) {
			setDisplayColor(roundOklch(oklchColor));
		}
		setOklchText(oklchString);
	};

	const handleOklchInput = (e: React.ChangeEvent<HTMLInputElement>) => {
		const oklchString = e.target.value;
		const oklchColor = oklch(parse(oklchString));
		if (oklchColor) {
			onColorChange(roundOklch(oklchColor));
			setOklchText(oklchString);
		} else {
			setOklchText(toOklchString(selectedColor));
		}
	};

	return (
		<div {...rest}>
			<Preview color={displayColor} i18n={i18n} />

			<div className="flex flex-col">
				<input
					className="mb-5 font-mono bg-zinc-200 dark:bg-zinc-700 h-10 px-3 rounded-xl focus:outline-none 
					focus:ring focus:ring-2 focus:ring-black dark:focus:ring-white"
					type="text"
					value={hexText}
					onChange={handleHexChange}
					onBlur={handleHexInput}
					id="colorHex"
				/>
				<input
					className="mb-5 font-mono bg-zinc-200 dark:bg-zinc-700 h-10 px-3 rounded-xl focus:outline-none 
					focus:ring focus:ring-2 focus:ring-black dark:focus:ring-white"
					type="text"
					value={oklchText}
					onChange={handleOklchChange}
					onBlur={handleOklchInput}
					id="colorOklch"
				/>
			</div>

			<Slider
				channel="l"
				color={displayColor}
				onChange={generateChangeHandler("l")}
				i18nProvider={i18n}
				useP3={useP3}
			/>

			<Slider
				channel="c"
				color={displayColor}
				onChange={generateChangeHandler("c")}
				i18nProvider={i18n}
				useP3={useP3}
			/>

			<Slider
				channel="h"
				color={displayColor}
				onChange={generateChangeHandler("h")}
				i18nProvider={i18n}
				useP3={useP3}
			/>
		</div>
	);
};
