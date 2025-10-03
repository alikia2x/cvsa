import React, { useRef } from "react";
import { useOklchCanvas } from "./useOklchCanvas";
import { Handle } from "./Handle";
import type { I18nProvider, LchChannel } from "./Picker";
import { type Oklch } from "culori";
import { round, precision, maxValue } from "./utils";

interface SliderProps {
	useP3: boolean;
	channel: LchChannel;
	color: Oklch;
	onChange: (value: number) => void;
	i18nProvider: I18nProvider;
}

export const Slider = ({ useP3, channel, color, onChange, i18nProvider }: SliderProps) => {
	const canvasRef = useRef<null | HTMLCanvasElement>(null);
	useOklchCanvas({ channel: channel, max: maxValue[channel], canvasRef: canvasRef, color, useP3 });
	const getSliderPosition = (value: number, max: number) => {
		return (value / max) * 100;
	};

	const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
		const value = parseFloat(e.target.value);
		onChange(value);
	};

	const buttonHanlder = (type: "increase" | "decrease") => {
		const factor = type === "increase" ? 1 : -1;
		const step = Math.pow(10, -precision[channel] + 2);
		const delta = factor * step;
		onChange(round(color[channel]! + delta, precision[channel]));
	};

	const handleOnChange = (value: number) => {
		onChange(round(value, precision[channel]));
	};


	return (
		<div className="mb-6">
			<div className="flex justify-between items-center mb-2">
				<label className="block font-bold text-lg">{i18nProvider(channel)}</label>
				<div className="relative">
					<input
						type="text"
						className="w-28 h-10 text-right text-[15px] font-mono bg-zinc-200 dark:bg-zinc-700 rounded-lg pl-3 pr-6 focus:outline-none 
                        focus:ring-2 focus:ring-black dark:focus:ring-white font-stretch-semi-expanded"
						value={color[channel]}
						onChange={onInputChange}
						step={Math.pow(10, -precision[channel])}
						aria-label={i18nProvider(channel)}
						aria-keyshortcuts={channel}
						role="spinbutton"
						aria-valuemin={0}
						aria-valuemax={maxValue[channel]}
						pattern="^[0-9+\/*.\-]+$"
						inputMode="decimal"
					/>
					<button
						className="field_control is-increase"
						tabIndex={-1}
						aria-hidden="true"
						onClick={() => buttonHanlder("increase")}
					/>
					<button
						className="field_control is-decrease"
						tabIndex={-1}
						aria-hidden="true"
						onClick={() => buttonHanlder("decrease")}
					/>
				</div>
			</div>

			<div className="relative h-10">
				<div className="absolute z-3 inset-0 rounded-xl overflow-hidden">
					<canvas ref={canvasRef} width={400} height={40} className="w-full h-full" />
				</div>
				<div className="absolute z-0 inset-0 rounded-xl border border-dashed border-gray-300 dark:border-gray-500 pointer-events-none" />
				<input
					type="range"
					min="0"
					max={maxValue[channel]}
					step={Math.pow(10, -precision[channel])}
					value={color[channel]}
					onChange={onInputChange}
					className="absolute z-5 inset-0 w-full h-full opacity-0 cursor-pointer"
				/>
				<Handle
					pos={getSliderPosition(color[channel]!, maxValue[channel])}
					color={color}
					onChange={handleOnChange}
					maxValue={maxValue[channel]}
				/>
			</div>
		</div>
	);
};
