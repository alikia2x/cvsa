import React, { useEffect, useRef, useState } from "react";
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
	const [value, setValue] = useState(color[channel]!.toFixed(precision[channel]));
	const containerRef = useRef<HTMLDivElement>(null);

	useEffect(() => {
		setValue(color[channel]!.toFixed(precision[channel]));
	}, [color.l, color.c, color.h]);

	const canvasRef = useRef<null | HTMLCanvasElement>(null);
	useOklchCanvas({ channel: channel, max: maxValue[channel], canvasRef: canvasRef, color, useP3 });

	const getSliderPosition = (value: number, max: number) => {
		return (value / max) * 100;
	};

	const getValueFromPosition = (clientX: number) => {
		if (!containerRef.current) return 0;

		const rect = containerRef.current.getBoundingClientRect();
		const x = clientX - rect.left;
		const percentage = Math.max(0, Math.min(1, x / rect.width));
		return round(percentage * maxValue[channel], precision[channel]);
	};

	const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
		const value = parseFloat(e.target.value);
		if (value > maxValue[channel]) onChange(maxValue[channel]);
		else if (value < 0 || isNaN(value)) onChange(0);
		else onChange(value);
		setValue(e.target.value);
	};

	const onBlur = (e: React.FocusEvent<HTMLInputElement>) => {
		const value = parseFloat(e.target.value);
		if (value > maxValue[channel]) {
			onChange(maxValue[channel]);
			setValue(maxValue[channel].toFixed(precision[channel]));
		} else if (value < 0 || isNaN(value)) {
			onChange(0);
			setValue("0");
		} else {
			onChange(value);
			setValue(value.toFixed(precision[channel]));
		}
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

	const handleTouchMove = (e: React.TouchEvent) => {
		e.preventDefault();
		const touch = e.touches[0];
		if (touch) {
			const newValue = getValueFromPosition(touch.clientX);
			handleOnChange(newValue);
		}
	};

	const handleTouchStart = (e: React.TouchEvent) => {
		const touch = e.touches[0];
		if (touch) {
			const newValue = getValueFromPosition(touch.clientX);
			handleOnChange(newValue);
		}
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
						value={value}
						onChange={onInputChange}
						onBlur={onBlur}
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

			<div
				ref={containerRef}
				className="relative h-10"
				onTouchMove={handleTouchMove}
				onTouchStart={handleTouchStart}
			>
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
					onTouchMove={handleTouchMove}
					onTouchStart={handleTouchStart}
				/>
			</div>
		</div>
	);
};
