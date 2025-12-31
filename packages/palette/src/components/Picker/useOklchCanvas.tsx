import { formatHex, inGamut, type Oklch, oklch } from "culori";
import { type RefObject, useEffect } from "react";

interface UseOklchCanvasOptions {
	useP3: boolean;
	channel: "l" | "c" | "h";
	max: number;
	canvasRef: RefObject<HTMLCanvasElement | null>;
	color: Oklch;
}

export function useOklchCanvas({ useP3, channel, max, canvasRef, color }: UseOklchCanvasOptions) {
	useEffect(() => {
		const colorGamut = useP3 ? "p3" : "rgb";
		const canvas = canvasRef.current;
		if (!canvas) return;

		const ctx = canvas.getContext("2d")!;
		const width = canvas.width;
		const height = canvas.height;

		ctx.clearRect(0, 0, width, height);

		const imageData = ctx.createImageData(width, height);
		const data = imageData.data;

		for (let x = 0; x < width; x++) {
			let value: number;
			const ratio = x / width;

			switch (channel) {
				case "l":
					value = ratio;
					break;
				case "c":
					value = ratio * max;
					break;
				case "h":
					value = ratio * max;
					break;
				default:
					value = ratio;
			}

			try {
				const testColor = oklch({
					c: channel === "c" ? value : color.c,
					h: channel === "h" ? value : color.h,
					l: channel === "l" ? value : color.l,
					mode: "oklch",
				});

				if (testColor && inGamut(colorGamut)(testColor)) {
					const hex = formatHex(testColor);
					const r = parseInt(hex.slice(1, 3), 16);
					const g = parseInt(hex.slice(3, 5), 16);
					const b = parseInt(hex.slice(5, 7), 16);

					for (let y = 0; y < height; y++) {
						const index = (y * width + x) * 4;
						data[index] = r;
						data[index + 1] = g;
						data[index + 2] = b;
						data[index + 3] = 255;
					}
				} else {
					for (let y = 0; y < height; y++) {
						const index = (y * width + x) * 4;
						data[index] = 0;
						data[index + 1] = 0;
						data[index + 2] = 0;
						data[index + 3] = 0;
					}
				}
			} catch {
				for (let y = 0; y < height; y++) {
					const index = (y * width + x) * 4;
					data[index] = 0;
					data[index + 1] = 0;
					data[index + 2] = 0;
					data[index + 3] = 0;
				}
			}
		}

		ctx.putImageData(imageData, 0, 0);
	}, [channel, color.l, color.c, color.h, canvasRef, useP3]);
}
