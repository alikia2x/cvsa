import type { Oklch } from "culori";
import { getAdjustedColor } from "./utils";

export type ThemeMode = "light" | "dark";

export const buildColorTokens = (base: Oklch) => {
	return {
		dark: {
			background: getAdjustedColor(base, 0.15, 0.002),
			"bg-elevated-1": getAdjustedColor(base, 0.2, 0.004),
			"body-text": getAdjustedColor(base, 0.9, 0.01),
			"border-var-1": getAdjustedColor(base, 0.3, 0.004),
			"border-var-2": getAdjustedColor(base, 0.4, 0.007),
			"border-var-3": getAdjustedColor(base, 0.5, 0.01),
			error: { c: 0.223, h: 27.8, l: 0.65, mode: "oklch" } as Oklch,
			"on-bg-var-2": getAdjustedColor(base, 0.83, 0.028),
			"on-error": getAdjustedColor(base, 0.9, 0.01),
			"on-primary": getAdjustedColor(base, 0.3, 0.08),
			primary: getAdjustedColor(base, 0.84, 0.1),
		},
		light: {
			background: getAdjustedColor(base, 0.98, 0.01),
			"bg-elevated-1": getAdjustedColor(base, 1, 0.008),
			"body-text": getAdjustedColor(base, 0.1, 0.01),
			"border-var-1": getAdjustedColor(base, 0.845, 0.004),
			"border-var-2": getAdjustedColor(base, 0.8, 0.007),
			"border-var-3": getAdjustedColor(base, 0.755, 0.01),
			error: { c: 0.192, h: 27.7, l: 0.506, mode: "oklch" } as Oklch,
			"on-bg-var-2": getAdjustedColor(base, 0.398, 0.0234),
			"on-error": getAdjustedColor(base, 0.99, 0.01),
			"on-primary": getAdjustedColor(base, 0.999, 0.001),
			primary: getAdjustedColor(base, 0.48, 0.08),
		},
	};
};
