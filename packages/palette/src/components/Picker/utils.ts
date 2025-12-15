import type { Oklch } from "culori";

export const round = (value: number, precision: number) => {
	return Math.round(value * 10 ** precision) / 10 ** precision;
};

export const roundOklch = (oklch: Oklch) => {
	return {
		...oklch,
		l: round(oklch.l, precision.l),
		c: round(oklch.c, precision.c),
		h: round(oklch.h!, precision.h),
	};
};

export const precision = {
	l: 4,
	c: 4,
	h: 2,
};

export const maxValue = {
	l: 1,
	c: 0.37,
	h: 360,
};

/**
 * Detects if the user's display supports the Display P3 color gamut.
 * This is based on the CSS color-gamut media feature.
 * * @returns {boolean} True if the display supports P3 or a wider gamut, false otherwise.
 */
export function displaySupportsP3(): boolean {
	if (typeof window === "undefined" || !window.matchMedia) {
		return false;
	}

	try {
		return window.matchMedia("(color-gamut: p3)").matches;
	} catch {
		return false;
	}
}
