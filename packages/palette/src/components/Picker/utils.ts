import type { Oklch } from "culori";

export const round = (value: number, precision: number) => {
	return Math.round(value * 10 ** precision) / 10 ** precision;
};

export const roundOklch = (oklch: Oklch) => {
	return {
		...oklch,
		c: round(oklch.c, precision.c),
		h: round(oklch.h!, precision.h),
		l: round(oklch.l, precision.l),
	};
};

export const precision = {
	c: 4,
	h: 2,
	l: 4,
};

export const maxValue = {
	c: 0.37,
	h: 360,
	l: 1,
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
