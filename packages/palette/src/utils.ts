import type { Oklch } from "culori";
import type { i18nKeys } from "./components/Picker/Picker";

export const i18nProvider = (key: i18nKeys) => {
	switch (key) {
		case "l":
			return "Lightness";
		case "c":
			return "Chroma";
		case "h":
			return "Hue";
		case "fallback":
			return "Fallback";
		case "unsupported":
			return "Unavailable on this monitor";
	}
};

export const getAdjustedColor = (color: Oklch, l?: number, c?: number, h?: number) => {
	const newColor = { ...color };
	if (l) newColor.l = l;
	if (c) newColor.c = c;
	if (h) newColor.h = h;
	return newColor;
};
