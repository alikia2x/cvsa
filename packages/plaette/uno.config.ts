import presetWind4 from "@unocss/preset-wind4";
import { defineConfig } from "unocss";

export default defineConfig({
	presets: [presetWind4({ dark: "media"})],
	rules: [
		['font-mono', { 'font-family': '"Martian Mono", monospace' }],
	]
});
