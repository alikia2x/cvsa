import react from "@vitejs/plugin-react";
import UnoCSS from "unocss/vite";
import { defineConfig } from "vite";

// https://vite.dev/config/
export default defineConfig({
	plugins: [
		//@ts-expect-error
		react({
			babel: {
				plugins: [["babel-plugin-react-compiler"]],
			},
		}),
		//@ts-expect-error
		UnoCSS(),
	],
});
