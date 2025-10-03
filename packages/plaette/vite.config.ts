import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import UnoCSS from "unocss/vite";

// https://vite.dev/config/
export default defineConfig({
	plugins: [
		//@ts-ignore
		react({
			babel: {
				plugins: [["babel-plugin-react-compiler"]]
			}
		}),
		//@ts-ignore
		UnoCSS()
	]
});
