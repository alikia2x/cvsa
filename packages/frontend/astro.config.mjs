// @ts-check
import { defineConfig } from "astro/config";
import tailwind from "@astrojs/tailwind";

// https://astro.build/config
import tsconfigPaths from "vite-tsconfig-paths";

export default defineConfig({
	integrations: [tailwind()],
	vite: {
		server: {
			fs: {
				allow: [".", "../../"],
			},
		},
		plugins: [tsconfigPaths(),]
	},
});
