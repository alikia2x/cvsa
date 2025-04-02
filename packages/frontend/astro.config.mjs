// @ts-check
import { defineConfig } from "astro/config";
import tailwind from "@astrojs/tailwind";

// https://astro.build/config
import tsconfigPaths from "vite-tsconfig-paths";
import node from "@astrojs/node";
import svelte from "@astrojs/svelte";

export default defineConfig({
	output: "server",
	adapter: node({
		mode: "standalone",
	}),
	integrations: [tailwind(), svelte()],
	vite: {
		server: {
			fs: {
				allow: [".", "../../"],
			},
		},
		plugins: [tsconfigPaths()]
	},
});
