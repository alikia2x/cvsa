// @ts-check
import { defineConfig } from "astro/config";

// https://astro.build/config
import tsconfigPaths from "vite-tsconfig-paths";
import node from "@astrojs/node";
import svelte from "@astrojs/svelte";

import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
	output: "server",
	adapter: node({
		mode: "standalone"
	}),
	integrations: [svelte()],
	vite: {
		server: {
			fs: {
				allow: [".", "../../"]
			}
		},
		plugins: [tsconfigPaths(), tailwindcss()]
	},
	markdown: {
		remarkRehype: { footnoteLabel: "脚注", footnoteBackLabel: "回到引用 1" }
	}
});
