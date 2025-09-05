"use server";

import { defineConfig } from "@solidjs/start/config";
import tsconfigPaths from "vite-tsconfig-paths";

export default defineConfig({
	vite: {
		plugins: [tsconfigPaths()],
		optimizeDeps: {
			include: ["@m3-components/solid"],
			esbuildOptions: {
				jsx: "automatic",
				jsxDev: true,
				jsxImportSource: "solid-js/h"
			}
		}
	}
});
