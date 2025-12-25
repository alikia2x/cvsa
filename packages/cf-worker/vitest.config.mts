import { defineWorkersConfig } from "@cloudflare/vitest-pool-workers/config";

const SECOND = 1000;

export default defineWorkersConfig({
	test: {
		poolOptions: {
			workers: {
				wrangler: { configPath: "./wrangler.jsonc" },
			},
		},
		testTimeout: 30 * SECOND,
	},
});
