import type { NextConfig } from "next";
import createNextIntlPlugin from "next-intl/plugin";

const nextConfig: NextConfig = {
	devIndicators: false,
	experimental: {
		externalDir: true
	},
	turbopack: {
		rules: {
			"*.txt": {
				loaders: ["raw-loader"],
				as: "*.js"
			}
		}
	},
	webpack(config: import("webpack").Configuration) {
		config.module?.rules?.push({
			test: /\.txt/i,
			use: "raw-loader"
		});
		return config;
	},
	transpilePackages: ["next-mdx-remote"]
};

const withNextIntl = createNextIntlPlugin();

export default withNextIntl(nextConfig);
