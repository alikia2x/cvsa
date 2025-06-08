import type { NextConfig } from "next";
import createNextIntlPlugin from "next-intl/plugin";
import { createMDX } from "fumadocs-mdx/next";

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
	pageExtensions: ["js", "jsx", "md", "mdx", "ts", "tsx"]
};

const withNextIntl = createNextIntlPlugin();

const withMDX = createMDX();

export default withNextIntl(withMDX(nextConfig));
