import type { NextConfig } from "next";
import createNextIntlPlugin from "next-intl/plugin";
import { createMDX } from "fumadocs-mdx/next";

const nextConfig: NextConfig = {
	devIndicators: false,
	experimental: {
		externalDir: true
	},
	pageExtensions: ["js", "jsx", "md", "mdx", "ts", "tsx"]
};

const withNextIntl = createNextIntlPlugin();

const withMDX = createMDX();

export default withNextIntl(withMDX(nextConfig));
