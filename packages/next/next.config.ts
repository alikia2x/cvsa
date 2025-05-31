import type { NextConfig } from "next";
import createNextIntlPlugin from "next-intl/plugin";

const nextConfig: NextConfig = {
	devIndicators: false,
	experimental: {
		externalDir: true
	},
	transpilePackages: ["@cvsa/backend"]
};

const withNextIntl = createNextIntlPlugin();

export default withNextIntl(nextConfig);
