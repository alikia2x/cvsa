import "dotenv/config";
import { defineConfig } from "drizzle-kit";

export default defineConfig({
	dbCredentials: {
		url: process.env.DATABASE_URL_MAIN!,
	},
	dialect: "postgresql",
	out: "./drizzle/main",
	schemaFilter: ["public", "credentials", "internal"],
});
