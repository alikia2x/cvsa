import "dotenv/config";
import { defineConfig } from "drizzle-kit";

export default defineConfig({
	out: "./drizzle/main",
	dialect: "postgresql",
	dbCredentials: {
		url: process.env.DATABASE_URL_MAIN!
	}
});
