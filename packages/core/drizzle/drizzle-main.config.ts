import "dotenv/config";
import { defineConfig } from "drizzle-kit";

if (!process.env.DATABASE_URL_MAIN) {
	throw new Error("DATABASE_URL_MAIN is not defined");
}

export default defineConfig({
	dbCredentials: {
		url: process.env.DATABASE_URL_MAIN,
	},
	dialect: "postgresql",
	out: "./drizzle/main",
});
