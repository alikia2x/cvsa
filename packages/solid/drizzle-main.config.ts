import "dotenv/config";
import { defineConfig } from "drizzle-kit";

export default defineConfig({
	out: "./src/drizzle/main",
	dialect: "postgresql",
	dbCredentials: {
		url: process.env.DATABASE_URL_MAIN!
	},
	tablesFilter: ["*"],
});
