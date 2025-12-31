import { defineConfig } from "drizzle-kit";

export default defineConfig({
	dbCredentials: {
		url: process.env.DATABASE_URL_CRED!,
	},
	dialect: "postgresql",
	out: "./cred",
});
