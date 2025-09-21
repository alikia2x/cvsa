import { defineConfig } from "drizzle-kit";

export default defineConfig({
	out: "./src/drizzle/cred",
	schema: "./src/db/schema.ts",
	dialect: "postgresql",
	dbCredentials: {
		url: process.env.DATABASE_URL_CRED!
	}
});
