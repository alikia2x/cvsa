import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";

const API_URL = "https://api.bilibili.com/x/web-interface/newlist?rid=30&ps=50&pn=";

const requiredEnvVars = ["DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD", "DB_PORT"];

const unsetVars = requiredEnvVars.filter((key) => !Deno.env.get(key));

if (unsetVars.length > 0) {
	throw new Error(`Missing required environment variables: ${unsetVars.join(", ")}`);
}

const databaseHost = Deno.env.get("DB_HOST")!;
const databaseName = Deno.env.get("DB_NAME")!;
const databaseUser = Deno.env.get("DB_USER")!;
const databasePassword = Deno.env.get("DB_PASSWORD")!;
const databasePort = Deno.env.get("DB_PORT")!;

const postgresConfig = {
	hostname: databaseHost,
	port: parseInt(databasePort),
	database: databaseName,
	user: databaseUser,
	password: databasePassword,
};

async function connectToPostgres() {
	const client = new Client(postgresConfig);
	await client.connect();
	return client;
}

export async function getLatestVideos() {
	const client = await connectToPostgres();
}
