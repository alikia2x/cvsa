import { Redis } from "ioredis";

const host = process.env.REDIS_HOST || "localhost";
const port = parseInt(process.env.REDIS_PORT) || 6379;

export const redis = new Redis({
	port: port,
	host: host,
	maxRetriesPerRequest: null,
});
