import { type Client, Pool } from "https://deno.land/x/postgres@v0.19.3/mod.ts";
import { postgresConfig, postgresConfigCred } from "@core/db/pgConfig.ts";
import { createMiddleware } from "hono/factory";

const pool = new Pool(postgresConfig, 4);
const poolCred = new Pool(postgresConfigCred, 2);

export const db = pool;
export const dbCred = poolCred;

export const dbMiddleware = createMiddleware(async (c, next) => {
    const connection = await pool.connect();
	c.set("db", connection);
	await next();
	connection.release();
});

export const dbCredMiddleware = createMiddleware(async (c, next) => {
    const connection = await poolCred.connect();
	c.set("dbCred", connection);
	await next();
	connection.release();
})

declare module "hono" {
	interface ContextVariableMap {
		db: Client;
		dbCred: Client;
	}
}
