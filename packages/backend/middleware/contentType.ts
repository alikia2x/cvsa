import { Context, Next } from "hono";

export const contentType = async (c: Context, next: Next) => {
	await next();
	c.header("Content-Type", "application/json; charset=utf-8");
};
