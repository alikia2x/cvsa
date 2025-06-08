import { cors } from "hono/cors";
import { Context, Next } from "hono";

export const corsMiddleware = async (c: Context, next: Next) => {
	if (c.req.path.startsWith("/user") || c.req.path.startsWith("/login")) {
		const corsMiddlewareHandler = cors({
			origin: c.req.header("Origin"),
			credentials: true
		});
		return corsMiddlewareHandler(c, next);
	}
	const corsMiddlewareHandler = cors();
	return corsMiddlewareHandler(c, next);
};
