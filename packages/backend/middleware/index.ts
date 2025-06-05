import { Context, Hono } from "hono";
import { Variables } from "hono/types";
import { bodyLimitForPing } from "./bodyLimits.ts";
import { pingHandler } from "routes/ping";
import { registerRateLimiter } from "./rateLimiters.ts";
import { preetifyResponse } from "./preetifyResponse.ts";
import { logger } from "./logger.ts";
import { timing } from "hono/timing";
import { contentType } from "./contentType.ts";
import { captchaMiddleware } from "./captcha.ts";
import { cors } from "hono/cors";

export function configureMiddleWares(app: Hono<{ Variables: Variables }>) {
	app.use("*", async (c, next) => {
		if (c.req.path === "/user") {
			const corsMiddlewareHandler = cors({
				origin: c.req.header("Origin"),
				credentials: true
			});
			return corsMiddlewareHandler(c, next);
		}
		const corsMiddlewareHandler = cors();
		return corsMiddlewareHandler(c, next);
	});

	app.use("*", contentType);
	app.use(timing());
	app.use("*", preetifyResponse);
	app.use("*", logger({}));

	app.post("/user", registerRateLimiter);
	app.post("/user", captchaMiddleware);
	app.all("/ping", bodyLimitForPing, ...pingHandler);
}
