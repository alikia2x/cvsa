import { Hono } from "hono";
import { timing } from "hono/timing";
import { Variables } from "hono/types";
import { pingHandler } from "routes/ping";
import { logger } from "middleware/logger.ts";
import { corsMiddleware } from "@/middleware/cors";
import { contentType } from "middleware/contentType.ts";
import { captchaMiddleware } from "middleware/captcha.ts";
import { bodyLimitForPing } from "middleware/bodyLimits.ts";
import { registerRateLimiter } from "middleware/rateLimiters.ts";
import { preetifyResponse } from "middleware/preetifyResponse.ts";

export function configureMiddleWares(app: Hono<{ Variables: Variables }>) {
	app.use("*", corsMiddleware);

	app.use("*", contentType);
	app.use(timing());
	app.use("*", preetifyResponse);
	app.use("*", logger({}));

	app.post("/user", registerRateLimiter);
	app.post("/user", captchaMiddleware);
	app.all("/ping", bodyLimitForPing, ...pingHandler);
}
