import { Hono } from "hono";
import { timing } from "hono/timing";
import { Variables } from "hono/types";
import { pingHandler } from "routes/ping";
import { logger } from "middleware/logger";
import { corsMiddleware } from "@/middleware/cors";
import { contentType } from "middleware/contentType";
import { captchaMiddleware } from "middleware/captcha";
import { bodyLimitForPing } from "middleware/bodyLimits";
import { registerRateLimiter } from "middleware/rateLimiters";
import { preetifyResponse } from "middleware/preetifyResponse";

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
