import type { BlankEnv } from "hono/types";
import { getConnInfo } from "hono/bun";
import { Context, Next } from "hono";
import { generateRandomId } from "@core/lib/randomID.ts";
import { RateLimiter } from "@koshnic/ratelimit";
import { ErrorResponse } from "@/src/schema";
import { redis } from "@core/db/redis.ts";

export const getUserIP = (c: Context) => {
	let ipAddr = null;
	const info = getConnInfo(c);
	if (info.remote && info.remote.address) {
		ipAddr = info.remote.address;
	}
	const forwardedFor = c.req.header("X-Forwarded-For");
	if (forwardedFor) {
		ipAddr = forwardedFor.split(",")[0];
	}
	return ipAddr;
};

export const getIdentifier = (c: Context, includeIP: boolean = true) => {
	let ipAddr = generateRandomId(6);
	if (getUserIP(c)) {
		ipAddr = getUserIP(c);
	}
	const path = c.req.path;
	const method = c.req.method;
	const ipIdentifier = includeIP ? `@${ipAddr}` : "";
	return `${method}-${path}${ipIdentifier}`;
};

export const registerRateLimiter = async (c: Context<BlankEnv, "/user", {}>, next: Next) => {
	const limiter = new RateLimiter(redis);
	const identifier = getIdentifier(c, true);
	const { allowed, retryAfter } = await limiter.allow(identifier, {
		burst: 5,
		ratePerPeriod: 5,
		period: 120,
		cost: 1
	});

	if (!allowed) {
		const response: ErrorResponse = {
			message: `Too many requests, please retry after ${Math.round(retryAfter)} seconds.`,
			code: "RATE_LIMIT_EXCEEDED"
		};
		return c.json<ErrorResponse>(response, 429);
	}

	await next();
};
