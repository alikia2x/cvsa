import { rateLimiter, Store } from "hono-rate-limiter";
import type { BlankEnv } from "hono/types";
import { MINUTE } from "@core/const/time.ts";
import { getConnInfo } from "hono/bun";
import type { Context } from "hono";
import { redis } from "@core/db/redis.ts";
import { RedisStore } from "rate-limit-redis";
import { generateRandomId } from "@core/lib/randomID.ts";

export const getIdentifier = (c: Context, includeIP: boolean = true) => {
	let ipAddr = generateRandomId(6);
	const info = getConnInfo(c);
	if (info.remote && info.remote.address) {
		ipAddr = info.remote.address;
	}
	const forwardedFor = c.req.header("X-Forwarded-For");
	if (forwardedFor) {
		ipAddr = forwardedFor.split(",")[0];
	}
	const path = c.req.path;
	const method = c.req.method;
	const ipIdentifier = includeIP ? `@${ipAddr}` : "";
	return `${method}-${path}${ipIdentifier}`
}

export const registerRateLimiter = rateLimiter<BlankEnv, "/user", {}>({
	windowMs: 60 * MINUTE,
	limit: 10,
	standardHeaders: "draft-6",
	keyGenerator: getIdentifier,
	store: new RedisStore({
		// @ts-expect-error - Known issue: the `c`all` function is not present in @types/ioredis
		sendCommand: (...args: string[]) => redis.call(...args)
	}) as unknown as Store
});