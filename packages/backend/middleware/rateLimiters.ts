import { rateLimiter, Store } from "hono-rate-limiter";
import type { BlankEnv } from "hono/types";
import { MINUTE } from "@core/const/time.ts";
import { getConnInfo } from "hono/bun";
import type { Context } from "hono";
import { redis } from "@core/db/redis.ts";
import { RedisStore } from "rate-limit-redis";

export const registerRateLimiter = rateLimiter<BlankEnv, "/user", {}>({
	windowMs: 60 * MINUTE,
	limit: 10,
	standardHeaders: "draft-6",
	keyGenerator: (c) => {
		const info = getConnInfo(c as unknown as Context<BlankEnv, "/user", {}>);
		if (!info.remote || !info.remote.address) {
			return crypto.randomUUID();
		}
		const addr = info.remote.address;
		const path = new URL(c.req.url).pathname;
		const method = c.req.method;
		return `${method}-${path}@${addr}`;
	},
	store: new RedisStore({
		// @ts-expect-error - Known issue: the `c`all` function is not present in @types/ioredis
		sendCommand: (...args: string[]) => redis.call(...args)
	}) as unknown as Store
});