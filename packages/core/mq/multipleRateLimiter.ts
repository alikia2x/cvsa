import { RateLimiter as Limiter } from "@koshnic/ratelimit";
import { redis } from "bun";
import Redis from "ioredis";

export interface RateLimiterConfig {
	duration: number;
	max: number;
}

export class RateLimiterError extends Error {
	public code: string;

	constructor(message: string) {
		super(message);
		this.name = "RateLimiterError";
		this.code = "RATE_LIMIT_EXCEEDED";
	}
}

export class MultipleRateLimiter {
	private readonly name: string;
	private readonly configs: RateLimiterConfig[] = [];
	private readonly limiter: Limiter;

	/*
	 * @param name The name of the rate limiter
	 * @param configs The configuration of the rate limiter, containing:
	 * - duration: The duration of window in seconds
	 * - max: The maximum number of tokens allowed in the window
	 */
	constructor(name: string, configs: RateLimiterConfig[]) {
		this.configs = configs;
		this.limiter = new Limiter(redis as unknown as Redis);
		this.name = name;
	}

	/*
	 * Trigger an event in the rate limiter
	 */
	async trigger(shouldThrow = true): Promise<void> {
		for (let i = 0; i < this.configs.length; i++) {
			const { duration, max } = this.configs[i];
			const { allowed } = await this.limiter.allow(`cvsa:${this.name}_${i}`, {
				burst: max,
				ratePerPeriod: max,
				period: duration,
				cost: 1
			});
			if (!allowed && shouldThrow) {
				throw new RateLimiterError("Rate limit exceeded");
			}
		}
	}
}
