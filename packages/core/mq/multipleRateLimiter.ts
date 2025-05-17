import { RateLimiter as Limiter } from "@koshnic/ratelimit";
import { redis } from "@core/db/redis.ts";

export interface RateLimiterConfig {
    duration: number;
    max: number;
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
    constructor(
        name: string,
        configs: RateLimiterConfig[]
    ) {
        this.configs = configs;
        this.limiter = new Limiter(redis);
        this.name = name;
    }

    /*
     * Check if the event has reached the rate limit
     */
    async getAvailability(): Promise<boolean> {
        for (let i = 0; i < this.configs.length; i++) {
            const { duration, max } = this.configs[i];
            const { remaining } = await this.limiter.allow(`cvsa:${this.name}_${i}`, {
                burst: max,
                ratePerPeriod: max,
                period: duration,
                cost: 0
            });

            if (remaining < 1) {
                return false;
            }
        }
        return true;
    }

    /*
     * Trigger an event in the rate limiter
     */
    async trigger(): Promise<void> {
        for (let i = 0; i < this.configs.length; i++) {
            const { duration, max } = this.configs[i];
            await this.limiter.allow(`cvsa:${this.name}_${i}`, {
                burst: max,
                ratePerPeriod: max,
                period: duration,
                cost: 1
            });
        }
    }
}