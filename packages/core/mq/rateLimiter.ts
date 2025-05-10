import { TokenBucket } from "./tokenBucket.ts";

export interface RateLimiterConfig {
    duration: number;
    max: number;
}

export class RateLimiter {
    private configs: RateLimiterConfig[] = [];
    private buckets: TokenBucket[] = [];
    private identifierFn: (configIndex: number) => string;

    /*
     * @param name The name of the rate limiter
     * @param configs The configuration of the rate limiter, containing:
     * - tokenBucket: The token bucket instance
     * - max: The maximum number of tokens allowed per operation
     */
    constructor(
        name: string,
        configs: RateLimiterConfig[],
        identifierFn?: (configIndex: number) => string
    ) {
        this.configs = configs;
        this.identifierFn = identifierFn || ((index) => `${name}_config_${index}`);
        for (let i = 0; i < configs.length; i++) {
            const config = configs[i];
            const bucket = new TokenBucket({
                capacity: config.max,
                rate: config.max / config.duration,
                identifier: this.identifierFn(i),
            })
            this.buckets.push(bucket);
        }
    }

    /*
     * Check if the event has reached the rate limit
     */
    async getAvailability(): Promise<boolean> {
        for (let i = 0; i < this.configs.length; i++) {
            const remaining = await this.buckets[i].getRemainingTokens();

            if (remaining === null) {
                return false; // Rate limit exceeded
            }
        }
        return true;
    }

    /*
     * Trigger an event in the rate limiter
     */
    async trigger(): Promise<void> {
        for (let i = 0; i < this.configs.length; i++) {
            await this.buckets[i].consume(1);
        }
    }

    /*
     * Clear all buckets for all configurations
     */
    async clear(): Promise<void> {
        for (let i = 0; i < this.configs.length; i++) {
            await this.buckets[i].reset();
        }
    }
}