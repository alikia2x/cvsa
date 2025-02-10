import { SlidingWindow } from "lib/mq/slidingWindow.ts";

export interface RateLimiterConfig {
	window: SlidingWindow;
	max: number;
}

export class RateLimiter {
	private readonly configs: RateLimiterConfig[];
    private readonly configEventNames: string[];

	/*
	 * @param name The name of the rate limiter
	 * @param configs The configuration of the rate limiter
	 */
	constructor(name: string, configs: RateLimiterConfig[]) {
		this.configs = configs;
        this.configEventNames = configs.map((_, index) => `${name}_config_${index}`);
	}

	/*
	 * Check if the event has reached the rate limit
	 */
	async getAvailability(): Promise<boolean> {
		for (let i = 0; i < this.configs.length; i++) {
			const config = this.configs[i];
			const eventName = this.configEventNames[i];
			const count = await config.window.count(eventName);
			if (count >= config.max) {
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
			const config = this.configs[i];
			const eventName = this.configEventNames[i];
			await config.window.event(eventName);
		}
	}
}