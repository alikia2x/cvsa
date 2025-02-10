import { Redis } from "ioredis";

export class SlidingWindow {
	private redis: Redis;
	private readonly windowSize: number;

	/*
	 * Create a new sliding window
	 * @param redisClient The Redis client used to store the data
	 * @param windowSize The size of the window in seconds
	 */
	constructor(redisClient: Redis, windowSize: number) {
		this.redis = redisClient;
		this.windowSize = windowSize * 1000;
	}

	/*
	 * Trigger an event in the sliding window
	 * @param eventName The name of the event
	 */
	async event(eventName: string): Promise<void> {
		const now = Date.now();
		const key = `cvsa:sliding_window:${eventName}`;
		
		const uniqueMember = `${now}-${Math.random()}`;
		// Add current timestamp to an ordered set
		await this.redis.zadd(key, now, uniqueMember);

		// Remove timestamps outside the window
		await this.redis.zremrangebyscore(key, 0, now - this.windowSize);
	}

	/*
	 * Count the number of events in the sliding window
	 * @param eventName The name of the event
	 */
	async count(eventName: string): Promise<number> {
		const key = `cvsa:sliding_window:${eventName}`;
		const now = Date.now();

		// Remove timestamps outside the window
		await this.redis.zremrangebyscore(key, 0, now - this.windowSize);
		// Get the number of timestamps in the window
		return this.redis.zcard(key);
	}

	clear(eventName: string): Promise<number> {
		const key = `cvsa:sliding_window:${eventName}`;
		return this.redis.del(key);
	}
}
