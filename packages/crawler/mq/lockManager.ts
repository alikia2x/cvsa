import { Redis } from "ioredis";
import { redis } from "db/redis.ts";

class LockManager {
	private redis: Redis;

	/*
	 * Create a new LockManager
	 * @param redisClient The Redis client used to store the lock data
	 */
	constructor(redisClient: Redis) {
		this.redis = redisClient;
	}

	/*
	 * Acquire a lock for a given ID
	 * @param id The unique identifier for the lock
	 * @param timeout Optional timeout in seconds after which the lock will automatically be released
	 * @returns true if the lock was successfully acquired, false otherwise
	 */
	async acquireLock(id: string, timeout?: number): Promise<boolean> {
		const key = `cvsa:lock:${id}`;
		const result = await this.redis.set(key, "locked", "NX");

		if (result !== "OK") {
			return false;
		}
		if (timeout) {
			await this.redis.expire(key, timeout);
		}
		return true;
	}

	/*
	 * Release a lock for a given ID
	 * @param id The unique identifier for the lock
	 * @returns true if the lock was successfully released, false otherwise
	 */
	async releaseLock(id: string): Promise<boolean> {
		const key = `cvsa:lock:${id}`;
		const result = await this.redis.del(key);
		return result === 1;
	}

	/*
	 * Check if a lock is currently held for a given ID
	 * @param id The unique identifier for the lock
	 * @returns true if the lock is currently held, false otherwise
	 */
	async isLocked(id: string): Promise<boolean> {
		const key = `cvsa:lock:${id}`;
		const result = await this.redis.exists(key);
		return result === 1;
	}
}

export const lockManager = new LockManager(redis);
