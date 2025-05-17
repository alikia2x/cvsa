import { redis } from "@core/db/redis";
import { SECOND } from "@core/const/time";

export interface TokenBucketRateOptions {
	capacity: number;
	rate: number;
	identifier: string;
	keyPrefix?: string;
}

export interface TokenBucketDurationOptions {
	duration: number;
	max: number;
	identifier: string;
	keyPrefix?: string;
}

export type TokenBucketConstructorOptions = TokenBucketRateOptions | TokenBucketDurationOptions;

export class TokenBucket {
	private readonly capacity: number;
	private readonly rate: number;
	private readonly keyPrefix: string;
	private readonly identifier: string;

	constructor(options: TokenBucketConstructorOptions) {
		if (!options.identifier) {
			throw new Error("Identifier is required.");
		}
		this.identifier = options.identifier;
		this.keyPrefix = options.keyPrefix || "cvsa:token_bucket:";

		const isRateOptions = 'capacity' in options && 'rate' in options;
		const isDurationOptions = 'duration' in options && 'max' in options;

		if (isRateOptions && isDurationOptions) {
			throw new Error("Provide either 'capacity'/'rate' or 'duration'/'max', not both.");
		} else if (isRateOptions) {
			if (options.capacity <= 0 || options.rate <= 0) {
				throw new Error("'capacity' and 'rate' must be greater than zero.");
			}
			this.capacity = options.capacity;
			this.rate = options.rate;
		} else if (isDurationOptions) {
			if (options.duration <= 0 || options.max <= 0) {
				throw new Error("'duration' and 'max' must be greater than zero.");
			}
			this.capacity = options.max;
			this.rate = options.max / options.duration;

		} else {
			throw new Error("Provide either 'capacity'/'rate' or 'duration'/'max'.");
		}
	}

	getKey(): string {
		return `${this.keyPrefix}${this.identifier}`;
	}

	/**
	 * Try to consume a specified number of tokens
	 * @param count The number of tokens to be consumed
	 * @returns If consumption is successful, returns the number of remaining tokens; otherwise returns null
	 */
	public async consume(count: number): Promise<number | null> {
		const key = this.getKey();
		const now = Math.floor(Date.now() / SECOND);

		const script = `
            local tokens_key = KEYS[1]
            local last_refilled_key = KEYS[2]
            local now = tonumber(ARGV[1])
            local count = tonumber(ARGV[2])
            local capacity = tonumber(ARGV[3])
            local rate = tonumber(ARGV[4])

            local last_refilled = tonumber(redis.call('GET', last_refilled_key)) or now
            local current_tokens = tonumber(redis.call('GET', tokens_key)) or capacity

            local elapsed = now - last_refilled
            local new_tokens = elapsed * rate
            current_tokens = math.min(capacity, current_tokens + new_tokens)

            if current_tokens >= count then
                current_tokens = current_tokens - count
                redis.call('SET', tokens_key, current_tokens)
                redis.call('SET', last_refilled_key, now)
                return current_tokens
            else
                return nil
            end
        `;

		const keys = [`${key}:tokens`, `${key}:last_refilled`];
		const args = [now, count, this.capacity, this.rate];

		const result = await redis.eval(script, keys.length, ...keys, ...args);

		return result as number | null;
	}

	public async getRemainingTokens(): Promise<number> {
		const key = this.getKey();
		const tokens = await redis.get(`${key}:tokens`);
		return Number(tokens) || this.capacity;
	}

	public async reset(): Promise<void> {
		const key = this.getKey();
		await redis.del(`${key}:tokens`, `${key}:last_refilled`);
	}
}
