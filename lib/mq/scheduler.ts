import logger from "lib/log/logger.ts";
import { RateLimiter, RateLimiterConfig } from "lib/mq/rateLimiter.ts";
import { SlidingWindow } from "lib/mq/slidingWindow.ts";
import { redis } from "lib/db/redis.ts";
import Redis from "ioredis";
import { SECOND } from "$std/datetime/constants.ts";

interface Proxy {
	type: string;
	data: string;
}

interface Task {
	provider: string;
	proxies: string[] | "all";
}

interface ProxiesMap {
	[name: string]: Proxy;
}

type NetSchedulerErrorCode =
	| "NO_AVAILABLE_PROXY"
	| "PROXY_RATE_LIMITED"
	| "PROXY_NOT_FOUND"
	| "FETCH_ERROR"
	| "NOT_IMPLEMENTED";

export class NetSchedulerError extends Error {
	public code: NetSchedulerErrorCode;
	public rawError: unknown | undefined;
	constructor(message: string, errorCode: NetSchedulerErrorCode, rawError?: unknown) {
		super(message);
		this.name = "NetSchedulerError";
		this.code = errorCode;
		this.rawError = rawError;
	}
}

type LimiterMap = {
	[name: string]: RateLimiter;
};

type OptionalLimiterMap = {
	[name: string]: RateLimiter | null;
};

type TaskMap = {
	[name: string]: Task;
};

function shuffleArray<T>(array: T[]): T[] {
	const newArray = [...array]; // Create a shallow copy to avoid in-place modification
	for (let i = newArray.length - 1; i > 0; i--) {
		const j = Math.floor(Math.random() * (i + 1));
		[newArray[i], newArray[j]] = [newArray[j], newArray[i]]; // Swap elements
	}
	return newArray;
}

class NetScheduler {
	private proxies: ProxiesMap = {};
	private providerLimiters: LimiterMap = {};
	private proxyLimiters: OptionalLimiterMap = {};
	private tasks: TaskMap = {};

	addProxy(proxyName: string, type: string, data: string): void {
		this.proxies[proxyName] = { type, data };
	}

	removeProxy(proxyName: string): void {
		if (!this.proxies[proxyName]) {
			throw new Error(`Proxy ${proxyName} not found`);
		}
		delete this.proxies[proxyName];
		// Clean up associated limiters
		this.cleanupProxyLimiters(proxyName);
	}

	private cleanupProxyLimiters(proxyName: string): void {
		for (const limiterId in this.proxyLimiters) {
			if (limiterId.startsWith(`proxy-${proxyName}`)) {
				delete this.proxyLimiters[limiterId];
			}
		}
	}

	addTask(taskName: string, provider: string, proxies: string[] | "all"): void {
		this.tasks[taskName] = { provider, proxies };
	}

	getTaskProxies(taskName: string): string[] {
		if (!this.tasks[taskName]) {
			return [];
		}
		if (this.tasks[taskName].proxies === "all") {
			return Object.keys(this.proxies);
		}
		return this.tasks[taskName].proxies;
	}

	setTaskLimiter(taskName: string, config: RateLimiterConfig[] | null): void {
		const proxies = this.getTaskProxies(taskName);
		for (const proxyName of proxies) {
			const limiterId = "proxy-" + proxyName + "-" + taskName;
			this.proxyLimiters[limiterId] = config ? new RateLimiter(limiterId, config) : null;
		}
	}

	async triggerLimiter(task: string, proxy: string): Promise<void> {
		const limiterId = "proxy-" + proxy + "-" + task;
		if (!this.proxyLimiters[limiterId]) {
			return;
		}
		try {
			await this.proxyLimiters[limiterId].trigger();
		} catch (e) {
			const error = e as Error;
			if (e instanceof Redis.ReplyError) {
				logger.error(error, "redis");
			}
			logger.warn(`Unhandled error: ${error.message}`, "mq", "proxyRequest");
		}
	}

	setProviderLimiter(providerName: string, config: RateLimiterConfig[]): void {
		let bindProxies: string[] = [];
		for (const taskName in this.tasks) {
			if (this.tasks[taskName].provider !== providerName) continue;
			const proxies = this.getTaskProxies(taskName);
			bindProxies = bindProxies.concat(proxies);
		}
		for (const proxyName of bindProxies) {
			const limiterId = "provider-" + proxyName + "-" + providerName;
			this.providerLimiters[limiterId] = new RateLimiter(limiterId, config);
		}
	}

	/*
	 * Make a request to the specified URL with any available proxy
	 * @param {string} url - The URL to request.
	 * @param {string} method - The HTTP method to use for the request. Default is "GET".
	 * @returns {Promise<any>} - A promise that resolves to the response body.
	 * @throws {NetSchedulerError} - The error will be thrown in following cases:
	 * - No available proxy currently: with error code NO_AVAILABLE_PROXY
	 * - Proxy is under rate limit: with error code PROXY_RATE_LIMITED
	 * - The native `fetch` function threw an error: with error code FETCH_ERROR
	 * - The proxy type is not supported: with error code NOT_IMPLEMENTED
	 */
	async request<R>(url: string, task: string, method: string = "GET"): Promise<R> {
		// find a available proxy
		const proxiesNames = this.getTaskProxies(task);
		for (const proxyName of shuffleArray(proxiesNames)) {
			if (await this.getProxyAvailability(proxyName, task)) {
				return await this.proxyRequest<R>(url, proxyName, task, method);
			}
		}
		throw new NetSchedulerError("No available proxy currently.", "NO_AVAILABLE_PROXY");
	}

	/*
	 * Make a request to the specified URL with the specified proxy
	 * @param {string} url - The URL to request.
	 * @param {string} proxyName - The name of the proxy to use.
	 * @param {string} task - The name of the task to use.
	 * @param {string} method - The HTTP method to use for the request. Default is "GET".
	 * @param {boolean} force - If true, the request will be made even if the proxy is rate limited. Default is false.
	 * @returns {Promise<any>} - A promise that resolves to the response body.
	 * @throws {NetSchedulerError} - The error will be thrown in following cases:
	 * - Proxy not found: with error code PROXY_NOT_FOUND
	 * - Proxy is under rate limit: with error code PROXY_RATE_LIMITED
	 * - The native `fetch` function threw an error: with error code FETCH_ERROR
	 * - The proxy type is not supported: with error code NOT_IMPLEMENTED
	 */
	async proxyRequest<R>(
		url: string,
		proxyName: string,
		task: string,
		method: string = "GET",
		force: boolean = false,
	): Promise<R> {
		const proxy = this.proxies[proxyName];
		if (!proxy) {
			throw new NetSchedulerError(`Proxy "${proxyName}" not found`, "PROXY_NOT_FOUND");
		}

		if (!force) {
			const isAvailable = await this.getProxyAvailability(proxyName, task);
			const limiter = "proxy-" + proxyName + "-" + task
			if (!isAvailable) {
				throw new NetSchedulerError(`Proxy "${limiter}" is rate limited`, "PROXY_RATE_LIMITED");
			}
		}

		const result = await this.makeRequest<R>(url, proxy, method);
		await this.triggerLimiter(task, proxyName);
		return result;
	}

	private async makeRequest<R>(url: string, proxy: Proxy, method: string): Promise<R> {
		switch (proxy.type) {
			case "native":
				return await this.nativeRequest<R>(url, method);
			default:
				throw new NetSchedulerError(`Proxy type ${proxy.type} not supported`, "NOT_IMPLEMENTED");
		}
	}

	private async getProxyAvailability(proxyName: string, taskName: string): Promise<boolean> {
		try {
			const task = this.tasks[taskName];
			const provider = task.provider;
			const proxyLimiterId = "proxy-" + proxyName + "-" + task;
			const providerLimiterId = "provider-" + proxyName + "-" + provider;
			if (!this.proxyLimiters[proxyLimiterId] || !this.providerLimiters[providerLimiterId]) {
				return true;
			}
			const proxyLimiter = this.proxyLimiters[proxyLimiterId];
			const providerLimiter = this.providerLimiters[providerLimiterId];
			const providerAvailable = await providerLimiter.getAvailability();
			const proxyAvailable = await proxyLimiter.getAvailability();
			return providerAvailable && proxyAvailable;
		} catch (e) {
			const error = e as Error;
			if (e instanceof Redis.ReplyError) {
				logger.error(error, "redis");
				return false;
			}
			logger.error(error, "mq", "getProxyAvailability");
			return false;
		}
	}

	private async nativeRequest<R>(url: string, method: string): Promise<R> {
		try {
			const controller = new AbortController();
			const timeout = setTimeout(() => controller.abort(), 10 * SECOND);

			const response = await fetch(url, {
				method,
				signal: controller.signal,
			});

			clearTimeout(timeout);

			return await response.json() as R;
		} catch (e) {
			throw new NetSchedulerError("Fetch error", "FETCH_ERROR", e);
		}
	}
}

const netScheduler = new NetScheduler();
const videoInfoRateLimiterConfig: RateLimiterConfig[] = [
	{
		window: new SlidingWindow(redis, 0.3),
		max: 1,
	},
	{
		window: new SlidingWindow(redis, 3),
		max: 5,
	},
	{
		window: new SlidingWindow(redis, 30),
		max: 30,
	},
	{
		window: new SlidingWindow(redis, 2 * 60),
		max: 50,
	},
];
const biliLimiterConfig: RateLimiterConfig[] = [
	{
		window: new SlidingWindow(redis, 1),
		max: 5,
	},
	{
		window: new SlidingWindow(redis, 30),
		max: 100,
	},
	{
		window: new SlidingWindow(redis, 5 * 60),
		max: 180,
	},
];
netScheduler.addProxy("native", "native", "");
netScheduler.addTask("getVideoInfo", "bilibili", "all");
netScheduler.addTask("getLatestVideos", "bilibili", "all");
netScheduler.setTaskLimiter("getVideoInfo", videoInfoRateLimiterConfig);
netScheduler.setTaskLimiter("getLatestVideos", null);
netScheduler.setProviderLimiter("bilibili", biliLimiterConfig);

export default netScheduler;
