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
	| "NO_PROXY_AVAILABLE"
	| "PROXY_RATE_LIMITED"
	| "PROXY_NOT_FOUND"
	| "FETCH_ERROR"
	| "NOT_IMPLEMENTED"
	| "ALICLOUD_PROXY_ERR";

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
		const providerLimiterId = "provider-" + proxy + "-" + this.tasks[task].provider;
		try {
			await this.proxyLimiters[limiterId]?.trigger();
			await this.providerLimiters[providerLimiterId]?.trigger();
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
	 * - No proxy is available currently: with error code `NO_PROXY_AVAILABLE`
	 * - The native `fetch` function threw an error: with error code `FETCH_ERROR`
	 * - The alicloud-fc threw an error: with error code `ALICLOUD_FC_ERROR`
	 * - The proxy type is not supported: with error code `NOT_IMPLEMENTED`
	 */
	async request<R>(url: string, task: string, method: string = "GET"): Promise<R> {
		// find a available proxy
		const proxiesNames = this.getTaskProxies(task);
		for (const proxyName of shuffleArray(proxiesNames)) {
			if (await this.getProxyAvailability(proxyName, task)) {
				return await this.proxyRequest<R>(url, proxyName, task, method);
			}
		}
		throw new NetSchedulerError("No proxy is available currently.", "NO_PROXY_AVAILABLE");
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
	 * - Proxy not found: with error code `PROXY_NOT_FOUND`
	 * - Proxy is under rate limit: with error code `PROXY_RATE_LIMITED`
	 * - The native `fetch` function threw an error: with error code `FETCH_ERROR`
	 * - The alicloud-fc threw an error: with error code `ALICLOUD_FC_ERROR`
	 * - The proxy type is not supported: with error code `NOT_IMPLEMENTED`
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
			const limiter = "proxy-" + proxyName + "-" + task;
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
			case "alicloud-fc":
				return await this.alicloudFcRequest<R>(url, proxy.data);
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
			if (!this.proxyLimiters[proxyLimiterId]) {
				const providerLimiter = this.providerLimiters[providerLimiterId];
				const providerAvailable = await providerLimiter.getAvailability();
				return providerAvailable;
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

	private async alicloudFcRequest<R>(url: string, region: string): Promise<R> {
		try {
			const decoder = new TextDecoder();
			const output = await new Deno.Command("aliyun", {
				args: [
					"fc",
					"POST",
					`/2023-03-30/functions/proxy-${region}/invocations`,
					"--qualifier",
					"LATEST",
					"--header",
					"Content-Type=application/json;x-fc-invocation-type=Sync;x-fc-log-type=None;",
					"--body",
					JSON.stringify({ url: url }),
					"--retry-count",
					"5",
					"--read-timeout",
					"30",
					"--connect-timeout",
					"10",
					"--profile",
					`CVSA-${region}`,
				],
			}).output();
			const out = decoder.decode(output.stdout);
			const rawData = JSON.parse(out);
			if (rawData.statusCode !== 200) {
				throw new NetSchedulerError(
					`Error proxying ${url} to ali-fc region ${region}, code: ${rawData.statusCode}.`,
					"ALICLOUD_PROXY_ERR",
				);
			} else {
				return JSON.parse(JSON.parse(rawData.body)) as R;
			}
		} catch (e) {
			logger.error(e as Error, "net", "fn:alicloudFcRequest");
			throw new NetSchedulerError(`Unhandled error: Cannot proxy ${url} to ali-fc.`, "ALICLOUD_PROXY_ERR", e);
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
		max: 6,
	},
	{
		window: new SlidingWindow(redis, 5),
		max: 20,
	},
	{
		window: new SlidingWindow(redis, 30),
		max: 100,
	},
	{
		window: new SlidingWindow(redis, 5 * 60),
		max: 200,
	},
];

/*
Execution order for setup:

1. addProxy(proxyName, type, data):
   - Must be called first. Registers proxies in the system, making them available for tasks.
   - Define all proxies before proceeding to define tasks or set up limiters.
2. addTask(taskName, provider, proxies):
   - Call after addProxy. Defines tasks and associates them with providers and proxies.
   - Relies on proxies being already added.
   - Must be called before setting task-specific or provider-specific limiters.
3. setTaskLimiter(taskName, config):
   - Call after addProxy and addTask. Configures rate limiters specifically for tasks and their associated proxies.
   - Depends on tasks and proxies being defined to apply limiters correctly.
4. setProviderLimiter(providerName, config):
   - Call after addProxy and addTask.
   - It sets rate limiters at the provider level, affecting all proxies used by tasks of that provider.
   - Depends on tasks and proxies being defined to identify which proxies to apply provider-level limiters to.

In summary: addProxy -> addTask -> (setTaskLimiter and/or setProviderLimiter).
The order of setTaskLimiter and setProviderLimiter relative to each other is flexible,
but both should come after addProxy and addTask to ensure proper setup and dependencies are met.
*/

const regions = ["shanghai", "hangzhou", "qingdao", "beijing", "zhangjiakou", "chengdu", "shenzhen", "hohhot"];
netScheduler.addProxy("native", "native", "");
for (const region of regions) {
	netScheduler.addProxy(`alicloud-${region}`, "alicloud-fc", region);
}
netScheduler.addTask("getVideoInfo", "bilibili", "all");
netScheduler.addTask("getLatestVideos", "bilibili", "all");
netScheduler.addTask("snapshotMilestoneVideo", "bilibili", regions.map((region) => `alicloud-${region}`));
netScheduler.addTask("snapshotVideo", "bilibili", [
	"alicloud-qingdao",
	"alicloud-shanghai",
	"alicloud-zhangjiakou",
	"alicloud-chengdu",
	"alicloud-shenzhen",
	"alicloud-hohhot",
]);
netScheduler.setTaskLimiter("getVideoInfo", videoInfoRateLimiterConfig);
netScheduler.setTaskLimiter("getLatestVideos", null);
netScheduler.setTaskLimiter("snapshotMilestoneVideo", null);
netScheduler.setTaskLimiter("snapshotVideo", videoInfoRateLimiterConfig);
netScheduler.setProviderLimiter("bilibili", biliLimiterConfig);

export default netScheduler;
