import logger from "@core/log";
import {
	MultipleRateLimiter,
	type RateLimiterConfig,
	RateLimiterError
} from "@core/mq/multipleRateLimiter";
import { ReplyError } from "ioredis";
import { SECOND } from "@core/lib";
import FC20230330, * as $FC20230330 from "@alicloud/fc20230330";
import Credential from "@alicloud/credentials";
import * as OpenApi from "@alicloud/openapi-client";
import Stream from "@alicloud/darabonba-stream";
import * as Util from "@alicloud/tea-util";
import { Readable } from "stream";

type ProxyType = "native" | "alicloud-fc" | "baidu-cfc";

interface FCResponse {
	statusCode: number;
	body: string;
	serverTime: number;
}

interface Proxy {
	type: ProxyType;
	data: string;
}

interface Task {
	provider: string;
	proxies: string[] | "all";
}

interface ProxiesMap {
	[name: string]: Proxy;
}

type NetworkDelegateErrorCode =
	| "NO_PROXY_AVAILABLE"
	| "PROXY_RATE_LIMITED"
	| "PROXY_NOT_FOUND"
	| "FETCH_ERROR"
	| "NOT_IMPLEMENTED"
	| "ALICLOUD_PROXY_ERR";

export class NetSchedulerError extends Error {
	public code: NetworkDelegateErrorCode;
	public rawError: unknown | undefined;
	constructor(message: string, errorCode: NetworkDelegateErrorCode, rawError?: unknown) {
		super(message);
		this.name = "NetSchedulerError";
		this.code = errorCode;
		this.rawError = rawError;
	}
}

type LimiterMap = {
	[name: string]: MultipleRateLimiter;
};

type OptionalLimiterMap = {
	[name: string]: MultipleRateLimiter | null;
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

const getEndpoint = (region: string) => `fcv3.cn-${region}.aliyuncs.com`;

const getAlicloudClient = (region: string) => {
	const credential = new Credential();
	const config = new OpenApi.Config({
		credential: credential
	});
	config.endpoint = getEndpoint(region);
	return new FC20230330(config);
};

const streamToString = async (readableStream: Readable) => {
	let data = "";
	for await (const chunk of readableStream) {
		data += chunk.toString();
	}
	return data;
};

class NetworkDelegate {
	private proxies: ProxiesMap = {};
	private providerLimiters: LimiterMap = {};
	private proxyLimiters: OptionalLimiterMap = {};
	private tasks: TaskMap = {};

	addProxy(proxyName: string, type: ProxyType, data: string): void {
		this.proxies[proxyName] = { type, data };
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
			this.proxyLimiters[limiterId] = config
				? new MultipleRateLimiter(limiterId, config)
				: null;
		}
	}

	async triggerLimiter(task: string, proxy: string, force: boolean = false): Promise<void> {
		const limiterId = "proxy-" + proxy + "-" + task;
		const providerLimiterId = "provider-" + proxy + "-" + this.tasks[task].provider;
		try {
			await this.proxyLimiters[limiterId]?.trigger(!force);
			await this.providerLimiters[providerLimiterId]?.trigger(!force);
		} catch (e) {
			const error = e as Error;
			if (e instanceof ReplyError) {
				logger.error(error, "redis", "fn:triggerLimiter");
			} else if (e instanceof RateLimiterError) {
				// Re-throw it to ensure this.request can catch it
				throw e;
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
			this.providerLimiters[limiterId] = new MultipleRateLimiter(limiterId, config);
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
	async request<R>(
		url: string,
		task: string
	): Promise<{
		data: R;
		time: number;
	}> {
		// find a available proxy
		const proxiesNames = this.getTaskProxies(task);
		for (const proxyName of shuffleArray(proxiesNames)) {
			try {
				return await this.proxyRequest<R>(url, proxyName, task);
			} catch (e) {
				if (e instanceof RateLimiterError) {
					continue;
				}
				throw e;
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
		force: boolean = false
	): Promise<{
		data: R;
		time: number;
	}> {
		const proxy = this.proxies[proxyName];
		if (!proxy) {
			throw new NetSchedulerError(`Proxy "${proxyName}" not found`, "PROXY_NOT_FOUND");
		}

		await this.triggerLimiter(task, proxyName, force);
		return this.makeRequest<R>(url, proxy);
	}

	private async makeRequest<R>(
		url: string,
		proxy: Proxy
	): Promise<{
		data: R;
		time: number;
	}> {
		switch (proxy.type) {
			case "native":
				return await this.nativeRequest<R>(url);
			case "alicloud-fc":
				return await this.alicloudFcRequest<R>(url, proxy.data);
			default:
				throw new NetSchedulerError(
					`Proxy type ${proxy.type} not supported`,
					"NOT_IMPLEMENTED"
				);
		}
	}

	private async nativeRequest<R>(url: string): Promise<{
		data: R;
		time: number;
	}> {
		try {
			const controller = new AbortController();
			const timeout = setTimeout(() => controller.abort(), 10 * SECOND);

			const response = await fetch(url, {
				signal: controller.signal
			});

			clearTimeout(timeout);

			const start = Date.now();
			const data = await response.json();
			const end = Date.now();
			const serverTime = start + (end - start) / 2;
			return {
				data: data as R,
				time: serverTime
			};
		} catch (e) {
			throw new NetSchedulerError("Fetch error", "FETCH_ERROR", e);
		}
	}

	private async alicloudFcRequest<R>(
		url: string,
		region: string
	): Promise<{
		data: R;
		time: number;
	}> {
		try {
			const client = getAlicloudClient(region);
			const bodyStream = Stream.readFromString(JSON.stringify({ url: url }));
			const headers = new $FC20230330.InvokeFunctionHeaders({});
			const request = new $FC20230330.InvokeFunctionRequest({
				body: bodyStream
			});
			const runtime = new Util.RuntimeOptions({});
			const response = await client.invokeFunctionWithOptions(
				`proxy-${region}`,
				request,
				headers,
				runtime
			);
			if (response.statusCode !== 200) {
				// noinspection ExceptionCaughtLocallyJS
				throw new NetSchedulerError(
					`Error proxying ${url} to ali-fc region ${region}, code: ${response.statusCode} (Not correctly invoked).`,
					"ALICLOUD_PROXY_ERR"
				);
			}
			const rawData = JSON.parse(await streamToString(response.body)) as FCResponse;
			if (rawData.statusCode !== 200) {
				// noinspection ExceptionCaughtLocallyJS
				throw new NetSchedulerError(
					`Error proxying ${url} to ali-fc region ${region}, code: ${rawData.statusCode}. (fetch error)`,
					"ALICLOUD_PROXY_ERR"
				);
			} else {
				return {
					data: JSON.parse(rawData.body) as R,
					time: rawData.serverTime
				};
			}
		} catch (e) {
			logger.error(e as Error, "net", "fn:alicloudFcRequest");
			throw new NetSchedulerError(
				`Unhandled error: Cannot proxy ${url} to ali-fc-${region}.`,
				"ALICLOUD_PROXY_ERR",
				e
			);
		}
	}
}

const networkDelegate = new NetworkDelegate();
const videoInfoRateLimiterConfig: RateLimiterConfig[] = [
	{
		duration: 0.3,
		max: 1
	},
	{
		duration: 3,
		max: 5
	},
	{
		duration: 30,
		max: 30
	},
	{
		duration: 2 * 60,
		max: 50
	}
];
const biliLimiterConfig: RateLimiterConfig[] = [
	{
		duration: 1,
		max: 20
	},
	{
		duration: 15,
		max: 130
	},
	{
		duration: 5 * 60,
		max: 2000
	}
];

const bili_normal = [...biliLimiterConfig];
bili_normal[0].max = 5;
bili_normal[1].max = 40;
bili_normal[2].max = 200;

const bili_strict = [...biliLimiterConfig];
bili_strict[0].max = 1;
bili_strict[1].max = 6;
bili_strict[2].max = 100;

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

const aliRegions = ["beijing", "hangzhou"];
const fcProxies = aliRegions.map((region) => `alicloud-${region}`);
const fcProxiesL = aliRegions.slice(1).map((region) => `alicloud-${region}`);
networkDelegate.addProxy("native", "native", "");
for (const region of aliRegions) {
	networkDelegate.addProxy(`alicloud-${region}`, "alicloud-fc", region);
}

networkDelegate.addTask("test", "test", "all");
networkDelegate.addTask("getVideoInfo", "bilibili", "all");
networkDelegate.addTask("getLatestVideos", "bilibili", "all");
networkDelegate.addTask("snapshotMilestoneVideo", "bilibili", fcProxies);
networkDelegate.addTask("snapshotVideo", "bilibili", fcProxiesL);
networkDelegate.addTask("bulkSnapshot", "bilibili", fcProxiesL);

networkDelegate.setTaskLimiter("getVideoInfo", videoInfoRateLimiterConfig);
networkDelegate.setTaskLimiter("bulkSnapshot", bili_strict);
networkDelegate.setTaskLimiter("getLatestVideos", bili_strict);
networkDelegate.setTaskLimiter("getVideoInfo", bili_strict);
networkDelegate.setTaskLimiter("snapshotVideo", bili_normal);
networkDelegate.setProviderLimiter("test", []);
networkDelegate.setProviderLimiter("bilibili", biliLimiterConfig);

export default networkDelegate;
