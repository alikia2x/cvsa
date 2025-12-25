// noinspection ExceptionCaughtLocallyJS

import type { Readable } from "node:stream";
import Credential from "@alicloud/credentials";
import Stream from "@alicloud/darabonba-stream";
import FC20230330, * as $FC20230330 from "@alicloud/fc20230330";
import * as OpenApi from "@alicloud/openapi-client";
import * as Util from "@alicloud/tea-util";
import { SECOND } from "@core/lib";
import logger from "@core/log";
import {
	MultipleRateLimiter,
	type RateLimiterConfig,
	RateLimiterError,
} from "@core/mq/multipleRateLimiter";
import {
	aliFCCounter,
	aliFCErrorCounter,
	ipProxyCounter,
	ipProxyErrorCounter,
} from "crawler/metrics";
import { ReplyError } from "ioredis";

type ProxyType = "native" | "alicloud-fc" | "ip-proxy" | "cf-worker";

const aliRegions = ["hangzhou"] as const;
type AliRegion = (typeof aliRegions)[number];

function createAliProxiesObject<T extends readonly string[]>(regions: T) {
	return regions.reduce(
		(result, currentRegion) => {
			result[`alicloud_${currentRegion}`] = {
				data: {
					region: currentRegion,
					timeout: 15000,
				},
				type: "alicloud-fc" as const,
			} as ProxyDef<AlicloudFcProxyData>;
			return result;
		},
		{} as Record<`alicloud_${AliRegion}`, ProxyDef<AlicloudFcProxyData>>
	);
}

const aliProxiesObject = createAliProxiesObject(aliRegions);
const aliProxies = aliRegions.map((region) => `alicloud_${region}` as `alicloud_${AliRegion}`);

const proxies = {
	native: {
		data: {},
		type: "native" as const,
	},

	...aliProxiesObject,

	"cf-worker": {
		data: {
			url: Bun.env.CF_WORKER_URL,
		},
		type: "cf-worker",
	},
} satisfies Record<string, ProxyDef>;

interface FCResponse {
	statusCode: number;
	body: string;
	serverTime: number;
}

type NativeProxyData = Record<string, never>;

interface AlicloudFcProxyData {
	region: (typeof aliRegions)[number];
	timeout?: number;
}

interface CFFcProxyData {
	url: string;
	timeout?: number;
}

// New IP proxy system interfaces
interface IPEntry {
	address: string;
	/*
	Lifespan of this IP addressin milliseconds
	 */
	lifespan: number;
	port?: number;
	/*
	When this IP was created, UNIX timestamp in milliseconds
	 */
	createdAt: number;
	used: boolean;
}

type IPExtractor = () => Promise<IPEntry[]>;

type IPRotationStrategy = "single-use" | "round-robin" | "random";

interface IPProxyConfig {
	extractor: IPExtractor;
	strategy?: IPRotationStrategy; // defaults to "single-use"
	minPoolSize?: number; // minimum IPs to maintain (default: 5)
	maxPoolSize?: number; // maximum IPs to cache (default: 50)
	refreshInterval?: number; // how often to check for new IPs (default: 30s)
	initialPoolSize?: number; // how many IPs to fetch initially (default: 10)
}

type ProxyData = NativeProxyData | AlicloudFcProxyData | IPProxyConfig | CFFcProxyData;

interface ProxyDef<T extends ProxyData = ProxyData> {
	type: ProxyType;
	data: T;
}

function isAlicloudFcProxy(proxy: ProxyDef): proxy is ProxyDef<AlicloudFcProxyData> {
	return proxy.type === "alicloud-fc";
}

function isIpProxy(proxy: ProxyDef): proxy is ProxyDef<IPProxyConfig> {
	return proxy.type === "ip-proxy";
}

function isCFFcProxy(proxy: ProxyDef): proxy is ProxyDef<CFFcProxyData> {
	return proxy.type === "cf-worker";
}

interface ProviderDef {
	limiters: readonly RateLimiterConfig[];
}

type ProxyName = keyof typeof proxies;

interface TaskDef<ProviderKeys extends string = string> {
	provider: ProviderKeys;
	proxies: readonly ProxyName[] | "all";
	limiters?: readonly RateLimiterConfig[];
}

type NetworkConfigInternal<ProviderKeys extends string, TaskKeys extends string> = {
	proxies: Record<string, ProxyDef>;
	providers: Record<ProviderKeys, ProviderDef>;
	tasks: Record<TaskKeys, TaskDef<ProviderKeys>>;
};

const biliLimiterConfig: RateLimiterConfig[] = [
	{ duration: 1, max: 20 },
	{ duration: 15, max: 130 },
	{ duration: 5 * 60, max: 2000 },
];

const bili_normal = structuredClone(biliLimiterConfig);
bili_normal[0].max = 5;
bili_normal[1].max = 40;
bili_normal[2].max = 200;

const bili_strict = structuredClone(biliLimiterConfig);
bili_strict[0].max = 1;
bili_strict[1].max = 6;
bili_strict[2].max = 100;

type MyProxyKeys = keyof typeof proxies;

const fcProxies = aliRegions.map((region) => `alicloud_${region}`) as MyProxyKeys[];

function createNetworkConfig<ProviderKeys extends string, TaskKeys extends string>(
	config: NetworkConfigInternal<ProviderKeys, TaskKeys>
): NetworkConfigInternal<ProviderKeys, TaskKeys> {
	return config;
}

const config = createNetworkConfig({
	providers: {
		bilibili: { limiters: [] },
		test: { limiters: [] },
		testCF: { limiters: [] },
	},
	proxies: proxies,
	tasks: {
		bulkSnapshot: {
			provider: "bilibili",
			proxies: ["cf-worker"],
		},
		getLatestVideos: {
			limiters: bili_strict,
			provider: "bilibili",
			proxies: "all",
		},
		getVideoInfo: {
			limiters: bili_strict,
			provider: "bilibili",
			proxies: "all",
		},
		snapshotMilestoneVideo: {
			provider: "bilibili",
			proxies: ["cf-worker"],
		},
		snapshotVideo: {
			provider: "bilibili",
			proxies: ["cf-worker"],
		},
		test: {
			provider: "test",
			proxies: fcProxies,
		},
		testCf: {
			provider: "testCF",
			proxies: ["cf-worker"],
		},
	},
});

type NetworkConfig = typeof config;

export type RequestTasks = keyof typeof config.tasks;

type NetworkDelegateErrorCode =
	| "NO_PROXY_AVAILABLE"
	| "PROXY_RATE_LIMITED"
	| "PROXY_NOT_FOUND"
	| "FETCH_ERROR"
	| "NOT_IMPLEMENTED"
	| "ALICLOUD_PROXY_ERR"
	| "IP_POOL_EXHAUSTED"
	| "IP_EXTRACTION_FAILED";

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

function shuffleArray<T>(array: readonly T[]): T[] {
	const newArray = [...array];
	for (let i = newArray.length - 1; i > 0; i--) {
		const j = Math.floor(Math.random() * (i + 1));
		[newArray[i], newArray[j]] = [newArray[j], newArray[i]];
	}
	return newArray;
}

class IPPoolManager {
	private pool: IPEntry[] = [];
	private readonly config: Required<IPProxyConfig>;
	protected refreshTimer: NodeJS.Timeout;
	private isRefreshing = false;

	constructor(config: IPProxyConfig) {
		this.config = {
			extractor: config.extractor,
			initialPoolSize: config.initialPoolSize ?? 10,
			maxPoolSize: config.maxPoolSize ?? 50,
			minPoolSize: config.minPoolSize ?? 5,
			refreshInterval: config.refreshInterval ?? 30_000,
			strategy: config.strategy ?? "single-use",
		};
	}

	async initialize(): Promise<void> {
		await this.refreshPool();
		this.startPeriodicRefresh();
	}

	private startPeriodicRefresh(): void {
		this.refreshTimer = setInterval(async () => {
			await this.refreshPool();
		}, this.config.refreshInterval);
	}

	async getNextIP(): Promise<IPEntry | null> {
		// Clean expired IPs first
		this.cleanExpiredIPs();

		// Try to get available IP based on strategy
		let selectedIP: IPEntry | null = null;

		switch (this.config.strategy) {
			case "single-use":
				selectedIP = this.getAvailableIP();
				break;
			case "round-robin":
				selectedIP = this.getRoundRobinIP();
				break;
			case "random":
				selectedIP = this.getRandomIP();
				break;
		}

		// If no IP available and pool is low, try to refresh
		if (!selectedIP && this.pool.length < this.config.minPoolSize) {
			await this.refreshPool();
			selectedIP = this.getAvailableIP();
		}

		return selectedIP;
	}

	private getAvailableIP(): IPEntry | null {
		const availableIPs = this.pool.filter((ip) => !ip.used);
		if (availableIPs.length === 0) return null;

		// For single-use, mark IP as used immediately
		const selectedIP = availableIPs[0];
		selectedIP.used = true;
		return selectedIP;
	}

	private getRoundRobinIP(): IPEntry | null {
		const availableIPs = this.pool.filter((ip) => !ip.used);
		if (availableIPs.length === 0) return null;

		const selectedIP = availableIPs[0];
		selectedIP.used = true;
		return selectedIP;
	}

	private getRandomIP(): IPEntry | null {
		const availableIPs = this.pool.filter((ip) => !ip.used);
		if (availableIPs.length === 0) return null;

		const randomIndex = Math.floor(Math.random() * availableIPs.length);
		const selectedIP = availableIPs[randomIndex];
		selectedIP.used = true;
		return selectedIP;
	}

	private cleanExpiredIPs(): void {
		const now = Date.now();
		this.pool = this.pool.filter((ip) => {
			const expiryTime = ip.createdAt + ip.lifespan;
			return expiryTime > now;
		});
	}

	private async refreshPool(): Promise<void> {
		if (this.isRefreshing) return;

		this.isRefreshing = true;
		try {
			const extractedIPs = await this.config.extractor();
			const newIPs = extractedIPs.slice(0, this.config.maxPoolSize - this.pool.length);

			// Add new IPs to pool
			for (const ipData of newIPs) {
				const ipEntry: IPEntry = {
					...ipData,
					createdAt: Date.now(),
					used: false,
				};
				this.pool.push(ipEntry);
			}
		} catch (error) {
			logger.error(error as Error, "net", "IPPoolManager.refreshPool");
		} finally {
			this.isRefreshing = false;
		}
	}

	async markIPUsed(address: string): Promise<void> {
		const ip = this.pool.find((p) => p.address === address);
		if (ip) {
			ip.used = true;
		}
	}
}

const getEndpoint = (region: string) => `fcv3.cn-${region}.aliyuncs.com`;

const getAlicloudClient = (region: string) => {
	const credential = new Credential();
	const config = new OpenApi.Config({ credential: credential });
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

export class NetworkDelegate<const C extends NetworkConfig> {
	private readonly proxies: Record<string, ProxyDef>;
	private readonly tasks: Record<string, { provider: string; proxies: string[] }>;
	private readonly ipPools: Record<string, IPPoolManager> = {};

	private providerLimiters: Record<string, MultipleRateLimiter> = {};
	private proxyLimiters: Record<string, MultipleRateLimiter> = {};

	constructor(config: C) {
		this.proxies = config.proxies;
		this.tasks = {};
		this.ipPools = {};

		// Initialize IP pools for ip-proxy configurations
		for (const [proxyName, proxyDef] of Object.entries(this.proxies)) {
			if (isIpProxy(proxyDef)) {
				this.ipPools[proxyName] = new IPPoolManager(proxyDef.data);
				// Initialize asynchronously but don't wait
				this.ipPools[proxyName].initialize().catch((error) => {
					logger.error(
						error as Error,
						"net",
						`Failed to initialize IP pool for ${proxyName}`
					);
				});
			}
		}

		const allProxyNames = Object.keys(this.proxies);

		for (const [taskName, taskDef] of Object.entries(config.tasks)) {
			const targetProxies =
				taskDef.proxies === "all" ? allProxyNames : (taskDef.proxies as readonly string[]);

			for (const p of targetProxies) {
				if (!this.proxies[p]) {
					throw new Error(`Task ${taskName} references missing proxy: ${p}`);
				}
			}

			this.tasks[taskName] = {
				provider: taskDef.provider,
				proxies: [...targetProxies],
			};

			if (taskDef.limiters && taskDef.limiters.length > 0) {
				for (const proxyName of targetProxies) {
					const limiterId = `proxy-${proxyName}-${taskName}`;
					this.proxyLimiters[limiterId] = new MultipleRateLimiter(limiterId, [
						...taskDef.limiters,
					]);
				}
			}
		}

		for (const [providerName, providerDef] of Object.entries(config.providers)) {
			if (!providerDef.limiters || providerDef.limiters.length === 0) continue;

			const boundProxies = new Set<string>();
			for (const [_taskName, taskImpl] of Object.entries(this.tasks)) {
				if (taskImpl.provider === providerName) {
					taskImpl.proxies.forEach((p) => {
						boundProxies.add(p);
					});
				}
			}

			for (const proxyName of boundProxies) {
				const limiterId = `provider-${proxyName}-${providerName}`;
				if (!this.providerLimiters[limiterId]) {
					this.providerLimiters[limiterId] = new MultipleRateLimiter(limiterId, [
						...providerDef.limiters,
					]);
				}
			}
		}
	}

	private async triggerLimiter(
		taskName: string,
		proxyName: string,
		force: boolean = false
	): Promise<void> {
		const taskImpl = this.tasks[taskName];
		if (!taskImpl) return;

		const proxyLimiterId = `proxy-${proxyName}-${taskName}`;
		const providerLimiterId = `provider-${proxyName}-${taskImpl.provider}`;

		try {
			if (this.proxyLimiters[proxyLimiterId]) {
				await this.proxyLimiters[proxyLimiterId].trigger(!force);
			}
			if (this.providerLimiters[providerLimiterId]) {
				await this.providerLimiters[providerLimiterId].trigger(!force);
			}
		} catch (e) {
			const error = e as Error;
			if (e instanceof ReplyError) {
				logger.error(error, "redis", "fn:triggerLimiter");
			} else if (e instanceof RateLimiterError) {
				throw e;
			} else {
				logger.warn(`Unhandled error: ${error.message}`, "mq", "proxyRequest");
			}
		}
	}

	async request<R>(url: string, task: keyof C["tasks"]): Promise<{ data: R; time: number }> {
		const taskName = task as string;
		const taskImpl = this.tasks[taskName];

		if (!taskImpl) {
			throw new Error(`Task definition missing for ${taskName}`);
		}

		const proxiesNames = taskImpl.proxies;

		for (const proxyName of shuffleArray(proxiesNames)) {
			try {
				return await this.proxyRequest<R>(url, proxyName, taskName);
			} catch (e) {
				if (e instanceof RateLimiterError) {
					continue;
				}
				throw e;
			}
		}
		throw new NetSchedulerError("No proxy is available currently.", "NO_PROXY_AVAILABLE");
	}

	async proxyRequest<R>(
		url: string,
		proxyName: string,
		task: string,
		force: boolean = false
	): Promise<{ data: R; time: number }> {
		const proxy = this.proxies[proxyName];
		if (!proxy) {
			throw new NetSchedulerError(`Proxy "${proxyName}" not found`, "PROXY_NOT_FOUND");
		}

		await this.triggerLimiter(task, proxyName, force);
		return this.makeRequest<R>(url, proxy);
	}

	private async makeRequest<R>(url: string, proxy: ProxyDef): Promise<{ data: R; time: number }> {
		switch (proxy.type) {
			case "native":
				return await this.nativeRequest<R>(url);
			case "alicloud-fc":
				if (!isAlicloudFcProxy(proxy)) {
					throw new NetSchedulerError(
						"Invalid alicloud-fc proxy configuration",
						"ALICLOUD_PROXY_ERR"
					);
				}
				try {
					return await this.alicloudFcRequest<R>(url, proxy.data);
				} catch (e) {
					aliFCErrorCounter.add(1);
					throw e;
				} finally {
					aliFCCounter.add(1);
				}
			case "ip-proxy":
				if (!isIpProxy(proxy)) {
					throw new NetSchedulerError(
						"Invalid ip-proxy configuration",
						"NOT_IMPLEMENTED"
					);
				}
				try {
					return await this.ipProxyRequest<R>(url, proxy);
				} catch (e) {
					ipProxyErrorCounter.add(1);
					throw e;
				} finally {
					ipProxyCounter.add(1);
				}
			case "cf-worker":
				if (!isCFFcProxy(proxy)) {
					throw new NetSchedulerError(
						"Invalid cf-worker proxy configuration",
						"NOT_IMPLEMENTED"
					);
				}
				return await this.cfWorkerRequest<R>(url, proxy.data);
			default:
				throw new NetSchedulerError(
					`Proxy type ${proxy.type} not supported`,
					"NOT_IMPLEMENTED"
				);
		}
	}

	private async nativeRequest<R>(url: string): Promise<{ data: R; time: number }> {
		try {
			const controller = new AbortController();
			const timeout = setTimeout(() => controller.abort(), 10 * SECOND);

			const response = await fetch(url, { signal: controller.signal });
			clearTimeout(timeout);

			const start = Date.now();
			const data = await response.json();
			const end = Date.now();
			const serverTime = start + (end - start) / 2;
			return { data: data as R, time: serverTime };
		} catch (e) {
			throw new NetSchedulerError("Fetch error", "FETCH_ERROR", e);
		}
	}

	private async alicloudFcRequest<R>(
		url: string,
		proxyData: AlicloudFcProxyData
	): Promise<{ data: R; time: number }> {
		try {
			const client = getAlicloudClient(proxyData.region);
			const bodyStream = Stream.readFromString(JSON.stringify({ url: url }));
			const headers = new $FC20230330.InvokeFunctionHeaders({});
			const request = new $FC20230330.InvokeFunctionRequest({ body: bodyStream });
			const runtime = new Util.RuntimeOptions({});

			const response = await client.invokeFunctionWithOptions(
				`proxy-${proxyData.region}`,
				request,
				headers,
				runtime
			);

			if (response.statusCode !== 200) {
				throw new NetSchedulerError(
					`Error proxying ${url} to ali-fc region ${proxyData.region}, code: ${response.statusCode}`,
					"ALICLOUD_PROXY_ERR"
				);
			}

			const rawData = JSON.parse(await streamToString(response.body)) as FCResponse;
			if (rawData.statusCode !== 200) {
				throw new NetSchedulerError(
					`Error proxying ${url} to ali-fc region ${proxyData.region}, remote code: ${rawData.statusCode}`,
					"ALICLOUD_PROXY_ERR"
				);
			} else {
				return {
					data: JSON.parse(rawData.body) as R,
					time: rawData.serverTime,
				};
			}
		} catch (e) {
			logger.error(e as Error, "net", "fn:alicloudFcRequest");
			throw new NetSchedulerError(
				`Unhandled error: Cannot proxy ${url} to ali-fc-${proxyData.region}.`,
				"ALICLOUD_PROXY_ERR",
				e
			);
		}
	}

	private async ipProxyRequest<R>(
		url: string,
		proxyDef: ProxyDef<IPProxyConfig>
	): Promise<{ data: R; time: number }> {
		const proxyName = Object.entries(this.proxies).find(
			([_, proxy]) => proxy === proxyDef
		)?.[0];
		if (!proxyName || !this.ipPools[proxyName]) {
			throw new NetSchedulerError("IP pool not found", "IP_POOL_EXHAUSTED");
		}

		const ipPool = this.ipPools[proxyName];
		const maxRetries = 5;

		let lastError: Error | null = null;

		for (let attempt = 0; attempt < maxRetries; attempt++) {
			const ipEntry = await ipPool.getNextIP();

			if (!ipEntry) {
				throw new NetSchedulerError("No IP available in pool", "IP_POOL_EXHAUSTED");
			}

			try {
				const controller = new AbortController();
				const now = Date.now();
				const timeout = setTimeout(
					() => controller.abort(),
					ipEntry.lifespan - (now - ipEntry.createdAt)
				);

				const response = await fetch(url, {
					proxy: `http://${ipEntry.address}:${ipEntry.port}`,
					signal: controller.signal,
				});

				clearTimeout(timeout);

				const start = Date.now();
				const data = await response.json();
				const end = Date.now();
				const serverTime = start + (end - start) / 2;

				return { data: data as R, time: serverTime };
			} catch (error) {
				lastError = error as Error;

				// If this is not the last attempt, retry immediately
				if (attempt < maxRetries - 1) {
					continue;
				}

				throw new NetSchedulerError(
					"IP proxy request failed",
					"IP_EXTRACTION_FAILED",
					error
				);
			} finally {
				await ipPool.markIPUsed(ipEntry.address);
			}
		}

		throw new NetSchedulerError(
			"IP proxy request failed after all retries",
			"IP_EXTRACTION_FAILED",
			lastError
		);
	}

	private async cfWorkerRequest<R>(
		url: string,
		proxyData: CFFcProxyData
	): Promise<{ data: R; time: number }> {
		const cfClientId = process.env.CF_CLIENT_ID;
		const cfClientSecret = process.env.CF_CLIENT_SECRET;

		if (!cfClientId || !cfClientSecret) {
			throw new NetSchedulerError(
				"CF_ACCESS_CLIENT_ID or CF_ACCESS_CLIENT_SECRET not configured",
				"FETCH_ERROR"
			);
		}

		const payload = {
			headers: {
				Accept: "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
				"Accept-Encoding": "identity",
				"Accept-Language": "zh,en-US;q=0.9,en;q=0.8,zh-CN;q=0.7",
				Origin: "https://www.bilibili.com",
				referer: "https://www.bilibili.com",
				"User-Agent":
					"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36",
			},
			url: url,
		};

		const headers = {
			"CF-Access-Client-Id": cfClientId,
			"CF-Access-Client-Secret": cfClientSecret,
			"Content-Type": "application/json",
		};

		try {
			const controller = new AbortController();
			const timeout = setTimeout(() => controller.abort(), proxyData.timeout ?? 15000);

			const response = await fetch(proxyData.url, {
				body: JSON.stringify(payload),
				headers: headers,
				method: "POST",
				signal: controller.signal,
			});

			clearTimeout(timeout);

			const responseData = (await response.json()) as { data: string; time: number };

			if (!response.ok) {
				throw new NetSchedulerError(
					`CF Worker request failed: ${response.statusText}`,
					"FETCH_ERROR"
				);
			}

			return { data: JSON.parse(responseData.data) as R, time: responseData.time };
		} catch (e) {
			if (e instanceof NetSchedulerError) {
				throw e;
			}
			throw new NetSchedulerError("CF Worker request failed", "FETCH_ERROR", e);
		}
	}
}

const networkDelegate = new NetworkDelegate(config);

export default networkDelegate;
