// noinspection ExceptionCaughtLocallyJS

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

type ProxyType = "native" | "alicloud-fc" | "ip-proxy";

interface FCResponse {
	statusCode: number;
	body: string;
	serverTime: number;
}

interface NativeProxyData {
}

interface AlicloudFcProxyData {
	region: string;
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

type ProxyData = NativeProxyData | AlicloudFcProxyData | IPProxyConfig;

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

interface ProviderDef {
	limiters: readonly RateLimiterConfig[];
}

interface TaskDef<ProxyKeys extends string = string, ProviderKeys extends string = string> {
	provider: ProviderKeys;
	proxies: readonly ProxyKeys[] | "all";
	limiters?: readonly RateLimiterConfig[];
}

interface NetworkConfig {
	proxies: Record<string, ProxyDef>;
	providers: Record<string, ProviderDef>;
	tasks: Record<string, TaskDef<any, any>>;
}

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
			strategy: config.strategy ?? "single-use",
			minPoolSize: config.minPoolSize ?? 5,
			maxPoolSize: config.maxPoolSize ?? 50,
			refreshInterval: config.refreshInterval ?? 30_000,
			initialPoolSize: config.initialPoolSize ?? 10
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
			logger.debug("Refreshing IP pool", "net", "IPPoolManager.refreshPool");

			const extractedIPs = await this.config.extractor();
			const newIPs = extractedIPs.slice(0, this.config.maxPoolSize - this.pool.length);

			// Add new IPs to pool
			for (const ipData of newIPs) {
				const ipEntry: IPEntry = {
					...ipData,
					createdAt: Date.now(),
					used: false
				};
				this.pool.push(ipEntry);
			}

			logger.debug(
				`IP pool refreshed. Pool size: ${this.pool.length}`,
				"net",
				"IPPoolManager.refreshPool"
			);
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
				this.ipPools[proxyName].initialize().catch(error => {
					logger.error(error as Error, "net", `Failed to initialize IP pool for ${proxyName}`);
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
				proxies: [...targetProxies]
			};

			if (taskDef.limiters && taskDef.limiters.length > 0) {
				for (const proxyName of targetProxies) {
					const limiterId = `proxy-${proxyName}-${taskName}`;
					this.proxyLimiters[limiterId] = new MultipleRateLimiter(limiterId, [
						...taskDef.limiters
					]);
				}
			}
		}

		for (const [providerName, providerDef] of Object.entries(config.providers)) {
			if (!providerDef.limiters || providerDef.limiters.length === 0) continue;

			const boundProxies = new Set<string>();
			for (const [_taskName, taskImpl] of Object.entries(this.tasks)) {
				if (taskImpl.provider === providerName) {
					taskImpl.proxies.forEach((p) => boundProxies.add(p));
				}
			}

			for (const proxyName of boundProxies) {
				const limiterId = `provider-${proxyName}-${providerName}`;
				if (!this.providerLimiters[limiterId]) {
					this.providerLimiters[limiterId] = new MultipleRateLimiter(limiterId, [
						...providerDef.limiters
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
					throw new NetSchedulerError("Invalid alicloud-fc proxy configuration", "ALICLOUD_PROXY_ERR");
				}
				return await this.alicloudFcRequest<R>(url, proxy.data);
			case "ip-proxy":
				if (!isIpProxy(proxy)) {
					throw new NetSchedulerError("Invalid ip-proxy configuration", "NOT_IMPLEMENTED");
				}
				return await this.ipProxyRequest<R>(url, proxy);
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
					time: rawData.serverTime
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
		const proxyName = Object.entries(this.proxies).find(([_, proxy]) => proxy === proxyDef)?.[0];
		if (!proxyName || !this.ipPools[proxyName]) {
			throw new NetSchedulerError("IP pool not found", "IP_POOL_EXHAUSTED");
		}

		const ipPool = this.ipPools[proxyName];
		const ipEntry = await ipPool.getNextIP();

		if (!ipEntry) {
			throw new NetSchedulerError("No IP available in pool", "IP_POOL_EXHAUSTED");
		}

		try {
			const controller = new AbortController();
			const now = Date.now();
			const timeout = setTimeout(() => controller.abort(), ipEntry.lifespan - (now - ipEntry.createdAt));

			const response = await fetch(url, {
				signal: controller.signal,
				proxy: `http://${ipEntry.address}:${ipEntry.port}`
			});

			clearTimeout(timeout);

			const start = Date.now();
			const data = await response.json();
			const end = Date.now();
			const serverTime = start + (end - start) / 2;

			// Mark IP as used
			await ipPool.markIPUsed(ipEntry.address);

			return { data: data as R, time: serverTime };
		} catch (error) {
			// Mark IP as used even if request failed (single-use strategy)
			await ipPool.markIPUsed(ipEntry.address);
			throw new NetSchedulerError("IP proxy request failed", "IP_EXTRACTION_FAILED", error);
		}
	}
}

const biliLimiterConfig: RateLimiterConfig[] = [
	{ duration: 1, max: 20 },
	{ duration: 15, max: 130 },
	{ duration: 5 * 60, max: 2000 }
];

const bili_normal = structuredClone(biliLimiterConfig);
bili_normal[0].max = 5;
bili_normal[1].max = 40;
bili_normal[2].max = 200;

const bili_strict = structuredClone(biliLimiterConfig);
bili_strict[0].max = 1;
bili_strict[1].max = 6;
bili_strict[2].max = 100;

const aliRegions = ["hangzhou"] as const;

const proxies = {
	native: {
		type: "native" as const,
		data: {}
	},

	alicloud_hangzhou: {
		type: "alicloud-fc" as const,
		data: {
			region: "hangzhou",
			timeout: 15000
		}
	},

	ip_proxy_pool: {
		type: "ip-proxy" as const,
		data: {
			extractor: async (): Promise<IPEntry[]> => {
				interface APIResponse {
					code: number;
					data: {
						ip: string;
						port: number;
						endtime: string;
						city: string;
					}[]
				}
				const url = Bun.env.IP_PROXY_EXTRACTOR_URL;
				const response = await fetch(url);
				const data = await response.json() as APIResponse;
				if (data.code !== 0){
					throw new Error(`IP proxy extractor failed with code ${data.code}`);
				}
				const ips = data.data;
				return ips.map((item) => {
					return {
						address: item.ip,
						port: item.port,
						lifespan: Date.parse(item.endtime+'+08') - Date.now(),
						createdAt: Date.now(),
						used: false
					}
				})
			},
			strategy: "round-robin",
			minPoolSize: 10,
			maxPoolSize: 500,
			refreshInterval: 5 * SECOND,
			initialPoolSize: 10
		}
	}
} satisfies Record<string, ProxyDef>;

type MyProxyKeys = keyof typeof proxies;

const fcProxies = aliRegions.map((region) => `alicloud_${region}`) as MyProxyKeys[];

const config = {
	proxies: proxies,
	providers: {
		test: { limiters: [] },
		bilibili: { limiters: biliLimiterConfig }
	},
	tasks: {
		test: {
			provider: "test",
			proxies: fcProxies
		},
		test_ip: {
			provider: "test",
			proxies: ["ip_proxy_pool"]
		},
		getVideoInfo: {
			provider: "bilibili",
			proxies: "all",
			limiters: bili_strict
		},
		getLatestVideos: {
			provider: "bilibili",
			proxies: "all",
			limiters: bili_strict
		},
		snapshotMilestoneVideo: {
			provider: "bilibili",
			proxies: ["ip_proxy_pool"]
		},
		snapshotVideo: {
			provider: "bilibili",
			proxies: ["ip_proxy_pool"],
			limiters: bili_normal
		},
		bulkSnapshot: {
			provider: "bilibili",
			proxies: ["ip_proxy_pool"],
			limiters: bili_strict
		}
	}
} as const satisfies NetworkConfig;

export type RequestTasks = keyof typeof config.tasks;

const networkDelegate = new NetworkDelegate(config);

export default networkDelegate;
