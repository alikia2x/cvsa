import logger from "lib/log/logger.ts";
import {RateLimiter} from "lib/mq/rateLimiter.ts";

interface Proxy {
	type: string;
	task: string;
	limiter?: RateLimiter;
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
	public errorCode: NetSchedulerErrorCode;
	constructor(message: string, errorCode: NetSchedulerErrorCode) {
		super(message);
		this.name = "NetSchedulerError";
		this.errorCode = errorCode;
	}
}

export class NetScheduler {
	private proxies: ProxiesMap = {};

	addProxy(name: string, type: string, task: string): void {
		this.proxies[name] = { type, task };
	}

	removeProxy(name: string): void {
		delete this.proxies[name];
	}

	setProxyLimiter(name: string, limiter: RateLimiter): void {
		this.proxies[name].limiter = limiter;
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
	async request<R>(url: string, method: string = "GET", task: string): Promise<R | null> {
		// find a available proxy
		const proxiesNames = Object.keys(this.proxies);
		for (const proxyName of proxiesNames) {
			const proxy = this.proxies[proxyName];
			if (proxy.task !== task) continue;
			if (!proxy.limiter) {
				return await this.proxyRequest<R>(url, proxyName, method);
			}
			const proxyIsNotRateLimited = await proxy.limiter.getAvailability();
			if (proxyIsNotRateLimited) {
				return await this.proxyRequest<R>(url, proxyName, method);
			}
		}
		throw new NetSchedulerError("No available proxy currently.", "NO_AVAILABLE_PROXY");
	}

	/*
	 * Make a request to the specified URL with the specified proxy
	 * @param {string} url - The URL to request.
	 * @param {string} proxyName - The name of the proxy to use.
	 * @param {string} method - The HTTP method to use for the request. Default is "GET".
	 * @param {boolean} force - If true, the request will be made even if the proxy is rate limited. Default is false.
	 * @returns {Promise<any>} - A promise that resolves to the response body.
	 * @throws {NetSchedulerError} - The error will be thrown in following cases:
	 * - Proxy not found: with error code PROXY_NOT_FOUND
	 * - Proxy is under rate limit: with error code PROXY_RATE_LIMITED
	 * - The native `fetch` function threw an error: with error code FETCH_ERROR
	 * - The proxy type is not supported: with error code NOT_IMPLEMENTED
	 */
	async proxyRequest<R>(url: string, proxyName: string, method: string = "GET", force: boolean = false): Promise<R> {
		const proxy = this.proxies[proxyName];
		const limiterExists = proxy.limiter !== undefined;
		if (!proxy) {
			throw new NetSchedulerError(`Proxy "${proxy}" not found`, "PROXY_NOT_FOUND");
		}

		if (!force && limiterExists && !(await proxy.limiter!.getAvailability())) {
			throw new NetSchedulerError(`Proxy "${proxy}" is rate limited`, "PROXY_RATE_LIMITED");
		}

		if (limiterExists) {
			await proxy.limiter!.trigger();
		}

		switch (proxy.type) {
			case "native":
				return await this.nativeRequest<R>(url, method);
			default:
				throw new NetSchedulerError(`Proxy type ${proxy.type} not supported.`, "NOT_IMPLEMENTED");
		}
	}

	async getProxyAvailability(name: string): Promise<boolean> {
		const proxyConfig = this.proxies[name];
		if (!proxyConfig || !proxyConfig.limiter) {
			return true;
		}
		return await proxyConfig.limiter.getAvailability();
	}

	private async nativeRequest<R>(url: string, method: string): Promise<R> {
		try {
			const response = await fetch(url, { method });
			return await response.json() as R;
		} catch (e) {
			logger.error(e as Error);
			throw new NetSchedulerError("Fetch error", "FETCH_ERROR");
		}
	}
}
