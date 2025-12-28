/**
 * Cloudflare Worker Proxy
 *
 * Accepts POST requests with body format:
 * {
 *   url: string,
 *   headers: object
 * }
 *
 * Always sends GET requests to the target URL with the specified headers.
 * Returns JSON response with data and time fields.
 */

import { connect } from "cloudflare:sockets";
import { HttpParser, MessageType } from "@alikia/http-parser";

interface ProxyConfig {
	TIMEOUT_MS: number;
}

interface ProxyRequest {
	url: string;
	headers?: Record<string, string>;
}

interface ParsedHeaders {
	status: number;
	statusText: string;
	headers: Headers;
	headerEnd: number;
}

const CONFIG: ProxyConfig = {
	TIMEOUT_MS: 5000,
};

const encoder = new TextEncoder();
const decoder = new TextDecoder();

function concatUint8Arrays(...arrays: Uint8Array[]): Uint8Array {
	const total = arrays.reduce((sum, arr) => sum + arr.length, 0);
	const result = new Uint8Array(total);
	let offset = 0;
	for (const arr of arrays) {
		result.set(arr, offset);
		offset += arr.length;
	}
	return result;
}

function parseHttpHeaders(buff: Uint8Array): ParsedHeaders | null {
	const text = decoder.decode(buff);
	const headerEnd = text.indexOf("\r\n\r\n");
	if (headerEnd === -1) return null;

	const lines = text.slice(0, headerEnd).split("\r\n");
	const statusMatch = lines[0].match(/HTTP\/1\.[01] (\d+) (.*)/);
	if (!statusMatch) throw new Error("Invalid status line");

	const headers = new Headers();
	for (let i = 1; i < lines.length; i++) {
		const idx = lines[i].indexOf(": ");
		if (idx !== -1) {
			headers.append(lines[i].slice(0, idx), lines[i].slice(idx + 2));
		}
	}

	return {
		headerEnd,
		headers,
		status: Number(statusMatch[1]),
		statusText: statusMatch[2],
	};
}

function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
	return Promise.race([
		promise,
		new Promise<T>((_, reject) => setTimeout(() => reject(new Error("Request timeout")), ms)),
	]);
}

interface ProxyResponseData {
	data: string;
	time: number;
}

async function handleSocket(
	dstUrl: string,
	customHeaders: Record<string, string>,
	requestTime: number
): Promise<ProxyResponseData> {
	const targetUrl = new URL(dstUrl);
	const port = targetUrl.protocol === "https:" ? 443 : 80;

	const socket = connect(
		{
			hostname: targetUrl.hostname,
			port: port,
		},
		{ allowHalfOpen: false, secureTransport: targetUrl.protocol === "https:" ? "on" : "off" }
	);

	const writer = socket.writable.getWriter();
	const headers = new Headers(customHeaders);
	headers.set("Host", targetUrl.hostname);
	headers.set("Connection", "close");
	if (!headers.has("Accept-Encoding")) {
		headers.set("Accept-Encoding", "identity");
	}

	const requestLine =
		`GET ${targetUrl.pathname}${targetUrl.search} HTTP/1.1\r\n` +
		Array.from(headers.entries())
			.map(([k, v]) => `${k}: ${v}`)
			.join("\r\n") +
		"\r\n\r\n";

	await writer.write(encoder.encode(requestLine));

	const reader = socket.readable.getReader();
	const buffer: Uint8Array[] = [];

	while (true) {
		const { value, done } = await reader.read();
		if (done) break;
		buffer.push(value);
	}

	const rawContent = concatUint8Arrays(...buffer);

	const parser = new HttpParser();

	const parsed = parser.parse(rawContent);

	for (const msg of parsed) {
		if (msg.type === MessageType.RESPONSE) {
			return {
				data: new TextDecoder().decode(msg.body),
				time: Math.floor((requestTime + Date.now()) / 2),
			};
		}
	}

	throw new Error("Invalid response");
}

async function handleFetch(
	dstUrl: string,
	customHeaders: Record<string, string>,
	requestTime: number
): Promise<ProxyResponseData> {
	const response = await fetch(dstUrl, {
		headers: customHeaders,
		method: "GET",
	});

	const responseTime = Date.now();
	const combinedTime = Math.floor((requestTime + responseTime) / 2);
	const data = await response.text();

	return {
		data,
		time: combinedTime,
	};
}

function createJsonResponse(data: ProxyResponseData, requestId: string): Response {
	return new Response(JSON.stringify({
		...data,
		requestId,
	}), {
		headers: {
			"Access-Control-Allow-Origin": "*",
			"Content-Type": "application/json",
		},
	});
}

function createErrorResponse(message: string, status: number, requestId: string): Response {
	return new Response(
		JSON.stringify({
			data: "",
			error: message,
			time: Date.now(),
			requestId,
		}),
		{
			headers: {
				"Access-Control-Allow-Origin": "*",
				"Content-Type": "application/json",
			},
			status,
		}
	);
}

export default {
	async fetch(request: Request, _env: Env, _ctx: ExecutionContext): Promise<Response> {
		const requestId = crypto.randomUUID().slice(0, 8); // Track this specific request

		if (request.method !== "POST") {
			console.warn(`[${requestId}] Method Not Allowed: ${request.method}`);
			return createErrorResponse("Method not allowed", 405, requestId);
		}

		let targetUrl: string;
		let customHeaders: Record<string, string>;
		const requestTime = Date.now();

		try {
			const body = (await request.json()) as ProxyRequest;
			targetUrl = body.url;
			new URL(targetUrl);
			customHeaders = body.headers || {};
			console.log(`[${requestId}] Proxying request to: ${targetUrl}`);
		} catch (err) {
			console.error(`[${requestId}] Body Parse Error:`, err);
			return createErrorResponse("Invalid request", 400, requestId);
		}

		try {
			console.log(`[${requestId}] Attempting handleSocket...`);
			const data = await withTimeout(
				handleSocket(targetUrl, customHeaders, requestTime),
				CONFIG.TIMEOUT_MS
			);
			console.log(`[${requestId}] Success via handleSocket (${Date.now() - requestTime}ms)`);
			return createJsonResponse(data, requestId);
		} catch (socketErr: any) {
			console.warn(
				`[${requestId}] handleSocket failed: ${socketErr.message}. Falling back to fetch...`
			);

			try {
				const data = await withTimeout(
					handleFetch(targetUrl, customHeaders, requestTime),
					CONFIG.TIMEOUT_MS
				);
				console.log(
					`[${requestId}] Success via handleFetch (${Date.now() - requestTime}ms)`
				);
				return createJsonResponse(data, requestId);
			} catch (fetchErr: any) {
				console.error(
					`[${requestId}] Critical Failure: Both Socket and Fetch failed. Error: ${fetchErr.message}`
				);
				return createErrorResponse(`Proxy failure: ${fetchErr.message}`, 504, requestId);
			}
		}
	},
};
