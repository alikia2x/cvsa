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
		status: Number(statusMatch[1]),
		statusText: statusMatch[2],
		headers,
		headerEnd,
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
		{ secureTransport: targetUrl.protocol === "https:" ? "on" : "off", allowHalfOpen: false }
	);

	const writer = socket.writable.getWriter();
	const headers = new Headers(customHeaders);
	headers.set("Host", targetUrl.hostname);
	headers.set("Connection", "close");
	if (!headers.has("Accept-Encoding")) {
		headers.set("Accept-Encoding", "gzip, deflate, br");
	}

	const requestLine =
		`GET ${targetUrl.pathname}${targetUrl.search} HTTP/1.1\r\n` +
		Array.from(headers.entries())
			.map(([k, v]) => `${k}: ${v}`)
			.join("\r\n") +
		"\r\n\r\n";

	await writer.write(encoder.encode(requestLine));

	const reader = socket.readable.getReader();
	let buffer: Uint8Array<ArrayBufferLike> = new Uint8Array();

	while (true) {
		const { value, done } = await reader.read();
		if (done) break;
		buffer = concatUint8Arrays(buffer, value);

		const parsed = parseHttpHeaders(buffer);
		if (!parsed) {
			continue;
		}
		const { headers: respHeaders, headerEnd } = parsed;
		const initialData = buffer.slice(headerEnd + 4);
		const encoding = respHeaders.get("content-encoding");

		const chunks: Uint8Array[] = [];
		if (initialData.length > 0) chunks.push(initialData);

		const rawStream = new ReadableStream({
			start: async (ctrl) => {
				if (initialData.length > 0) ctrl.enqueue(initialData);
				try {
					while (true) {
						const { value, done } = await reader.read();
						if (done) break;
						ctrl.enqueue(value);
						chunks.push(value);
					}
					ctrl.close();
				} catch (e) {
					ctrl.error(e);
				}
			},
		});

		let decompressedChunks: Uint8Array[] = chunks;
		if (encoding === "gzip" || encoding === "deflate") {
			const decompressionStream = rawStream.pipeThrough(new DecompressionStream(encoding));
			decompressedChunks = [];
			const decompressReader = decompressionStream.getReader();
			while (true) {
				const { value, done } = await decompressReader.read();
				if (done) break;
				decompressedChunks.push(value);
			}
			respHeaders.delete("content-encoding");
			respHeaders.delete("content-length");
		}

		const responseTime = Date.now();
		const combinedTime = Math.floor((requestTime + responseTime) / 2);
		const data = new Uint8Array(decompressedChunks.reduce((sum, arr) => sum + arr.length, 0));
		let offset = 0;
		for (const chunk of decompressedChunks) {
			data.set(chunk, offset);
			offset += chunk.length;
		}

		return {
			data: new TextDecoder().decode(data),
			time: combinedTime,
		};
	}

	throw new Error("Unable to parse response");
}

async function handleFetch(
	dstUrl: string,
	customHeaders: Record<string, string>,
	requestTime: number
): Promise<ProxyResponseData> {
	const response = await fetch(dstUrl, {
		method: "GET",
		headers: customHeaders,
	});

	const responseTime = Date.now();
	const combinedTime = Math.floor((requestTime + responseTime) / 2);
	const data = await response.text();

	return {
		data,
		time: combinedTime,
	};
}

function createJsonResponse(data: ProxyResponseData): Response {
	return new Response(JSON.stringify(data), {
		headers: {
			"Content-Type": "application/json",
			"Access-Control-Allow-Origin": "*",
		},
	});
}

function createErrorResponse(message: string, status: number): Response {
	return new Response(
		JSON.stringify({
			data: "",
			time: Date.now(),
			error: message,
		}),
		{
			status,
			headers: {
				"Content-Type": "application/json",
				"Access-Control-Allow-Origin": "*",
			},
		}
	);
}

export default {
	async fetch(request: Request, _env: Env, _ctx: ExecutionContext): Promise<Response> {
		if (request.method !== "POST") {
			return createErrorResponse("Method not allowed", 405);
		}

		let targetUrl: string;
		let customHeaders: Record<string, string>;
		const requestTime = Date.now();

		try {
			const body = (await request.json()) as ProxyRequest;
			targetUrl = body.url;
			new URL(targetUrl);
			customHeaders = body.headers || {};
		} catch {
			return createErrorResponse("Invalid request", 400);
		}

		try {
			const data = await withTimeout(
				handleSocket(targetUrl, customHeaders, requestTime),
				CONFIG.TIMEOUT_MS
			);
			return createJsonResponse(data);
		} catch {
			try {
				const data = await withTimeout(
					handleFetch(targetUrl, customHeaders, requestTime),
					CONFIG.TIMEOUT_MS
				);
				return createJsonResponse(data);
			} catch {
				return createErrorResponse("Socket timeout", 504);
			}
		}
	},
};
