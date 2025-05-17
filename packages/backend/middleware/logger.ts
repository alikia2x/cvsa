// Color constants
import { Context, Next } from "hono";
import { TimingVariables } from "hono/timing";
import { getConnInfo } from "hono/bun";

const green = "\x1b[97;42m";
const white = "\x1b[90;47m";
const yellow = "\x1b[90;43m";
const red = "\x1b[97;41m";
const blue = "\x1b[97;44m";
const magenta = "\x1b[97;45m";
const cyan = "\x1b[97;46m";
const reset = "\x1b[0m";

let consoleColorMode = "auto";

function formatCurrentTime() {
	const now = new Date();
	const year = now.getFullYear();
	const month = String(now.getMonth() + 1).padStart(2, "0"); // Month is 0-indexed
	const day = String(now.getDate()).padStart(2, "0");
	const hours = String(now.getHours()).padStart(2, "0");
	const minutes = String(now.getMinutes()).padStart(2, "0");
	const seconds = String(now.getSeconds()).padStart(2, "0");
	const milliseconds = String(now.getMilliseconds()).padStart(3, "0");

	return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}.${milliseconds}`;
}

export const DisableConsoleColor = () => {
	consoleColorMode = "disable";
};

export const ForceConsoleColor = () => {
	consoleColorMode = "force";
};

const defaultFormatter = (params) => {
	const latency = params.latency > 60000 ? `${Math.round(params.latency / 1000)}s` : `${params.latency}ms`;

	let statusColor = white;
	if (params.isOutputColor) {
		if (params.status >= 100 && params.status < 300) statusColor = green;
		else if (params.status >= 300 && params.status < 400) statusColor = white;
		else if (params.status >= 400 && params.status < 500) statusColor = yellow;
		else statusColor = red;
	}

	let methodColor = reset;
	switch (params.method) {
		case "GET":
			methodColor = blue;
			break;
		case "POST":
			methodColor = cyan;
			break;
		case "PUT":
			methodColor = yellow;
			break;
		case "DELETE":
			methodColor = red;
			break;
		case "PATCH":
			methodColor = green;
			break;
		case "HEAD":
			methodColor = magenta;
			break;
		case "OPTIONS":
			methodColor = white;
			break;
	}

	return (
		`${params.timestamp} |${statusColor} ${params.status} ${reset}| ` +
		`${latency.padStart(7)} | ${params.ip.padStart(16)} |` +
		`${methodColor} ${params.method.padEnd(6)}${reset} ${params.path}`
	);
};
type Ctx = Context;
export const logger = (config) => {
	const { formatter = defaultFormatter, output = console, skipPaths = [], skip = null } = config;

	// Convert skipPaths to Set for faster lookups
	const skipPathsSet = new Set(skipPaths);

	return async (c: Ctx, next: Next) => {
		const start = Date.now();
		const url = new URL(c.req.url);
		const path = url.pathname;

		// Check if we should skip logging
		if (skipPathsSet.has(path) || (typeof skip === "function" && skip(c))) {
			return next();
		}

		try {
			await next();
		} catch (error) {
			// Handle errors
			const errorParams = {
				timestamp: formatCurrentTime(),
				latency: Date.now() - start,
				status: 500,
				ip: getClientIP(c),
				method: c.req.method,
				path,
				error: error.message,
				isOutputColor: shouldColorize(c)
			};

			output.error(
				formatter({
					...errorParams,
					errorMessage: error.message
				})
			);

			throw error;
		}

		const status = c.res.status;
		const latency = Date.now() - start;

		const params = {
			timestamp: formatCurrentTime(),
			latency,
			status,
			ip: getClientIP(c),
			method: c.req.method,
			path,
			bodySize: c.res.headers.get("content-length") || 0,
			isOutputColor: shouldColorize(c)
		};

		// Format and output the log
		const logMessage = formatter(params);

		if (status >= 400 && status < 500) {
			output.warn?.(logMessage) || output.log(logMessage);
		} else if (status >= 500) {
			output.error?.(logMessage) || output.log(logMessage);
		} else {
			output.log(logMessage);
		}
	};
};

function shouldColorize(c) {
	if (consoleColorMode === "disable") return false;
	if (consoleColorMode === "force") return true;

	// In development environment with TTY
	return process.stdout.isTTY;
}

export function getClientIP(c: Ctx) {
	const info = getConnInfo(c);
	return info.remote.address;
}
