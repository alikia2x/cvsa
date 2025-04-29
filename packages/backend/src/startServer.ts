import { serve } from "bun";
import { Hono } from "hono";
import os from "os";
import { BlankSchema, Variables } from "hono/types";

function getLocalIpAddress(): string {
	const interfaces = os.networkInterfaces();
	for (const name of Object.keys(interfaces)) {
		for (const iface of interfaces[name]!) {
			if (iface.family === "IPv4" && !iface.internal) {
				return iface.address;
			}
		}
	}
	return "localhost";
}

function logStartup(hostname: string, port: number, wasAutoIncremented: boolean, originalPort: number) {
	const localUrl = `http://localhost:${port}`;
	const networkIp = hostname === "0.0.0.0" ? getLocalIpAddress() : "";
	const networkUrl = networkIp ? `http://${networkIp}:${port}` : "";

	console.log("\n");
	console.log("🚀 Server is running at:");
	console.log(`> Local:   ${localUrl}`);
	if (networkIp) {
		console.log(`> Network: ${networkUrl}`);
	}
	if (wasAutoIncremented) {
		console.log(`\n⚠️ Port ${originalPort} is in use, using port ${port} instead.`);
	}
	console.log("\nPress Ctrl+C to quit.");
}

export async function startServer(app: Hono<{Variables: Variables }>) {
	const NODE_ENV = process.env.NODE_ENV || "production";
	const HOST = process.env.HOST ?? (NODE_ENV === "development" ? "0.0.0.0" : "127.0.0.1");
	const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : undefined;

	const DEFAULT_PORT = 3000;
	const MAX_ATTEMPTS = 15;

	if (PORT !== undefined) {
		try {
			const server = serve({
				fetch: app.fetch,
				hostname: HOST,
				port: PORT
			});
			logStartup(HOST, PORT, false, DEFAULT_PORT);
			return server;
		} catch (e: any) {
			console.error(`Failed to start server on port ${PORT}:`, e.message);
			process.exit(1);
		}
	}

	let attemptPort = DEFAULT_PORT;
	let success = false;
	let error: unknown = null;

	for (let i = 0; i <= MAX_ATTEMPTS; i++) {
		try {
			const server = serve({
				fetch: app.fetch,
				hostname: HOST,
				port: attemptPort
			});

			const wasAutoIncremented = attemptPort !== DEFAULT_PORT;

			logStartup(HOST, attemptPort, wasAutoIncremented, DEFAULT_PORT);
			return server;
		} catch (e: any) {
			if (e.code === "EADDRINUSE") {
				attemptPort++;
			} else {
				error = e;
				break;
			}
		}
	}

	if (!success) {
		console.error(`Could not find an available port after ${MAX_ATTEMPTS + 1} attempts.`);
		if (error) {
			console.error("Last error:", (error as Error).message);
		}
		process.exit(1);
	}
}
