import os from "os";
import chalk from "chalk";

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

export function logStartup(hostname: string, port: number) {
	const localUrl = `http://localhost:${port}`;
	const networkIp = hostname === "0.0.0.0" ? getLocalIpAddress() : "";
	const networkUrl = networkIp ? `http://${networkIp}:${port}` : "";

	console.log("\n");
	console.log(chalk.magenta("Chia has been activated."));
	console.log("Available on:");
	console.log(`${chalk.green("> Local:")}  ${chalk.blue(localUrl)}`);
	if (networkIp) {
		console.log(`${chalk.red("> Network:")}  ${chalk.blue(networkUrl)}`);
	}
	console.log("\nPress Ctrl+C to quit.");
}

export function getBindingInfo(): [string, number] {
	const DEFAULT_PORT = 15412;
	const NODE_ENV = process.env.NODE_ENV || "production";
	const HOST = process.env.HOST ?? (NODE_ENV === "development" ? "0.0.0.0" : "127.0.0.1");
	const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : DEFAULT_PORT;
	return [HOST, PORT];
}
