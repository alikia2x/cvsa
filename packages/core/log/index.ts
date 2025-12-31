import chalk from "chalk";
import type { TransformableInfo } from "logform";
import winston, { format, transports } from "winston";

/* -------------------------------------------------
 * Bun-style console formatter
 * ------------------------------------------------- */
function formatLikeConsole(input: unknown, colors: boolean): string {
	const inspect = (value: unknown) =>
		Bun.inspect(value, {
			colors,
			compact: false,
			depth: 6,
		});

	// console.log(...args)
	if (Array.isArray(input)) {
		return input.map(inspect).join(" ");
	}

	if (input instanceof Error) {
		const stack = input.stack ?? input.message;
		return colors ? chalk.red(stack) : stack;
	}

	if (typeof input === "string") {
		return input;
	}

	return inspect(input);
}

/* -------------------------------------------------
 * Console format (colored, human-readable)
 * ------------------------------------------------- */
const customConsoleFormat = format.printf((info: TransformableInfo) => {
	const { timestamp, level, message, service, codePath } = info;

	const coloredService = service ? chalk.magenta(service) : "";
	const coloredCodePath = codePath ? chalk.grey(`@${codePath}`) : "";
	const colon = service || codePath ? ": " : "";

	const renderedMessage = formatLikeConsole(message, true);

	return `${timestamp} [${level}] ${coloredService}${coloredCodePath}${colon}${renderedMessage}`;
});

/* -------------------------------------------------
 * Timestamp
 * ------------------------------------------------- */
const timestampFormat = format.timestamp({
	format: "YYYY-MM-DD HH:mm:ss.SSSZZ",
});

/* -------------------------------------------------
 * File transport factory (no colors)
 * ------------------------------------------------- */
const createTransport = (level: string, filename: string) => {
	const MB = 1_000_000;
	let maxsize: number | undefined;
	let maxFiles: number | undefined;
	const tailable = false;

	if (level === "silly") {
		maxsize = 500 * MB;
	} else if (level === "warn") {
		maxsize = 10 * MB;
		maxFiles = 5;
	}

	function replacer(_: unknown, value: unknown) {
		if (typeof value === "bigint") {
			return value.toString();
		}
		if (value instanceof Error) {
			return {
				message: value.message,
				name: value.name,
				stack: value.stack,
			};
		}
		return value;
	}

	return new transports.File({
		filename,
		format: format.combine(timestampFormat, format.json({ replacer })),
		level,
		maxFiles,
		maxsize,
		tailable,
	});
};

/* -------------------------------------------------
 * Paths
 * ------------------------------------------------- */
const sillyLogPath = process.env["LOG_VERBOSE"] ?? "logs/verbose.log";
const warnLogPath = process.env["LOG_WARN"] ?? "logs/warn.log";
const errorLogPath = process.env["LOG_ERROR"] ?? "logs/error.log";

/* -------------------------------------------------
 * Winston logger
 * ------------------------------------------------- */
const winstonLogger = winston.createLogger({
	levels: winston.config.npm.levels,
	transports: [
		new transports.Console({
			format: format.combine(timestampFormat, format.colorize(), customConsoleFormat),
			level: "debug",
		}),
		createTransport("silly", sillyLogPath),
		createTransport("warn", warnLogPath),
		createTransport("error", errorLogPath),
	],
});

/* -------------------------------------------------
 * Public logger API
 * ------------------------------------------------- */
const logger = {
	debug: (message: unknown, service?: string, codePath?: string) => {
		winstonLogger.debug(message as string, { codePath, service });
	},

	error: (message: unknown, service?: string, codePath?: string) => {
		winstonLogger.error(message as string, { codePath, service });
	},

	info: (message: unknown, service?: string, codePath?: string) => {
		winstonLogger.info(message as string, { codePath, service });
	},

	log: (message: unknown, service?: string, codePath?: string) => {
		winstonLogger.info(message as string, { codePath, service });
	},

	silly: (message: unknown, service?: string, codePath?: string) => {
		winstonLogger.silly(message as string, { codePath, service });
	},

	verbose: (message: unknown, service?: string, codePath?: string) => {
		winstonLogger.verbose(message as string, { codePath, service });
	},

	warn: (message: unknown, service?: string, codePath?: string) => {
		winstonLogger.warn(message as string, { codePath, service });
	},
};

export default logger;
