import winston, { format, transports } from "winston";
import type { TransformableInfo } from "logform";
import chalk from "chalk";

const customFormat = format.printf((info: TransformableInfo) => {
	const { timestamp, level, message, service, codePath, error } = info;
	const coloredService = service ? chalk.magenta(service) : "";
	const coloredCodePath = codePath ? chalk.grey(`@${codePath}`) : "";
	const colon = service || codePath ? ": " : "";
	const err = error as Error | undefined;
	if (err) {
		return `${timestamp} [${level}] ${coloredService}${colon}${message}\n${chalk.red(err.stack) ?? ""}`;
	}

	return coloredCodePath
		? `${timestamp} [${level}] ${coloredService}${coloredCodePath}${colon}${message}`
		: `${timestamp} [${level}] ${coloredService}${colon}${message}`;
});

const timestampFormat = format.timestamp({ format: "YYYY-MM-DD HH:mm:ss.SSS" });

const createTransport = (level: string, filename: string) => {
	const MB = 1000000;
	let maxsize = undefined;
	let maxFiles = undefined;
	let tailable = undefined;
	if (level === "silly") {
		maxsize = 500 * MB;
		maxFiles = undefined;
		tailable = false;
	} else if (level === "warn") {
		maxsize = 10 * MB;
		maxFiles = 5;
		tailable = false;
	}
	function replacer(key: unknown, value: unknown) {
		if (typeof value === "bigint") {
			return value.toString();
		}
		if (key == "error") {
			return undefined;
		}
		return value;
	}
	return new transports.File({
		level,
		filename,
		maxsize,
		tailable,
		maxFiles,
		format: format.combine(timestampFormat, format.json({ replacer }))
	});
};

const sillyLogPath = process.env["LOG_VERBOSE"] ?? "logs/verbose.log";
const warnLogPath = process.env["LOG_WARN"] ?? "logs/warn.log";
const errorLogPath = process.env["LOG_ERROR"] ?? "logs/error.log";

const winstonLogger = winston.createLogger({
	levels: winston.config.npm.levels,
	transports: [
		new transports.Console({
			level: "debug",
			format: format.combine(
				format.timestamp({ format: "YYYY-MM-DD HH:mm:ss.SSS" }),
				format.colorize(),
				format.errors({ stack: true }),
				customFormat
			)
		}),
		createTransport("silly", sillyLogPath),
		createTransport("warn", warnLogPath),
		createTransport("error", errorLogPath)
	]
});

const logger = {
	silly: (message: string, service?: string, codePath?: string) => {
		winstonLogger.silly(message, { service, codePath });
	},
	verbose: (message: string, service?: string, codePath?: string) => {
		winstonLogger.verbose(message, { service, codePath });
	},
	log: (message: string, service?: string, codePath?: string) => {
		winstonLogger.info(message, { service, codePath });
	},
	debug: (message: string, service?: string, codePath?: string) => {
		winstonLogger.debug(message, { service, codePath });
	},
	warn: (message: string, service?: string, codePath?: string) => {
		winstonLogger.warn(message, { service, codePath });
	},
	error: (error: string | Error, service?: string, codePath?: string) => {
		if (error instanceof Error) {
			winstonLogger.error(error.message, { service, error: error, codePath });
		} else {
			winstonLogger.error(error, { service, codePath });
		}
	}
};

export default logger;
