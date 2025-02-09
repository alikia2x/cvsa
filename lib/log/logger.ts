import winston, { format, transports } from "npm:winston";
import { TransformableInfo } from "npm:logform";
import chalk from "npm:chalk";
import stripAnsi from 'npm:strip-ansi';

const customFormat = format.printf((info: TransformableInfo) => {
	const { timestamp, level, message, service, codePath } = info;
	const coloredService = service ? chalk.magenta(service): "";
	const coloredCodePath = codePath ? chalk.grey(`@${codePath}`) : "";
    const colon = service || codePath ? ": " : "";

	return stripAnsi(level) === "debug"
		? `${timestamp} [${level}] ${coloredService}${coloredCodePath}${colon}${message}`
		: `${timestamp} [${level}] ${coloredService}${colon}${message}`;
});

const timestampFormat = format.timestamp({ format: "YYYY-MM-DD HH:mm:ss.SSS" });

const createTransport = (level: string, filename: string) => {
	return new transports.File({
		level,
		filename,
		format: format.combine(timestampFormat, format.json()),
	});
};

const winstonLogger = winston.createLogger({
	levels: winston.config.npm.levels,
	transports: [
		new transports.Console({
			level: "debug",
			format: format.combine(
				format.timestamp({ format: "HH:mm:ss.SSS" }), // Different format for console
				format.colorize(),
				customFormat,
			),
		}),
		createTransport("info", "logs/app.log"),
		createTransport("warn", "logs/warn.log"),
		createTransport("error", "logs/error.log"),
	],
});

const logger = {
	log: (message: string, service?: string, target: "term" | "file" | "both" = "both") => {
		const logLevels = [];
		if (target === "term" || target === "both") {
			logLevels.push("info");
		}
		if (target === "file" || target === "both") {
			logLevels.push("info");
		}
		logLevels.forEach((level) => winstonLogger.log(level, message, { service }));
	},
	debug: (message: string, service?: string, codePath?: string) => {
		winstonLogger.debug(message, { service, codePath });
	},
	warn: (message: string, service?: string) => {
		winstonLogger.warn(message, { service });
	},
	error: (message: string, service?: string) => {
		winstonLogger.error(message, { service });
	},
};

export default logger;
