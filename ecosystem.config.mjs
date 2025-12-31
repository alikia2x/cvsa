import "dotenv/config";

export const apps = [
	{
		cwd: "./packages/crawler",
		interpreter: "bun",
		name: "crawler-jobadder",
		script: "src/jobAdder.wrapper.ts",
	},
	{
		cwd: "./packages/crawler",
		env: {
			LOG_ERR: "logs/error.log",
			LOG_VERBOSE: "logs/verbose.log",
			LOG_WARN: "logs/warn.log",
		},
		interpreter: "bun",
		name: "crawler-worker",
		script: "src/worker.ts",
	},
	{
		cwd: "./packages/crawler",
		env: {
			LOG_ERR: "logs/error.log",
			LOG_VERBOSE: "logs/verbose.log",
			LOG_WARN: "logs/warn.log",
		},
		interpreter: "bun",
		name: "crawler-filter",
		script: "src/filterWorker.wrapper.ts",
	},
	{
		cwd: "./ml/api",
		env: {
			LOG_ERR: "logs/error.log",
			LOG_VERBOSE: "logs/verbose.log",
			LOG_WARN: "logs/warn.log",
			PYTHONPATH: "./ml/api:./ml/filter",
		},
		interpreter: process.env.PYTHON_INTERPRETER || "python3",
		name: "ml-api",
		script: "start.py",
	},
	{
		cwd: "./packages/backend",
		env: {
			NODE_ENV: "production",
		},
		interpreter: "bun",
		name: "cvsa-be",
		script: "src/index.ts",
	},
];
