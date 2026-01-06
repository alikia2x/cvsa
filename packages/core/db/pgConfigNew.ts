const requiredEnvVars = ["DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD", "DB_PORT"];

const getEnvVar = (key: string): string => {
	return process.env[key] || import.meta.env[key];
};

const unsetVars = requiredEnvVars.filter((key) => getEnvVar(key) === undefined);

if (unsetVars.length > 0) {
	throw new Error(`Missing required environment variables: ${unsetVars.join(", ")}`);
}

const databaseHost = getEnvVar("DB_HOST");
const databaseName = getEnvVar("DB_NAME");
const databaseUser = getEnvVar("DB_USER");
const databasePassword = getEnvVar("DB_PASSWORD");
const databasePort = getEnvVar("DB_PORT");

export const postgresConfig = {
	database: databaseName,
	host: databaseHost,
	password: databasePassword,
	port: parseInt(databasePort, 10),
	username: databaseUser,
};
