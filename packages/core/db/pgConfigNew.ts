const requiredEnvVars = ["DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD", "DB_PORT", "DB_NAME_CRED"];

const getEnvVar = (key: string) => {
    return process.env[key] || import.meta.env[key];
}

const unsetVars = requiredEnvVars.filter((key) => getEnvVar(key) === undefined);

if (unsetVars.length > 0) {
	throw new Error(`Missing required environment variables: ${unsetVars.join(", ")}`);
}

const databaseHost = getEnvVar("DB_HOST")!;
const databaseName = getEnvVar("DB_NAME");
const databaseNameCred = getEnvVar("DB_NAME_CRED")!;
const databaseUser = getEnvVar("DB_USER")!;
const databasePassword = getEnvVar("DB_PASSWORD")!;
const databasePort = getEnvVar("DB_PORT")!;

export const postgresConfig = {
	host: databaseHost,
	port: parseInt(databasePort),
	database: databaseName,
	username: databaseUser,
	password: databasePassword
};

export const postgresConfigCred = {
	hostname: databaseHost,
	port: parseInt(databasePort),
	database: databaseNameCred,
	user: databaseUser,
	password: databasePassword
};
