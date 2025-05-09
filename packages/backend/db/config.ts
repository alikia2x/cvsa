const requiredEnvVars = ["DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD", "DB_PORT", "DB_NAME_CRED"];

const unsetVars = requiredEnvVars.filter((key) => process.env[key] === undefined);

if (unsetVars.length > 0) {
	throw new Error(`Missing required environment variables: ${unsetVars.join(", ")}`);
}

const databaseHost = process.env["DB_HOST"]!;
const databaseName = process.env["DB_NAME"];
const databaseNameCred = process.env["DB_NAME_CRED"]!;
const databaseUser = process.env["DB_USER"]!;
const databasePassword = process.env["DB_PASSWORD"]!;
const databasePort = process.env["DB_PORT"]!;

export const postgresConfig = {
	hostname: databaseHost,
	port: parseInt(databasePort),
	database: databaseName,
	user: databaseUser,
	password: databasePassword
};

export const postgresConfigNpm = {
	host: databaseHost,
	port: parseInt(databasePort),
	database: databaseName,
	username: databaseUser,
	password: databasePassword
};

export const postgresCredConfigNpm = {
	host: databaseHost,
	port: parseInt(databasePort),
	database: databaseNameCred,
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
