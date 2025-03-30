const requiredEnvVars = ["DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD", "DB_PORT"];

const unsetVars = requiredEnvVars.filter((key) => Deno.env.get(key) === undefined);

if (unsetVars.length > 0) {
	throw new Error(`Missing required environment variables: ${unsetVars.join(", ")}`);
}

const databaseHost = Deno.env.get("DB_HOST")!;
const databaseName = Deno.env.get("DB_NAME")!;
const databaseNameCred = Deno.env.get("DB_NAME_CRED")!;
const databaseUser = Deno.env.get("DB_USER")!;
const databasePassword = Deno.env.get("DB_PASSWORD")!;
const databasePort = Deno.env.get("DB_PORT")!;

export const postgresConfig = {
	hostname: databaseHost,
	port: parseInt(databasePort),
	database: databaseName,
	user: databaseUser,
	password: databasePassword,
};

export const postgresConfigCred = {
	hostname: databaseHost,
	port: parseInt(databasePort),
	database: databaseNameCred,
	user: databaseUser,
	password: databasePassword,
};
