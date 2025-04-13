import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";
import { db } from "db/init.ts";

/**
 * Executes a function with a database connection.
 * @param operation The function that accepts the `client` as the parameter.
 * @param errorHandling Optional function to handle errors.
 * If no error handling function is provided, the error will be re-thrown.
 * @param cleanup Optional function to execute after the operation.
 * @returns The result of the operation or undefined if an error occurred.
 */
export async function withDbConnection<T>(
	operation: (client: Client) => Promise<T>,
	errorHandling?: (error: unknown, client: Client) => void,
	cleanup?: () => void,
): Promise<T | undefined> {
	const client = await db.connect();
	try {
		return await operation(client);
	} catch (error) {
		if (errorHandling) {
			errorHandling(error, client);
			return;
		}
		throw error;
	} finally {
		client.release();
		if (cleanup) {
			cleanup();
		}
	}
}
