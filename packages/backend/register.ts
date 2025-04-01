import { createHandlers } from "./utils.ts";
import Argon2id from "@rabbit-company/argon2id";
import { object, string, ValidationError } from "yup";
import type { Context } from "hono";
import type { Bindings, BlankEnv, BlankInput } from "hono/types";
import type { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";

const RegistrationBodySchema = object({
	username: string().trim().required("Username is required").max(50, "Username cannot exceed 50 characters"),
	password: string().required("Password is required"),
    nickname: string().optional(),
});

type ContextType = Context<BlankEnv & { Bindings: Bindings }, "/user", BlankInput>;

export const userExists = async (username: string, client: Client) => {
	const query = `
        SELECT * FROM users WHERE username = $1
    `;
	const result = await client.queryObject(query, [username]);
	return result.rows.length > 0;
}

export const registerHandler = createHandlers(async (c: ContextType) => {
	const client = c.get("dbCred");

	try {
		const body = await RegistrationBodySchema.validate(await c.req.json());
		const { username, password, nickname } = body;

        if (await userExists(username, client)) {
			return c.json({
				message: `User "${username}" already exists.`,
			}, 400);
        }

		const hash = await Argon2id.hashEncoded(password);

		const query = `
            INSERT INTO users (username, password, nickname) VALUES ($1, $2, $3)
        `;
		await client.queryObject(query, [username, hash, nickname || null]);

		return c.json({
			message: `User "${username}" registered successfully.`,
		}, 201);
	} catch (e) {
		if (e instanceof ValidationError) {
			return c.json({
				message: "Invalid registration data.",
				errors: e.errors,
			}, 400); 
		} else if (e instanceof SyntaxError) {
			return c.json({
				message: "Invalid JSON in request body.",
			}, 400);
		} else {
			console.error("Registration error:", e);
			return c.json({
				message: "An unexpected error occurred during registration.",
				error: (e as Error).message,
			}, 500);
		}
	}
});
