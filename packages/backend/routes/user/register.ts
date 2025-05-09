import { createHandlers } from "src/utils.ts";
import Argon2id from "@rabbit-company/argon2id";
import { object, string, ValidationError } from "yup";
import type { Context } from "hono";
import type { Bindings, BlankEnv, BlankInput } from "hono/types";
import { sqlCred } from "db/db.ts";
import { ErrorResponse, StatusResponse } from "src/schema";

const RegistrationBodySchema = object({
	username: string().trim().required("Username is required").max(50, "Username cannot exceed 50 characters"),
	password: string().required("Password is required"),
	nickname: string().optional()
});

type ContextType = Context<BlankEnv & { Bindings: Bindings }, "/user", BlankInput>;

export const userExists = async (username: string) => {
	const result = await sqlCred`
        SELECT 1
        FROM users
        WHERE username = ${username}
	`;
	return result.length > 0;
};

export const registerHandler = createHandlers(async (c: ContextType) => {
	try {
		const body = await RegistrationBodySchema.validate(await c.req.json());
		const { username, password, nickname } = body;

		if (await userExists(username)) {
			const response: StatusResponse = {
				message: `User "${username}" already exists.`
			};
			return c.json<StatusResponse>(response, 400);
		}

		const hash = await Argon2id.hashEncoded(password);

		await sqlCred`
            INSERT INTO users (username, password, nickname)
            VALUES (${username}, ${hash}, ${nickname ? nickname : null})
		`;

		const response: StatusResponse = {
			message: `User '${username}' registered successfully.`
		};

		return c.json<StatusResponse>(response, 201);
	} catch (e) {
		if (e instanceof ValidationError) {
			const response: ErrorResponse<string> = {
				message: "Invalid registration data.",
				errors: e.errors,
				code: "INVALID_PAYLOAD"
			};
			return c.json<ErrorResponse<string>>(response, 400);
		} else if (e instanceof SyntaxError) {
			const response: ErrorResponse<string> = {
				message: "Invalid JSON payload.",
				errors: [e.message],
				code: "INVALID_FORMAT"
			};
			return c.json<ErrorResponse<string>>(response, 400);
		} else {
			const response: ErrorResponse<string> = {
				message: "Invalid JSON payload.",
				errors: [(e as Error).message],
				code: "UNKNOWN_ERR"
			};
			return c.json<ErrorResponse<string>>(response, 500);
		}
	}
});
