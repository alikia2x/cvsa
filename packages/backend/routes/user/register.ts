import { createHandlers } from "src/utils.ts";
import Argon2id from "@rabbit-company/argon2id";
import { object, string, ValidationError } from "yup";
import type { Context } from "hono";
import type { Bindings, BlankEnv, BlankInput } from "hono/types";
import { sqlCred } from "@core/db/dbNew.ts";
import { ErrorResponse, StatusResponse } from "src/schema";
import { generateRandomId } from "@core/lib/randomID";
import { getUserIP } from "@/middleware/rateLimiters";

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

const createLoginSession = async (uid: number, ua?: string, ip?: string) => {
	const ip_address = ip || null;
	const user_agent = ua || null;
	const id = generateRandomId(16);
	await sqlCred`
        INSERT INTO login_sessions (id, uid, expire_at, ip_address, user_agent)
        VALUES (${id}, ${uid}, CURRENT_TIMESTAMP + INTERVAL '1 year', ${ip_address}, ${user_agent})
    `;
};

const getUserIDByName = async (username: string) => {
	const result = await sqlCred<{ id: number }[]>`
        SELECT id
        FROM users
        WHERE username = ${username}
    `;
	if (result.length === 0) {
		return null;
	}
	return result[0].id;
};

export const registerHandler = createHandlers(async (c: ContextType) => {
	try {
		const body = await RegistrationBodySchema.validate(await c.req.json());
		const { username, password, nickname } = body;

		if (await userExists(username)) {
			const response: ErrorResponse = {
				message: `User "${username}" already exists.`,
				code: "ENTITY_EXISTS",
				errors: [],
				i18n: {
					key: "backend.error.user_exists",
					values: {
						username: username
					}
				}
			};
			return c.json<ErrorResponse>(response, 400);
		}

		const hash = await Argon2id.hashEncoded(password);

		await sqlCred`
            INSERT INTO users (username, password, nickname)
            VALUES (${username}, ${hash}, ${nickname ? nickname : null})
		`;

		const uid = await getUserIDByName(username);

		if (!uid) {
			const response: ErrorResponse<string> = {
				message: "Cannot find registered user.",
				errors: [`Cannot find user ${username} in table 'users'.`],
				code: "ENTITY_NOT_FOUND",
				i18n: {
					key: "backend.error.user_not_found_after_register",
					values: {
						username: username
					}
				}
			};
			return c.json<ErrorResponse<string>>(response, 500);
		}

		createLoginSession(uid, c.req.header("User-Agent"), getUserIP(c));

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
				message: "Unknown error.",
				errors: [(e as Error).message],
				code: "UNKNOWN_ERROR"
			};
			return c.json<ErrorResponse<string>>(response, 500);
		}
	}
});
