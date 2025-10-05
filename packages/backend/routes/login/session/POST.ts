import { Context } from "hono";
import { Bindings, BlankEnv } from "hono/types";
import { ErrorResponse, LoginResponse } from "src/schema";
import { createHandlers } from "src/utils";
import { sqlCred } from "@core/db/dbNew";
import { object, string, ValidationError } from "yup";
import { setCookie } from "hono/cookie";
import Argon2id from "@rabbit-company/argon2id";
import { createLoginSession } from "routes/user/POST";
import { UserType } from "@core/db/schema";

const LoginBodySchema = object({
	username: string().trim().required("Username is required").max(50, "Username cannot exceed 50 characters"),
	password: string().required("Password is required")
});

export const loginHandler = createHandlers(
	async (c: Context<BlankEnv & { Bindings: Bindings }, "/user/session/:id">) => {
		try {
			const body = await LoginBodySchema.validate(await c.req.json());
			const { username, password: submittedPassword } = body;

			const result = await sqlCred<UserType[]>`
			    SELECT *
				FROM users
				WHERE username = ${username}
			`;

			if (result.length === 0) {
				const response: ErrorResponse = {
					message: `User does not exist.`,
					errors: [`User ${username} does not exist.`],
					code: "ENTITY_NOT_FOUND"
				};
				return c.json<ErrorResponse>(response, 400);
			}

			const storedPassword = result[0].password;
			const uid = result[0].id;
			const nickname = result[0].nickname;
			const role = result[0].role;

			const passwordAreSame = await Argon2id.verify(storedPassword, submittedPassword);

			if (!passwordAreSame) {
				const response: ErrorResponse = {
					message: "Incorrect password.",
					errors: [],
					i18n: {
						key: "backend.error.incorrect_password"
					},
					code: "INVALID_CREDENTIALS"
				};
				return c.json<ErrorResponse>(response, 401);
			}

			const sessionID = await createLoginSession(uid, c);

			const response: LoginResponse = {
				uid: uid,
				username: username,
				nickname: nickname,
				role: role,
				token: sessionID
			};

			const A_YEAR = 365 * 86400;
			const isDev = process.env.NODE_ENV === "development";

			setCookie(c, "session_id", sessionID, {
				path: "/",
				maxAge: A_YEAR,
				domain: process.env.DOMAIN,
				secure: isDev ? true : true,
				sameSite: isDev ? "None" : "Lax",
				httpOnly: true
			});

			return c.json<LoginResponse>(response, 200);
		} catch (e) {
			if (e instanceof ValidationError) {
				const response: ErrorResponse = {
					message: "Invalid registration data.",
					errors: e.errors,
					code: "INVALID_PAYLOAD"
				};
				return c.json<ErrorResponse>(response, 400);
			} else if (e instanceof SyntaxError) {
				const response: ErrorResponse = {
					message: "Invalid JSON payload.",
					errors: [e.message],
					code: "INVALID_FORMAT"
				};
				return c.json<ErrorResponse>(response, 400);
			} else {
				const response: ErrorResponse = {
					message: "Unknown error.",
					errors: [(e as Error).message],
					code: "UNKNOWN_ERROR"
				};
				return c.json<ErrorResponse>(response, 500);
			}
		}
	}
);
