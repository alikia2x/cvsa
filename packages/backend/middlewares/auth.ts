import { validateSession } from "@backend/lib/auth";
import type { SessionType, UserType } from "@core/drizzle";
import { Elysia } from "elysia";

export interface AuthenticatedContext {
	user: UserType;
	session: SessionType;
	isAuthenticated: boolean;
}

/**
 * Authentication middleware that validates session from cookie or authorization header
 *
 * This middleware:
 * 1. Checks for sessionId in cookie (primary method)
 * 2. Falls back to Authorization header with Bearer token
 * 3. Validates the session using validateSession from lib/auth
 * 4. Sets user and session context for authenticated routes
 * 5. Returns 401 if authentication fails
 */
export const requireAuth = new Elysia({ name: "require-auth" })
	.derive(async ({ cookie, headers, set }) => {
		let sessionId: string | null = null;

		// Try to get sessionId from cookie first
		if (cookie.sessionId && typeof cookie.sessionId.value === "string") {
			sessionId = cookie.sessionId.value;
		}
		// Fallback to Authorization header
		else if (headers.authorization) {
			const authHeader = headers.authorization;
			if (authHeader.startsWith("Bearer ")) {
				sessionId = authHeader.substring(7);
			} else if (authHeader.startsWith("Token ")) {
				sessionId = authHeader.substring(6);
			}
		}

		// If no sessionId found, return unauthenticated
		if (!sessionId) {
			set.status = 401;
			return {
				user: null,
				session: null,
				isAuthenticated: false,
			};
		}

		// Validate the session
		const validationResult = await validateSession(sessionId);

		if (!validationResult) {
			set.status = 401;
			return {
				user: null,
				session: null,
				isAuthenticated: false,
			};
		}

		// Session is valid, return user and session context
		return {
			user: validationResult.user,
			session: validationResult.session,
			isAuthenticated: true,
		};
	})
	.onBeforeHandle({ as: "scoped" }, ({ user, status }) => {
		if (!user) {
			return status(401, {
				message: "Authentication required.",
			});
		}
	})
	.as("scoped");

export default requireAuth;
