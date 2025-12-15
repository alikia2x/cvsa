import { generate as generateId } from "@alikia/random-key";
import {
	db,
	loginSessionsInCredentials,
	type SessionType,
	type UserType,
	usersInCredentials,
} from "@core/drizzle";
import logger from "@core/log";
import Argon2id from "@rabbit-company/argon2id";
import { and, eq, isNull } from "drizzle-orm";

export async function verifyUser(
	username: string,
	password: string
): Promise<Omit<UserType, "password"> | null> {
	const user = await db
		.select()
		.from(usersInCredentials)
		.where(eq(usersInCredentials.username, username))
		.limit(1);

	if (user.length === 0) {
		return null;
	}

	const foundUser = user[0];
	const isPasswordValid = await Argon2id.verify(foundUser.password, password);
	if (!isPasswordValid) {
		return null;
	}

	return {
		id: foundUser.id,
		username: foundUser.username,
		nickname: foundUser.nickname,
		role: foundUser.role,
		unqId: foundUser.unqId,
		createdAt: foundUser.createdAt,
	};
}

export async function createSession(
	userId: number,
	ipAddress: string | null,
	userAgent: string,
	expiresInDays: number = 30
): Promise<string> {
	const sessionId = await generateId(24);
	const expireAt = new Date();
	expireAt.setDate(expireAt.getDate() + expiresInDays);

	try {
		await db.insert(loginSessionsInCredentials).values({
			id: sessionId,
			uid: userId,
			ipAddress,
			userAgent,
			lastUsedAt: new Date().toISOString(),
			expireAt: expireAt.toISOString(),
		});
	} catch (error) {
		logger.error(error as Error);
		throw error;
	}

	return sessionId;
}

export async function validateSession(
	sessionId: string
): Promise<{ user: UserType; session: SessionType } | null> {
	const sessions = await db
		.select()
		.from(loginSessionsInCredentials)
		.where(
			and(
				eq(loginSessionsInCredentials.id, sessionId),
				isNull(loginSessionsInCredentials.deactivatedAt)
			)
		)
		.limit(1);

	if (sessions.length === 0) {
		return null;
	}

	const session = sessions[0];

	if (session.expireAt && new Date(session.expireAt) < new Date()) {
		return null;
	}

	const users = await db
		.select()
		.from(usersInCredentials)
		.where(eq(usersInCredentials.id, session.uid))
		.limit(1);

	if (users.length === 0) {
		return null;
	}

	await db
		.update(loginSessionsInCredentials)
		.set({ lastUsedAt: new Date().toISOString() })
		.where(eq(loginSessionsInCredentials.id, sessionId));

	return {
		user: users[0],
		session: session,
	};
}

export async function deactivateSession(sessionId: string): Promise<boolean> {
	const result = await db
		.update(loginSessionsInCredentials)
		.set({
			deactivatedAt: new Date().toISOString(),
		})
		.where(eq(loginSessionsInCredentials.id, sessionId));

	return result.length ? result.length > 0 : false;
}

export function getSessionExpirationDate(days: number = 30): Date {
	const expireAt = new Date();
	expireAt.setDate(expireAt.getDate() + days);
	return expireAt;
}
