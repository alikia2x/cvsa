import Argon2id from "@rabbit-company/argon2id";
import { db, usersInCredentials, loginSessionsInCredentials } from "@core/drizzle";
import { eq, and, isNull, getTableColumns } from "drizzle-orm";
import { generate as generateId } from "@alikia/random-key";
import logger from "@core/log";

export interface User {
	id: number;
	username: string;
	nickname: string | null;
	role: string;
	unqId: string;
}

export async function verifyUser(username: string, password: string): Promise<User | null> {
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
		unqId: foundUser.unqId
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
			expireAt: expireAt.toISOString()
		});
	} catch (error) {
		logger.error(error as Error);
		throw error;
	}

	return sessionId;
}

export async function validateSession(
	sessionId: string
): Promise<{ user: User; session: any } | null> {
	const session = await db
		.select({
			...getTableColumns(usersInCredentials),
			...getTableColumns(loginSessionsInCredentials)
		})
		.from(loginSessionsInCredentials)
		.innerJoin(usersInCredentials, eq(loginSessionsInCredentials.uid, usersInCredentials.id))
		.where(
			and(
				eq(loginSessionsInCredentials.id, sessionId),
				isNull(loginSessionsInCredentials.deactivatedAt)
			)
		)
		.limit(1);

	if (session.length === 0) {
		return null;
	}

	const foundSession = session[0];

	if (foundSession.expireAt && new Date(foundSession.expireAt) < new Date()) {
		return null;
	}

	const user = await db
		.select()
		.from(usersInCredentials)
		.where(eq(usersInCredentials.id, foundSession.uid))
		.limit(1);

	if (user.length === 0) {
		return null;
	}

	await db
		.update(loginSessionsInCredentials)
		.set({ lastUsedAt: new Date().toISOString() })
		.where(eq(loginSessionsInCredentials.id, sessionId));

	return {
		user: user[0],
		session: foundSession
	};
}

export async function deactivateSession(sessionId: string): Promise<boolean> {
	const result = await db
		.update(loginSessionsInCredentials)
		.set({
			deactivatedAt: new Date().toISOString()
		})
		.where(eq(loginSessionsInCredentials.id, sessionId));

	return result.length ? result.length > 0 : false;
}

export function getSessionExpirationDate(days: number = 30): Date {
	const expireAt = new Date();
	expireAt.setDate(expireAt.getDate() + days);
	return expireAt;
}
