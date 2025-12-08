import { db } from "./db";
import { users, sessions, projectPermissions, projects } from "./db/schema";
import { eq, and, or, gte } from "drizzle-orm";
import { randomBytes } from "crypto";
import Argon2id from "@rabbit-company/argon2id";

export async function passwordMatches(password: string, storedPassword: string) {
	return Argon2id.verify(storedPassword, password);
}

export async function hashPassword(password: string) {
	return Argon2id.hashEncoded(password);
}

// Generate a random session ID
function generateSessionId(): string {
	return randomBytes(32).toString("hex");
}

// Create a new user
export async function createUser(username: string, password: string) {
	const userId = randomBytes(16).toString("hex");
	const now = new Date();
	const hashedPassword = await Argon2id.hashEncoded(password);

	await db.insert(users).values({
		id: userId,
		username,
		password: hashedPassword,
		createdAt: now,
		updatedAt: now
	});

	return userId;
}

// Authenticate user
export async function authenticateUser(username: string, password: string) {
	const user = await db.select().from(users).where(eq(users.username, username)).get();
	if (!user) return null;
	const verified = await passwordMatches(password, user.password);
	if (!verified) {
		return null;
	}

	return user;
}

// Create a session
export async function createSession(userId: string) {
	const sessionId = generateSessionId();
	const expiresAt = new Date(Date.now() + 30 * 24 * 60 * 60 * 1000); // 30 days

	await db.insert(sessions).values({
		id: sessionId,
		userId,
		expiresAt,
		createdAt: new Date()
	});

	return sessionId;
}

// Validate session
export async function validateSession(sessionId: string) {
	const session = await db
		.select()
		.from(sessions)
		.where(and(eq(sessions.id, sessionId), gte(sessions.expiresAt, new Date())))
		.get();

	if (!session) {
		return null;
	}

	// Get user data
	const user = await db.select().from(users).where(eq(users.id, session.userId)).get();

	return user;
}

// Check if user can edit project
export async function canUserEditProject(userId: string, projectId: string) {
	// Admin users can edit all projects
	const user = await db.select().from(users).where(eq(users.id, userId)).get();
	if (user?.isAdmin) {
		return true;
	}

	// Check if user is project owner
	const project = await db
		.select()
		.from(projects)
		.where(and(eq(projects.id, projectId), eq(projects.ownerId, userId)))
		.get();
	if (project) {
		return true;
	}

	// Check if user has edit permission
	const permission = await db
		.select()
		.from(projectPermissions)
		.where(
			and(eq(projectPermissions.projectId, projectId), eq(projectPermissions.userId, userId))
		)
		.get();

	return !!permission;
}

// Check if user can view project
export async function canUserViewProject(userID: string, projectId: string) {
	if (await canUserEditProject(userID, projectId)) {
		return true;
	}

	// Check if project is public
	const project = await db
		.select()
		.from(projects)
		.where(and(eq(projects.id, projectId), eq(projects.isPublic, true)))
		.get();

	return !!project;
}

// Get projects accessible to user
export async function getUserProjects(userId: string) {
	const user = await db.select().from(users).where(eq(users.id, userId)).get();

	if (user?.isAdmin) {
		// Admin can see all projects
		return await db.select().from(projects).all();
	}

	// Get projects where user is owner or has permission
	const accessibleProjects = await db
		.select()
		.from(projects)
		.where(or(eq(projects.ownerId, userId), eq(projectPermissions.userId, userId)))
		.leftJoin(projectPermissions, eq(projects.id, projectPermissions.projectId))
		.all();

	return accessibleProjects.map((row) => row.projects);
}

// Delete session (logout)
export async function deleteSession(sessionId: string) {
	await db.delete(sessions).where(eq(sessions.id, sessionId));
}
