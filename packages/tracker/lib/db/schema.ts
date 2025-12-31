import { relations } from "drizzle-orm";
import { integer, sqliteTable, text } from "drizzle-orm/sqlite-core";

// Users table
export const users = sqliteTable("users", {
	createdAt: integer("created_at", { mode: "timestamp" }).notNull(),
	id: text("id").primaryKey(),
	isAdmin: integer("is_admin", { mode: "boolean" }).default(false),
	password: text("password_hash").notNull(),
	updatedAt: integer("updated_at", { mode: "timestamp" }).notNull(),
	username: text("username").notNull(),
});

// Sessions table for authentication
export const sessions = sqliteTable("sessions", {
	createdAt: integer("created_at", { mode: "timestamp" }).notNull(),
	expiresAt: integer("expires_at", { mode: "timestamp" }).notNull(),
	id: text("id").primaryKey(),
	userId: text("user_id")
		.notNull()
		.references(() => users.id, { onDelete: "cascade" }),
});

// Project permissions table
export const projectPermissions = sqliteTable("project_permissions", {
	canEdit: integer("can_edit", { mode: "boolean" }).default(false),
	createdAt: integer("created_at", { mode: "timestamp" }).notNull(),
	id: text("id").primaryKey(),
	projectId: text("project_id")
		.notNull()
		.references(() => projects.id, { onDelete: "cascade" }),
	userId: text("user_id")
		.notNull()
		.references(() => users.id, { onDelete: "cascade" }),
});

// Projects table
export const projects = sqliteTable("projects", {
	createdAt: integer("created_at", { mode: "timestamp" }).notNull(),
	description: text("description"),
	id: text("id").primaryKey(),
	isPublic: integer("is_public", { mode: "boolean" }).default(false),
	name: text("name").notNull(),
	ownerId: text("owner_id")
		.notNull()
		.references(() => users.id, { onDelete: "cascade" }),
	updatedAt: integer("updated_at", { mode: "timestamp" }).notNull(),
});

// Columns table for Kanban board
export const columns = sqliteTable("columns", {
	createdAt: integer("created_at", { mode: "timestamp" }).notNull(),
	id: text("id").primaryKey(),
	name: text("name").notNull(),
	position: integer("position").notNull(),
	projectId: text("project_id")
		.notNull()
		.references(() => projects.id, { onDelete: "cascade" }),
	updatedAt: integer("updated_at", { mode: "timestamp" }).notNull(),
});

// Tasks table
export const tasks = sqliteTable("tasks", {
	columnId: text("column_id")
		.notNull()
		.references(() => columns.id, { onDelete: "cascade" }),
	createdAt: integer("created_at", { mode: "timestamp" }).notNull(),
	description: text("description"),
	dueDate: integer("due_date", { mode: "timestamp" }),
	id: text("id").primaryKey(),
	priority: text("priority", { enum: ["low", "medium", "high"] }).default("medium"),
	projectId: text("project_id")
		.notNull()
		.references(() => projects.id, { onDelete: "cascade" }),
	title: text("title").notNull(),
	updatedAt: integer("updated_at", { mode: "timestamp" }).notNull(),
});

// Relations
export const usersRelations = relations(users, ({ many }) => ({
	ownedProjects: many(projects, { relationName: "owner" }),
	projectPermissions: many(projectPermissions),
	sessions: many(sessions),
}));

export const sessionsRelations = relations(sessions, ({ one }) => ({
	user: one(users, {
		fields: [sessions.userId],
		references: [users.id],
	}),
}));

export const projectsRelations = relations(projects, ({ one, many }) => ({
	columns: many(columns),
	owner: one(users, {
		fields: [projects.ownerId],
		references: [users.id],
		relationName: "owner",
	}),
	permissions: many(projectPermissions),
	tasks: many(tasks),
}));

export const projectPermissionsRelations = relations(projectPermissions, ({ one }) => ({
	project: one(projects, {
		fields: [projectPermissions.projectId],
		references: [projects.id],
	}),
	user: one(users, {
		fields: [projectPermissions.userId],
		references: [users.id],
	}),
}));

export const columnsRelations = relations(columns, ({ one, many }) => ({
	project: one(projects, {
		fields: [columns.projectId],
		references: [projects.id],
	}),
	tasks: many(tasks),
}));

export const tasksRelations = relations(tasks, ({ one }) => ({
	column: one(columns, {
		fields: [tasks.columnId],
		references: [columns.id],
	}),
	project: one(projects, {
		fields: [tasks.projectId],
		references: [projects.id],
	}),
}));

// Types
export type User = typeof users.$inferSelect;
export type NewUser = typeof users.$inferInsert;

export type Session = typeof sessions.$inferSelect;
export type NewSession = typeof sessions.$inferInsert;

export type Project = typeof projects.$inferSelect;
export type NewProject = typeof projects.$inferInsert;

export type ProjectPermission = typeof projectPermissions.$inferSelect;
export type NewProjectPermission = typeof projectPermissions.$inferInsert;

export type Column = typeof columns.$inferSelect;
export type NewColumn = typeof columns.$inferInsert;

export type Task = typeof tasks.$inferSelect;
export type NewTask = typeof tasks.$inferInsert;
