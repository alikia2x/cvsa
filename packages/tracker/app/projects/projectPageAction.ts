import { generate as generateId } from "@alikia/random-key";
import { canUserEditProject } from "@lib/auth";
import { getCurrentUser } from "@lib/auth-utils";
import { db } from "@lib/db";
import { columns, projects, tasks } from "@lib/db/schema";
import { asc, desc, eq } from "drizzle-orm";
import type { Route } from "./+types/projectPage";

export const projectPageAction = async ({ request, params }: Route.ActionArgs) => {
	const user = await getCurrentUser(request);
	if (!user) {
		throw new Response("Unauthorized", { status: 401 });
	}

	const formData = await request.formData();
	const intent = formData.get("intent");
	const projectId = params.id;

	// Check if user can edit this project for write operations
	const canEdit = await canUserEditProject(user.id, projectId);
	if (!canEdit && intent !== "getColumns") {
		throw new Response("You do not have permission to edit this project", { status: 403 });
	}

	if (intent === "getColumns") {
		const projectId = formData.get("projectId") as string;

		const projectColumns = await db
			.select()
			.from(columns)
			.where(eq(columns.projectId, projectId))
			.orderBy(asc(columns.position));

		const columnsWithTasks = await Promise.all(
			projectColumns.map(async (column) => {
				const columnTasks = await db
					.select()
					.from(tasks)
					.where(eq(tasks.columnId, column.id))
					.orderBy(desc(tasks.priority), asc(tasks.dueDate));

				return {
					...column,
					tasks: columnTasks.sort((a, b) => {
						if (a.dueDate === null && b.dueDate === null) return 0;
						if (a.dueDate === null) return 1;
						if (b.dueDate === null) return -1;
						return a.dueDate.getTime() - b.dueDate.getTime();
					}),
				};
			})
		);
		return { columnsWithTasks };
	}

	if (intent === "createTask") {
		const title = formData.get("title") as string;
		const description = formData.get("description") as string;
		const columnId = formData.get("columnId") as string;
		const priority = formData.get("priority") as "low" | "medium" | "high";
		const dueDate = formData.get("dueDate") as string;

		if (!title || !columnId) {
			return { error: "Title and column are required" };
		}

		const taskId = await generateId(7);

		await db.insert(tasks).values({
			columnId: columnId,
			createdAt: new Date(),
			description: description,
			dueDate: dueDate ? new Date(dueDate) : null,
			id: taskId,
			priority: priority,
			projectId: projectId,
			title: title,
			updatedAt: new Date(),
		});

		return { success: true, taskId };
	}

	if (intent === "updateTask") {
		const taskId = formData.get("taskId") as string;
		const title = formData.get("title") as string;
		const description = formData.get("description") as string;
		const columnId = formData.get("columnId") as string;
		const priority = formData.get("priority") as "low" | "medium" | "high";
		const dueDate = formData.get("dueDate") as string;

		if (!title || !columnId || !taskId) {
			return { error: "Title, column, and task ID are required" };
		}

		await db
			.update(tasks)
			.set({
				columnId: columnId,
				description: description,
				dueDate: dueDate ? new Date(dueDate) : null,
				priority: priority,
				title: title,
				updatedAt: new Date(),
			})
			.where(eq(tasks.id, taskId));

		return { success: true, taskId };
	}

	if (intent === "deleteTask") {
		const taskId = formData.get("taskId") as string;

		if (!taskId) {
			return { error: "Task ID is required" };
		}

		await db.delete(tasks).where(eq(tasks.id, taskId));

		return { success: true, taskId };
	}

	if (intent === "createColumn") {
		const name = formData.get("name") as string;

		if (!name) {
			return { error: "Column name is required" };
		}

		const columnId = await generateId(7);

		// Get the highest position for this project
		const existingColumns = await db
			.select()
			.from(columns)
			.where(eq(columns.projectId, projectId))
			.orderBy(asc(columns.position));

		const newPosition =
			existingColumns.length > 0
				? existingColumns[existingColumns.length - 1].position + 1
				: 0;

		await db.insert(columns).values({
			createdAt: new Date(),
			id: columnId,
			name: name,
			position: newPosition,
			projectId: projectId,
			updatedAt: new Date(),
		});

		return { columnId, success: true };
	}

	if (intent === "updateColumn") {
		const columnId = formData.get("columnId") as string;
		const name = formData.get("name") as string;
		const position = formData.get("position") as string;

		if (!name || !columnId) {
			return { error: "Column name and ID are required" };
		}

		await db
			.update(columns)
			.set({
				name: name,
				position: parseInt(position) || 0,
				updatedAt: new Date(),
			})
			.where(eq(columns.id, columnId));

		return { columnId, success: true };
	}

	if (intent === "deleteColumn") {
		const columnId = formData.get("columnId") as string;

		if (!columnId) {
			return { error: "Column ID is required" };
		}

		// Check if column has tasks
		const columnTasks = await db.select().from(tasks).where(eq(tasks.columnId, columnId));

		if (columnTasks.length > 0) {
			return { error: "Cannot delete column with tasks. Please move or delete tasks first." };
		}

		await db.delete(columns).where(eq(columns.id, columnId));

		return { columnId, success: true };
	}

	if (intent === "reorderColumns") {
		const columnOrder = JSON.parse(formData.get("columnOrder") as string) as string[];

		// Update positions for all columns
		for (let i = 0; i < columnOrder.length; i++) {
			const columnId = columnOrder[i];
			await db
				.update(columns)
				.set({
					position: i,
					updatedAt: new Date(),
				})
				.where(eq(columns.id, columnId));
		}

		return { success: true };
	}

	if (intent === "updateProject") {
		const name = formData.get("name") as string;
		const description = formData.get("description") as string;
		const isPublic = formData.get("isPublic") === "true";

		if (!name) {
			return { error: "Project name is required" };
		}

		await db
			.update(projects)
			.set({
				description: description,
				isPublic: isPublic,
				name: name,
				updatedAt: new Date(),
			})
			.where(eq(projects.id, projectId));

		return { success: true };
	}

	if (intent === "deleteProject") {
		// Check if project has columns
		const projectColumns = await db
			.select()
			.from(columns)
			.where(eq(columns.projectId, projectId));

		if (projectColumns.length > 0) {
			return {
				error: "Cannot delete project with columns. Please delete all columns first.",
			};
		}

		await db.delete(projects).where(eq(projects.id, projectId));

		return { redirect: "/", success: true };
	}

	return { error: "Unknown action" };
};
