import { canUserEditProject, canUserViewProject } from "@lib/auth";
import { getCurrentUser } from "@lib/auth-utils";
import { db } from "@lib/db";
import { type Column, columns, projects, type Task, tasks } from "@lib/db/schema";
import { asc, desc, eq } from "drizzle-orm";
import { ArrowLeft, Plus, SquarePen } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, useRevalidator } from "react-router";
import { ColumnDialog } from "@/components/column/ColumnDialog";
import Layout from "@/components/layout";
import { ProjectDialog } from "@/components/project/ProjectDialog";
import { TaskDialog } from "@/components/task/TaskDialog";
import { Button } from "@/components/ui/button";
import type { Route } from "./+types/projectPage";
import { projectPageAction } from "./projectPageAction";

export function meta({ loaderData }: Route.MetaArgs) {
	return [
		{ title: `${loaderData.project.name} - FramSpor` },
		{ content: `Manage tasks for ${loaderData.project.name}`, name: "description" },
	];
}

export async function loader({ params, request }: Route.LoaderArgs) {
	const user = await getCurrentUser(request);

	const projectId = params.id;

	// Check if user can view this project
	const canView = await canUserViewProject(user?.id || "", projectId);
	if (!canView) {
		throw new Response("You do not have permission to view this project", { status: 403 });
	}

	// Fetch the project
	const projectResult = await db
		.select()
		.from(projects)
		.where(eq(projects.id, projectId))
		.limit(1);

	if (projectResult.length === 0) {
		throw new Response("Project not found", { status: 404 });
	}

	const project = projectResult[0];

	// Check if user can edit this project
	const canEdit = await canUserEditProject(user?.id || "", projectId);

	// Fetch columns for this project
	const projectColumns = await db
		.select()
		.from(columns)
		.where(eq(columns.projectId, projectId))
		.orderBy(asc(columns.position));

	// Fetch tasks for each column
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

	return {
		canEdit,
		columns: columnsWithTasks,
		project,
		user,
	};
}

interface ColumnWithTasks extends Column {
	tasks: Task[];
}

export const action = projectPageAction;

export default function ProjectBoard({ loaderData }: Route.ComponentProps) {
	const { project, columns: initialColumns, user, canEdit } = loaderData;
	const [columns, setColumns] = useState(initialColumns);
	const [isTaskDialogOpen, setIsTaskDialogOpen] = useState(false);
	const [isColumnDialogOpen, setIsColumnDialogOpen] = useState(false);
	const [isProjectDialogOpen, setIsProjectDialogOpen] = useState(false);
	const [selectedColumnId, setSelectedColumnId] = useState<string | null>(null);
	const [editingTask, setEditingTask] = useState<any>(null);
	const [editingColumn, setEditingColumn] = useState<any>(null);
	const revalidator = useRevalidator();

	useEffect(() => {
		setColumns(loaderData.columns);
	}, [loaderData, loaderData.columns, loaderData.project]);

	const handleAddTask = (columnId?: string) => {
		setSelectedColumnId(columnId || null);
		setEditingTask(null);
		setIsTaskDialogOpen(true);
	};

	const handleEditTask = (task: Task) => {
		setEditingTask(task);
		setSelectedColumnId(null);
		setIsTaskDialogOpen(true);
	};

	const handleAddColumn = () => {
		setEditingColumn(null);
		setIsColumnDialogOpen(true);
	};

	const handleEditColumn = (column: ColumnWithTasks) => {
		setEditingColumn(column);
		setIsColumnDialogOpen(true);
	};

	const handleEditProject = () => {
		setIsProjectDialogOpen(true);
	};

	const handleColumnSubmit = async (data: { name: string }) => {
		const formData = new FormData();

		if (editingColumn) {
			formData.append("intent", "updateColumn");
			formData.append("columnId", editingColumn.id);
		} else {
			formData.append("intent", "createColumn");
		}

		formData.append("name", data.name);

		const response = await fetch(`/project/${project.id}`, {
			body: formData,
			method: "POST",
		});

		revalidator.revalidate();
	};

	const handleDeleteColumn = async (columnId: string) => {
		const formData = new FormData();
		formData.append("intent", "deleteColumn");
		formData.append("columnId", columnId);

		const response = await fetch(`/project/${project.id}`, {
			body: formData,
			method: "POST",
		});

		if (response.ok) {
			// Refresh the data
			revalidator.revalidate();
		}
	};

	const handleProjectSubmit = async (data: {
		name: string;
		description: string;
		isPublic: boolean;
	}) => {
		const formData = new FormData();
		formData.append("intent", "updateProject");
		formData.append("name", data.name);
		formData.append("description", data.description);
		formData.append("isPublic", data.isPublic ? "true" : "false");

		const response = await fetch(`/project/${project.id}`, {
			body: formData,
			method: "POST",
		});

		if (response.ok) {
			// Refresh the data
			revalidator.revalidate();
		}
	};

	const handleDeleteProject = async () => {
		const formData = new FormData();
		formData.append("intent", "deleteProject");

		const response = await fetch(`/project/${project.id}`, {
			body: formData,
			method: "POST",
		});

		if (response.ok) {
			// Redirect to home page
			window.location.href = "/";
		}
	};

	const handleTaskSubmit = async (data: {
		title: string;
		description: string;
		columnId: string;
		priority: "low" | "medium" | "high";
		dueDate?: Date;
	}) => {
		const formData = new FormData();

		if (editingTask) {
			formData.append("intent", "updateTask");
			formData.append("taskId", editingTask.id);
		} else {
			formData.append("intent", "createTask");
		}

		formData.append("title", data.title);
		formData.append("description", data.description);
		formData.append("columnId", selectedColumnId || data.columnId);
		formData.append("priority", data.priority);
		if (data.dueDate) {
			formData.append("dueDate", data.dueDate.toISOString());
		}

		const response = await fetch(`/project/${project.id}`, {
			body: formData,
			method: "POST",
		});

		revalidator.revalidate();
	};

	const handleTaskDelete = async () => {
		const formData = new FormData();
		formData.append("intent", "deleteTask");
		formData.append("taskId", editingTask.id);

		const response = await fetch(`/project/${project.id}`, {
			body: formData,
			method: "POST",
		});

		revalidator.revalidate();
	};

	return (
		<Layout>
			<TaskDialog
				open={isTaskDialogOpen}
				onOpenChange={setIsTaskDialogOpen}
				projectId={project.id}
				columns={columns}
				onSubmit={handleTaskSubmit}
				onDelete={handleTaskDelete}
				initialData={
					editingTask || (selectedColumnId ? { columnId: selectedColumnId } : undefined)
				}
				isEditing={!!editingTask}
				canEdit={canEdit}
			/>
			<ColumnDialog
				open={isColumnDialogOpen}
				onOpenChange={setIsColumnDialogOpen}
				onSubmit={handleColumnSubmit}
				onDelete={editingColumn ? () => handleDeleteColumn(editingColumn.id) : undefined}
				initialData={editingColumn}
				isEditing={!!editingColumn}
				columns={columns.length}
			/>
			<ProjectDialog
				open={isProjectDialogOpen}
				onOpenChange={setIsProjectDialogOpen}
				onSubmit={handleProjectSubmit}
				onDelete={handleDeleteProject}
				initialData={{
					description: project.description || "",
					name: project.name,
				}}
				isEditing={true}
			/>
			{/* Header */}
			<div className="max-md:flex-col max-md:gap-4 flex justify-between mb-8">
				<div className="flex items-start gap-4">
					<div>
						<h1 className="text-3xl font-bold tracking-tight">{project.name}</h1>
						<p className="text-muted-foreground mt-2">
							{project.description || "No description."}
							{!canEdit && <span className="ml-2 text-orange-600">(View Only)</span>}
						</p>
					</div>
				</div>
				<div className="max-sm:flex-col flex gap-2">
					{user && (
						<Button variant="outline" asChild>
							<Link to="/">
								<ArrowLeft className="size-4.5 mr-1" />
								Back to Projects
							</Link>
						</Button>
					)}
					{canEdit && (
						<>
							<Link to={`/project/${project.id}/settings`}>
								<Button variant="outline" onClick={handleEditProject}>
									<SquarePen className="size-4 mr-1" />
									Edit Project
								</Button>
							</Link>

							<Button onClick={() => handleAddTask()}>
								<Plus className="size-4.5 mr-1" />
								Add Task
							</Button>
						</>
					)}
				</div>
			</div>

			{/* Kanban Board */}
			<div className="max-lg:flex-col flex gap-4 overflow-x-auto pb-6">
				{columns.map((column) => (
					<div
						key={column.id}
						className="min-w-80 lg:w-100 xl:w-100 2xl:w-110 flex-shrink-0"
					>
						<div className="border rounded-lg p-4 px-5 bg-card">
							<div className="flex items-center justify-between mb-4">
								<h3 className="font-semibold text-lg">{column.name}</h3>
								{canEdit && (
									<Button
										variant="ghost"
										size="icon"
										onClick={() => handleEditColumn(column)}
									>
										<SquarePen className="w-4 h-4" />
									</Button>
								)}
							</div>
							<div className="space-y-3">
								{column.tasks.map((task) => (
									<div
										key={task.id}
										className="border rounded-md py-4 px-4.5 bg-background gap-3
										hover:shadow-sm transition-shadow cursor-pointer flex flex-col"
										onClick={() => handleEditTask(task)}
									>
										<h4 className="font-medium text-sm">{task.title}</h4>
										{task.description && (
											<p className="text-xs text-muted-foreground line-clamp-3 overflow-ellipsis">
												<pre>{task.description}</pre>
											</p>
										)}
										<div className="flex items-center justify-between text-xs">
											{task.priority && (
												<span
													className={`px-2 py-1 text-xs ${
														task.priority === "high"
															? "bg-red-100 text-red-800"
															: task.priority === "medium"
																? "bg-yellow-100 text-yellow-800"
																: "bg-blue-200 text-blue-800"
													}`}
												>
													{task.priority}
												</span>
											)}
											{task.dueDate && (
												<span className="text-muted-foreground">
													{new Date(task.dueDate).toLocaleDateString()}
												</span>
											)}
										</div>
									</div>
								))}
								{canEdit && (
									<Button
										variant="ghost"
										className="w-full justify-start text-muted-foreground"
										onClick={() => handleAddTask(column.id)}
									>
										<Plus className="w-4 h-4 mr-2" />
										Add Task
									</Button>
								)}
							</div>
						</div>
					</div>
				))}
				{canEdit && (
					<div className="w-80 flex-shrink-0">
						<div className="border-dashed border-2 rounded-lg p-6 flex items-center justify-center h-32">
							<Button
								variant="ghost"
								className="text-muted-foreground"
								onClick={handleAddColumn}
							>
								<Plus className="w-4 h-4 mr-2" />
								Add Column
							</Button>
						</div>
					</div>
				)}
			</div>
		</Layout>
	);
}
