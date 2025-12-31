import { canUserEditProject } from "@lib/auth";
import { getCurrentUser } from "@lib/auth-utils";
import { db } from "@lib/db";
import {
	type Project,
	type ProjectPermission,
	projectPermissions,
	projects,
	type User,
	users,
} from "@lib/db/schema";
import { and, eq, like } from "drizzle-orm";
import { ArrowLeft, Save, Search, Trash2, UserMinus, UserPlus, X } from "lucide-react";
import { useState } from "react";
import { Form, Link, redirect } from "react-router";
import Layout from "@/components/layout";
import { UserSearchModal } from "@/components/project/UserSearch";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import {
	Dialog,
	DialogContent,
	DialogDescription,
	DialogHeader,
	DialogTitle,
	DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
	Table,
	TableBody,
	TableCell,
	TableHead,
	TableHeader,
	TableRow,
} from "@/components/ui/table";
import { Textarea } from "@/components/ui/textarea";
import type { Route } from "./+types/settings";

export function meta({}: Route.MetaArgs) {
	return [
		{ title: "Project Settings" },
		{ content: "Manage project settings and permissions", name: "description" },
	];
}

export async function loader({ request, params }: Route.LoaderArgs) {
	const user = await getCurrentUser(request);
	if (!user) {
		throw new Response("Unauthorized", { status: 401 });
	}

	const projectId = params.id;
	if (!projectId) {
		throw new Response("Project ID required", { status: 400 });
	}

	// Get project details
	const project = await db.select().from(projects).where(eq(projects.id, projectId)).get();
	if (!project) {
		throw new Response("Project not found", { status: 404 });
	}

	// Check if user can edit this project
	const canEdit = await canUserEditProject(user.id, projectId);
	if (!canEdit) {
		throw new Response("Forbidden", { status: 403 });
	}

	// Get all users for the user management section
	const allUsers = await db.select().from(users).orderBy(users.username);

	// Get current project permissions
	const currentPermissions = await db
		.select()
		.from(projectPermissions)
		.where(eq(projectPermissions.projectId, projectId));

	const isOwner = await db
		.select()
		.from(projects)
		.where(and(eq(projects.id, projectId), eq(projects.ownerId, user.id)))
		.get();

	return { allUsers, currentPermissions, currentUser: user, isOwner, project };
}

export async function action({ request, params }: Route.ActionArgs) {
	const user = await getCurrentUser(request);
	const projectId = params.id;
	if (!projectId) {
		throw new Response("Project ID required", { status: 400 });
	}

	const project = await db.select().from(projects).where(eq(projects.id, projectId)).get();
	if (!user) {
		throw new Response("Unauthorized", { status: 401 });
	}

	const formData = await request.formData();
	const intent = formData.get("intent") as string;

	// Check if user can edit this project
	const canEdit = await canUserEditProject(user.id, projectId);
	if (!canEdit) {
		throw new Response("Forbidden", { status: 403 });
	}

	const isOwner = await db
		.select()
		.from(projects)
		.where(and(eq(projects.id, projectId), eq(projects.ownerId, user.id)))
		.get();

	if (intent === "updateProject") {
		const name = formData.get("name") as string;
		const description = formData.get("description") as string;
		const isPublic = isOwner ? formData.get("isPublic") === "on" : project?.isPublic;

		if (!name) {
			return { error: "Project name is required" };
		}

		await db
			.update(projects)
			.set({
				description,
				isPublic,
				name,
				updatedAt: new Date(),
			})
			.where(eq(projects.id, projectId));

		return redirect(`/project/${projectId}`);
	}

	// Danger zone (below): only project owner have access
	if (!project || project.ownerId !== user.id) {
		throw new Response("You do not have permission to edit this project", { status: 403 });
	}

	if (intent === "deleteProject") {
		await db.delete(projects).where(eq(projects.id, projectId));
		return redirect(`/projects`);
	}

	if (intent === "addUser") {
		const userId = formData.get("userId") as string;
		const canEditPermission = formData.get("canEdit") === "on";

		if (!userId) {
			return { error: "User ID is required" };
		}

		// Check if permission already exists
		const existingPermission = await db
			.select()
			.from(projectPermissions)
			.where(
				and(
					eq(projectPermissions.projectId, projectId),
					eq(projectPermissions.userId, userId)
				)
			)
			.get();

		if (existingPermission) {
			return { error: "User already has permission for this project" };
		}

		await db.insert(projectPermissions).values({
			canEdit: canEditPermission,
			createdAt: new Date(),
			id: crypto.randomUUID(),
			projectId,
			userId,
		});

		return redirect(`/project/${projectId}`);
	}

	if (intent === "removeUser") {
		const userId = formData.get("userId") as string;

		if (!userId) {
			return { error: "User ID is required" };
		}

		// Don't allow removing the project owner
		const project = await db.select().from(projects).where(eq(projects.id, projectId)).get();
		if (project && project.ownerId === userId) {
			return { error: "Cannot remove project owner" };
		}

		await db
			.delete(projectPermissions)
			.where(
				and(
					eq(projectPermissions.projectId, projectId),
					eq(projectPermissions.userId, userId)
				)
			);

		return redirect(`/project/${projectId}`);
	}

	if (intent === "updatePermission") {
		const userId = formData.get("userId") as string;
		const canEdit = formData.get("canEdit") === "on";

		if (!userId) {
			return { error: "User ID is required" };
		}

		// Don't allow changing the project owner's permissions
		const project = await db.select().from(projects).where(eq(projects.id, projectId)).get();
		if (project && project.ownerId === userId) {
			return { error: "Cannot change project owner's permissions" };
		}

		await db
			.update(projectPermissions)
			.set({ canEdit })
			.where(
				and(
					eq(projectPermissions.projectId, projectId),
					eq(projectPermissions.userId, userId)
				)
			);

		return redirect(`/project/${projectId}`);
	}

	return { error: "Unknown action" };
}

interface UsersManagementProps {
	project: Project;
	availableUsers: User[];
	currentPermissions: ProjectPermission[];
	allUsers: User[];
}

export function UsersManagement({
	project,
	availableUsers,
	currentPermissions,
	allUsers,
}: UsersManagementProps) {
	return (
		<Card className="mb-6">
			<CardHeader>
				<CardTitle>User Permissions</CardTitle>
				<CardDescription>Manage who can view and edit this project</CardDescription>
			</CardHeader>
			<CardContent>
				{/* Add User Form */}
				<div className="mb-6">
					<h3 className="text-lg font-semibold mb-4">Add User</h3>
					<UserSearchModal availableUsers={availableUsers} projectId={project.id} />
				</div>

				{/* Current Users Table */}
				<div>
					<h3 className="text-lg font-semibold mb-4">Current Users</h3>
					<Table>
						<TableHeader>
							<TableRow>
								<TableHead>Username</TableHead>
								<TableHead>Role</TableHead>
								<TableHead>Permissions</TableHead>
								<TableHead>Actions</TableHead>
							</TableRow>
						</TableHeader>
						<TableBody>
							{/* Project Owner */}
							<TableRow>
								<TableCell className="font-medium">
									{allUsers.find((u) => u.id === project.ownerId)?.username}
									<Badge variant="default" className="ml-2">
										Owner
									</Badge>
								</TableCell>
								<TableCell>Owner</TableCell>
								<TableCell>Full Access</TableCell>
								<TableCell>-</TableCell>
							</TableRow>

							{/* Users with permissions */}
							{currentPermissions.map((permission) => {
								const user = allUsers.find((u) => u.id === permission.userId);
								if (!user) return null;

								return (
									<TableRow key={permission.id}>
										<TableCell className="font-medium">
											{user.username}
										</TableCell>
										<TableCell>Collaborator</TableCell>
										<TableCell>
											<Form method="post">
												<input
													type="hidden"
													name="intent"
													value="updatePermission"
												/>
												<input
													type="hidden"
													name="userId"
													value={user.id}
												/>
												<div className="flex items-center gap-2">
													<Checkbox
														id={`canEdit-${user.id}`}
														name="canEdit"
														defaultChecked={permission.canEdit || false}
														className="rounded border-gray-300"
														onCheckedChange={(e) => {
															const canEdit = e;
															const formData = new FormData();
															formData.append(
																"intent",
																"updatePermission"
															);
															formData.append("userId", user.id);
															formData.append(
																"canEdit",
																canEdit ? "on" : ""
															);
															fetch(
																`/project/${project.id}/settings`,
																{
																	body: formData,
																	method: "POST",
																}
															);
														}}
													/>
													<Label htmlFor={`canEdit-${user.id}`}>
														Can Edit
													</Label>
												</div>
											</Form>
										</TableCell>
										<TableCell>
											<Form method="post">
												<input
													type="hidden"
													name="intent"
													value="removeUser"
												/>
												<input
													type="hidden"
													name="userId"
													value={user.id}
												/>
												<Button
													type="submit"
													variant="destructive"
													size="sm"
												>
													<UserMinus className="size-4 mr-1" />
													Remove
												</Button>
											</Form>
										</TableCell>
									</TableRow>
								);
							})}
						</TableBody>
					</Table>
				</div>
			</CardContent>
		</Card>
	);
}

export default function ProjectSettings({ loaderData }: Route.ComponentProps) {
	const { project, allUsers, currentPermissions, currentUser } = loaderData;

	// Create a map of user permissions for easy lookup
	const userPermissions = new Map();
	currentPermissions.forEach((permission) => {
		userPermissions.set(permission.userId, permission);
	});

	// Get users who don't have permissions yet
	const availableUsers = allUsers.filter(
		(user) => !userPermissions.has(user.id) && user.id !== project.ownerId
	);

	return (
		<Layout>
			{/* Header */}
			<div className="max-sm:flex-col max-sm:gap-6 flex sm:items-center justify-between mb-8">
				<div>
					<h1 className="text-3xl font-bold tracking-tight">Project Settings</h1>
					<p className="text-muted-foreground mt-2">
						Manage project details and user permissions
					</p>
				</div>
				<div className="flex gap-2">
					<Button variant="outline" asChild>
						<Link to={`/project/${project.id}`}>
							<ArrowLeft className="size-4.5 mr-1" />
							Back to Project
						</Link>
					</Button>
				</div>
			</div>

			{/* Project Details */}
			<Card className="mb-6">
				<CardHeader>
					<CardTitle>Project Details</CardTitle>
					<CardDescription>
						Update project name, description, and visibility
					</CardDescription>
				</CardHeader>
				<CardContent>
					<Form method="post">
						<input type="hidden" name="intent" value="updateProject" />
						<div className="grid gap-4">
							<div className="grid gap-2">
								<Label htmlFor="name">Project Name</Label>
								<Input
									id="name"
									name="name"
									defaultValue={project.name}
									placeholder="Enter project name"
									required
								/>
							</div>
							<div className="grid gap-2">
								<Label htmlFor="description">Description</Label>
								<Textarea
									id="description"
									name="description"
									defaultValue={project.description || ""}
									placeholder="Enter project description (optional)"
									rows={3}
								/>
							</div>
							{loaderData.isOwner && (
								<div className="flex items-center gap-2">
									<Checkbox
										id="isPublic"
										name="isPublic"
										defaultChecked={project.isPublic || false}
										className="rounded border-gray-300"
									/>
									<Label htmlFor="isPublic">Public Project</Label>
								</div>
							)}
							<div>
								<Button type="submit">
									<Save className="size-4 mr-1" />
									Save Changes
								</Button>
							</div>
						</div>
					</Form>
				</CardContent>
			</Card>

			{loaderData.isOwner && (
				<UsersManagement
					project={project}
					allUsers={allUsers}
					currentPermissions={currentPermissions}
					availableUsers={availableUsers}
				/>
			)}

			{/* Danger Zone */}
			{project.ownerId === currentUser.id && (
				<Card className="border-destructive">
					<CardHeader>
						<CardTitle className="text-destructive">Danger Zone</CardTitle>
						<CardDescription>
							Permanently delete this project and all its data
						</CardDescription>
					</CardHeader>
					<CardContent>
						<Form method="post">
							<input type="hidden" name="intent" value="deleteProject" />
							<Button
								type="submit"
								variant="destructive"
								onClick={(e) => {
									if (
										!confirm(
											"Are you sure you want to delete this project? This action cannot be undone."
										)
									) {
										e.preventDefault();
									}
								}}
							>
								<Trash2 className="size-4 mr-1" />
								Delete Project
							</Button>
						</Form>
					</CardContent>
				</Card>
			)}
		</Layout>
	);
}
