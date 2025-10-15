import type { Route } from "./+types/users";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
	Table,
	TableBody,
	TableCell,
	TableHead,
	TableHeader,
	TableRow
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { ArrowLeft, Plus, Trash2, Edit, Shield, ShieldOff, UserPlus } from "lucide-react";
import { Link, Form } from "react-router";
import { db } from "@lib/db";
import { users } from "@lib/db/schema";
import { getCurrentUser } from "@lib/auth-utils";
import Layout from "@/components/layout";
import { eq } from "drizzle-orm";
import { hashPassword } from "@lib/auth";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { generate as generateId } from "@alikia/random-key";
import { Checkbox } from "@/components/ui/checkbox";

export function meta({}: Route.MetaArgs) {
	return [
		{ title: "User Management - Admin" },
		{ name: "description", content: "Manage users and permissions" }
	];
}

export async function loader({ request }: Route.LoaderArgs) {
	const user = await getCurrentUser(request);
	if (!user || !user.isAdmin) {
		throw new Response("You do not have permission to view this page", { status: 403 });
	}

	// Fetch all users
	const allUsers = await db.select().from(users).orderBy(users.createdAt);

	return { users: allUsers, currentUser: user };
}

export async function action({ request }: Route.ActionArgs) {
	const user = await getCurrentUser(request);
	if (!user || !user.isAdmin) {
		throw new Response("You do not have permission to view this page", { status: 403 });
	}

	const formData = await request.formData();
	const intent = formData.get("intent");
	const userId = formData.get("userId") as string;

	if (intent === "toggleAdmin") {
		const targetUser = await db.select().from(users).where(eq(users.id, userId)).get();
		if (!targetUser) {
			return { error: "User not found" };
		}

		// Prevent self-demotion
		if (targetUser.id === user.id) {
			return { error: "Cannot change your own admin status" };
		}

		await db
			.update(users)
			.set({ isAdmin: !targetUser.isAdmin })
			.where(eq(users.id, userId));

		return { success: true };
	}

	if (intent === "deleteUser") {
		const targetUser = await db.select().from(users).where(eq(users.id, userId)).get();
		if (!targetUser) {
			return { error: "User not found" };
		}

		// Prevent self-deletion
		if (targetUser.id === user.id) {
			return { error: "Cannot delete your own account" };
		}

		// Prevent deleting admin users
		if (targetUser.isAdmin) {
			return { error: "Cannot delete admin users" };
		}

		await db.delete(users).where(eq(users.id, userId));

		return { success: true };
	}

	if (intent === "createUser") {
		const username = formData.get("username") as string;
		const password = formData.get("password") as string;
		const isAdmin = formData.get("isAdmin") === "on";

		if (!username || !password) {
			return { error: "Username and password are required" };
		}

		// Check if username already exists
		const existingUser = await db.select().from(users).where(eq(users.username, username)).get();
		if (existingUser) {
			return { error: "Username already exists" };
		}

		const hashedPassword = await hashPassword(password);
		await db.insert(users).values({
			id: await generateId(6),
			username,
			password: hashedPassword,
			isAdmin,
			createdAt: new Date(),
			updatedAt: new Date()
		});

		return { success: true };
	}

	if (intent === "updateUser") {
		const username = formData.get("username") as string;
		const password = formData.get("password") as string;
		const isAdmin = formData.get("isAdmin") === "on";

		if (!username) {
			return { error: "Username is required" };
		}

		const targetUser = await db.select().from(users).where(eq(users.id, userId)).get();
		if (!targetUser) {
			return { error: "User not found" };
		}

		// Check if username already exists (excluding current user)
		const existingUser = await db.select().from(users).where(eq(users.username, username)).get();
		if (existingUser && existingUser.id !== userId) {
			return { error: "Username already exists" };
		}

		const updateData: any = {
			username,
			isAdmin,
		};

		// Only update password if provided
		if (password) {
			updateData.password = await hashPassword(password);
		}

		await db
			.update(users)
			.set(updateData)
			.where(eq(users.id, userId));

		return { success: true };
	}

	return { error: "Unknown action" };
}

export default function UserManagement({ loaderData }: Route.ComponentProps) {
	const { users, currentUser } = loaderData;

	return (
		<Layout>
			{/* Header */}
			<div className="max-sm:flex-col max-sm:gap-6 flex sm:items-center justify-between mb-8">
				<div>
					<h1 className="text-3xl font-bold tracking-tight">User Management</h1>
					<p className="text-muted-foreground mt-2">Manage users and their permissions</p>
				</div>
				<div className="flex gap-2">
					<Dialog>
						<DialogTrigger asChild>
							<Button>
								<UserPlus className="size-4.5 mr-1" />
								Add User
							</Button>
						</DialogTrigger>
						<DialogContent>
							<DialogHeader>
								<DialogTitle>Add New User</DialogTitle>
								<DialogDescription>
									Create a new user account with username and password.
								</DialogDescription>
							</DialogHeader>
							<Form method="post">
								<input type="hidden" name="intent" value="createUser" />
								<div className="grid gap-4 py-4">
									<div className="grid gap-2">
										<Label htmlFor="username">Username</Label>
										<Input
											id="username"
											name="username"
											placeholder="Enter username"
											required
										/>
									</div>
									<div className="grid gap-2">
										<Label htmlFor="password">Password</Label>
										<Input
											id="password"
											name="password"
											type="password"
											placeholder="Enter password"
											required
										/>
									</div>
									<div className="flex items-center gap-2">
										<Checkbox
											id="isAdmin"
											name="isAdmin"
											className="rounded border-gray-300"
										/>
										<Label htmlFor="isAdmin">Admin User</Label>
									</div>
								</div>
								<DialogFooter>
									<Button type="submit">Create User</Button>
								</DialogFooter>
							</Form>
						</DialogContent>
					</Dialog>
					<Button variant="outline" asChild>
						<Link to="/">
							<ArrowLeft className="size-4.5 mr-1" />
							Back to Projects
						</Link>
					</Button>
				</div>
			</div>

			{/* Users Table */}
			<Card>
				<CardHeader>
					<CardTitle>Users</CardTitle>
					<CardDescription>Manage user accounts and permissions</CardDescription>
				</CardHeader>
				<CardContent>
					<Table>
						<TableHeader>
							<TableRow>
								<TableHead>Username</TableHead>
								<TableHead>Admin</TableHead>
								<TableHead>Created</TableHead>
								<TableHead>Actions</TableHead>
							</TableRow>
						</TableHeader>
						<TableBody>
							{users.map((user) => (
								<TableRow key={user.id}>
									<TableCell className="font-medium">{user.username}</TableCell>
									<TableCell>
										{user.isAdmin ? (
											<Badge variant="default">Admin</Badge>
										) : (
											<Badge variant="secondary">User</Badge>
										)}
									</TableCell>
									<TableCell>
										{new Date(user.createdAt).toLocaleDateString()}
									</TableCell>
									<TableCell>
										<div className="flex gap-2">
											<Dialog>
												<DialogTrigger asChild>
													<Button variant="outline" size="sm">
														<Edit className="size-4 mr-1" />
														Edit
													</Button>
												</DialogTrigger>
												<DialogContent>
													<DialogHeader>
														<DialogTitle>Edit User</DialogTitle>
														<DialogDescription>
															Update user information and permissions.
														</DialogDescription>
													</DialogHeader>
													<Form method="post">
														<input type="hidden" name="userId" value={user.id} />
														<input type="hidden" name="intent" value="updateUser" />
														<div className="grid gap-4 py-4">
															<div className="grid gap-2">
																<Label htmlFor={`username-${user.id}`}>Username</Label>
																<Input
																	id={`username-${user.id}`}
																	name="username"
																	defaultValue={user.username}
																	placeholder="Enter username"
																	required
																/>
															</div>
															<div className="grid gap-2">
																<Label htmlFor={`password-${user.id}`}>New Password</Label>
																<Input
																	id={`password-${user.id}`}
																	name="password"
																	type="password"
																	placeholder="Leave blank to keep current password"
																/>
															</div>
															<div className="flex items-center gap-2">
																<input
																	type="checkbox"
																	id={`isAdmin-${user.id}`}
																	name="isAdmin"
																	defaultChecked={user.isAdmin || false}
																	className="rounded border-gray-300"
																	disabled={user.id === currentUser.id}
																/>
																<Label htmlFor={`isAdmin-${user.id}`}>Admin User</Label>
																{user.id === currentUser.id && (
																	<span className="text-xs text-muted-foreground">(Cannot change your own admin status)</span>
																)}
															</div>
														</div>
														<DialogFooter>
															<Button type="submit">Update User</Button>
														</DialogFooter>
													</Form>
												</DialogContent>
											</Dialog>
											<Form method="post">
												<input type="hidden" name="userId" value={user.id} />
												<input type="hidden" name="intent" value="toggleAdmin" />
												<Button
													type="submit"
													variant="outline"
													size="sm"
													disabled={user.id === currentUser.id} // Prevent self-demotion
												>
													{user.isAdmin ? (
														<ShieldOff className="size-4 mr-1" />
													) : (
														<Shield className="size-4 mr-1" />
													)}
													{user.isAdmin ? "Remove Admin" : "Make Admin"}
												</Button>
											</Form>
											<Form method="post">
												<input type="hidden" name="userId" value={user.id} />
												<input type="hidden" name="intent" value="deleteUser" />
												<Button
													type="submit"
													variant="destructive"
													size="sm"
													disabled={user.isAdmin || user.id === currentUser.id} // Prevent deleting admin or self
												>
													<Trash2 className="size-4 mr-1" />
													Delete
												</Button>
											</Form>
										</div>
									</TableCell>
								</TableRow>
							))}
						</TableBody>
					</Table>
				</CardContent>
			</Card>
		</Layout>
	);
}
