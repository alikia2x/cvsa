import { hashPassword, passwordMatches } from "@lib/auth";
import { getCurrentUser } from "@lib/auth-utils";
import { db } from "@lib/db";
import { users } from "@lib/db/schema";
import { eq } from "drizzle-orm";
import { ArrowLeft } from "lucide-react";
import { useEffect, useState } from "react";
import { Form, Link, useActionData } from "react-router";
import Layout from "@/components/layout";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import type { Route } from "./+types/profile";

export function meta({}: Route.MetaArgs) {
	return [
		{ title: "User Profile" },
		{ content: "Manage your account settings", name: "description" },
	];
}

export async function loader({ request }: Route.LoaderArgs) {
	const user = await getCurrentUser(request);
	if (!user) {
		throw new Response("Unauthorized", { status: 401 });
	}

	return { user };
}

export async function action({ request }: Route.ActionArgs) {
	const user = await getCurrentUser(request);
	if (!user) {
		throw new Response("Unauthorized", { status: 401 });
	}

	const formData = await request.formData();
	const intent = formData.get("intent") as string;

	if (intent === "changePassword") {
		const currentPassword = formData.get("currentPassword") as string;
		const newPassword = formData.get("newPassword") as string;
		const confirmPassword = formData.get("confirmPassword") as string;

		if (!currentPassword || !newPassword || !confirmPassword) {
			return { error: "All password fields are required" };
		}

		if (newPassword !== confirmPassword) {
			return { error: "New passwords do not match" };
		}

		const currentUser = await db.select().from(users).where(eq(users.id, user.id)).get();
		if (!currentUser) {
			return { error: "User not found" };
		}

		const isCurrentPasswordValid = await passwordMatches(currentPassword, currentUser.password);
		if (!isCurrentPasswordValid) {
			return { error: "Current password is incorrect" };
		}

		const hashedNewPassword = await hashPassword(newPassword);
		await db
			.update(users)
			.set({ password: hashedNewPassword, updatedAt: new Date() })
			.where(eq(users.id, user.id));

		return { message: "Password updated successfully", success: true };
	}

	if (intent === "changeUsername") {
		const newUsername = formData.get("newUsername") as string;

		if (!newUsername) {
			return { error: "Username is required" };
		}

		const existingUser = await db
			.select()
			.from(users)
			.where(eq(users.username, newUsername))
			.get();
		if (existingUser && existingUser.id !== user.id) {
			return { error: "Username already exists" };
		}

		await db
			.update(users)
			.set({ updatedAt: new Date(), username: newUsername })
			.where(eq(users.id, user.id));

		const updatedUser = await db.select().from(users).where(eq(users.id, user.id)).get();

		return {
			message: "Username updated successfully",
			success: true,
			updatedUser,
		};
	}

	return { error: "Unknown action" };
}

export default function UserProfile({ loaderData }: Route.ComponentProps) {
	const { user } = loaderData;
	const actionData = useActionData();
	const [userName, setUserName] = useState(user.username);
	const [error, setError] = useState<string | null>(null);
	const [success, setSuccess] = useState<string | null>(null);

	// 处理动作返回的消息
	useEffect(() => {
		if (actionData?.error) {
			setError(actionData.error);
			setSuccess(null);
		} else if (actionData?.success) {
			setSuccess(actionData.message);
			setError(null);

			// 如果用户名更新了，同步状态
			if (actionData.updatedUser?.username) {
				setUserName(actionData.updatedUser.username);
			}
		}
	}, [actionData]);

	return (
		<Layout>
			{/* 头部 */}
			<div className="max-sm:flex-col max-sm:gap-6 flex sm:items-center justify-between mb-8">
				<div>
					<h1 className="text-3xl font-bold tracking-tight">User Profile</h1>
					<p className="text-muted-foreground mt-2">Manage your account settings</p>
				</div>
				<div className="flex gap-2">
					<Button variant="outline" asChild>
						<Link to="/">
							<ArrowLeft className="size-4.5 mr-1" />
							Back to Projects
						</Link>
					</Button>
				</div>
			</div>

			{/* 消息提示 */}
			{error && <div className="mb-4 p-3 bg-red-50 text-red-600 rounded-md">{error}</div>}
			{success && (
				<div className="mb-4 p-3 bg-green-50 text-green-600 rounded-md">{success}</div>
			)}

			{/* 用户信息 */}
			<Card className="mb-6">
				<CardHeader>
					<CardTitle>Account Information</CardTitle>
					<CardDescription>Your basic account details</CardDescription>
				</CardHeader>
				<CardContent>
					<Form method="post">
						<input type="hidden" name="intent" value="changeUsername" />
						<div className="grid gap-4">
							<div>
								<Label htmlFor="username">Username</Label>
								<div className="flex gap-2 mt-1">
									<Input
										id="username"
										name="newUsername"
										onChange={(e) => setUserName(e.target.value)}
										value={userName}
										className="flex-1"
									/>
									<Button type="submit" variant="outline">
										Change
									</Button>
								</div>
							</div>
							<div>
								<Label htmlFor="role">Role</Label>
								<Input
									id="role"
									value={user.isAdmin ? "Administrator" : "User"}
									disabled
									className="mt-1"
								/>
							</div>
							<div>
								<Label htmlFor="created">Account Created</Label>
								<Input
									id="created"
									value={new Date(user.createdAt).toLocaleDateString()}
									disabled
									className="mt-1"
								/>
							</div>
						</div>
					</Form>
				</CardContent>
			</Card>

			{/* 更改密码 */}
			<Card>
				<CardHeader>
					<CardTitle>Change Password</CardTitle>
					<CardDescription>Update your account password</CardDescription>
				</CardHeader>
				<CardContent>
					<Form method="post">
						<input type="hidden" name="intent" value="changePassword" />
						<div className="grid gap-4">
							<div>
								<Label htmlFor="currentPassword">Current Password</Label>
								<Input
									id="currentPassword"
									name="currentPassword"
									type="password"
									placeholder="Enter your current password"
									required
									className="mt-1"
								/>
							</div>
							<div>
								<Label htmlFor="newPassword">New Password</Label>
								<Input
									id="newPassword"
									name="newPassword"
									type="password"
									placeholder="Enter new password"
									required
									className="mt-1"
								/>
							</div>
							<div>
								<Label htmlFor="confirmPassword">Confirm New Password</Label>
								<Input
									id="confirmPassword"
									name="confirmPassword"
									type="password"
									placeholder="Confirm new password"
									required
									className="mt-1"
								/>
							</div>
							<div>
								<Button type="submit">Change Password</Button>
							</div>
						</div>
					</Form>
				</CardContent>
			</Card>
		</Layout>
	);
}
