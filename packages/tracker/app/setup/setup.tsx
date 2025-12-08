import type { Route } from "./+types/setup";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Form, redirect } from "react-router";
import { db } from "@lib/db";
import { users } from "@lib/db/schema";
import { createUser, createSession } from "@lib/auth";
import { eq } from "drizzle-orm";

export function meta({}: Route.MetaArgs) {
	return [
		{ title: "Initial Setup" },
		{ name: "description", content: "Create initial admin user" }
	];
}

export async function loader() {
	// Check if there are any users
	const existingUsers = await db.select().from(users).limit(1);

	// If users exist, redirect to login
	if (existingUsers.length > 0) {
		return redirect("/login");
	}

	return {};
}

export async function action({ request }: Route.ActionArgs) {
	const formData = await request.formData();
	const username = formData.get("username") as string;
	const password = formData.get("password") as string;

	if (!username || !password) {
		return { error: "Username and password are required" };
	}

	try {
		// Create admin user
		const userId = await createUser(username, password);

		// Make user admin
		await db.update(users).set({ isAdmin: true }).where(eq(users.id, userId));

		// Create session and redirect
		const sessionId = await createSession(userId);

		const headers = new Headers();
		headers.append(
			"Set-Cookie",
			`session=${sessionId}; Path=/; HttpOnly; SameSite=Lax; Max-Age=${30 * 24 * 60 * 60}`
		);
		headers.append("Location", "/");

		return redirect("/", { headers });
	} catch (error) {
		console.error("Failed to create admin user:", error);
		return { error: "Failed to create admin user. Please try again." };
	}
}

export default function SetupPage() {
	return (
		<div className="min-h-screen flex items-center justify-center bg-background p-4">
			<Card className="w-full max-w-md">
				<CardHeader>
					<CardTitle>Initial Setup</CardTitle>
					<CardDescription>Create the first admin user for the system</CardDescription>
				</CardHeader>
				<CardContent>
					<Form method="post" className="space-y-4">
						<div className="space-y-2">
							<Label htmlFor="username">Admin Username</Label>
							<Input
								id="username"
								name="username"
								type="text"
								required
								placeholder="Enter admin username"
							/>
						</div>
						<div className="space-y-2">
							<Label htmlFor="password">Password</Label>
							<Input
								id="password"
								name="password"
								type="password"
								required
								placeholder="Enter password"
							/>
						</div>
						<Button type="submit" className="w-full">
							Create Admin User
						</Button>
					</Form>
				</CardContent>
			</Card>
		</div>
	);
}
