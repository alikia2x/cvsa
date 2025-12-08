import { Form, redirect } from "react-router";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { getCurrentUser } from "@lib/auth-utils";
import { db } from "@lib/db";
import { users } from "@lib/db/schema";
import type { Route } from "./+types/page";
import { useEffect } from "react";
import { toast } from "sonner";

export async function loader({ request }: { request: Request }) {
	const existingUsers = await db.select().from(users).limit(1);

	if (existingUsers.length === 0) {
		return redirect("/setup");
	}

	const user = await getCurrentUser(request);
	if (user) {
		return redirect("/");
	}
}

export default function LoginPage({ actionData }: Route.ComponentProps) {
	useEffect(() => {
		if (actionData?.error) {
			toast(actionData.error);
		}
	}, [actionData]);

	return (
		<div className="min-h-screen flex items-center justify-center bg-background p-4">
			<Card className="w-full max-w-md">
				<CardHeader>
					<CardTitle>Login</CardTitle>
					<CardDescription>Sign in to your account</CardDescription>
				</CardHeader>
				<CardContent>
					<Form method="post" className="space-y-4">
						<div className="space-y-2">
							<Label htmlFor="username">Username</Label>
							<Input
								id="username"
								name="username"
								type="text"
								required
								placeholder="Enter your username"
							/>
						</div>
						<div className="space-y-2">
							<Label htmlFor="password">Password</Label>
							<Input
								id="password"
								name="password"
								type="password"
								required
								placeholder="Enter your password"
							/>
						</div>
						<Button type="submit" className="w-full">
							Sign In
						</Button>
					</Form>
				</CardContent>
			</Card>
		</div>
	);
}

export { action } from "./action";
