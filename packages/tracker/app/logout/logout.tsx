import { deleteSession } from "@lib/auth";
import { getCurrentUser } from "@lib/auth-utils";
import { db } from "@lib/db";
import { users } from "@lib/db/schema";
import { Link, redirect } from "react-router";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export async function loader({ request }: { request: Request }) {
	const existingUsers = await db.select().from(users).limit(1);

	if (existingUsers.length === 0) {
		return redirect("/setup");
	}

	const user = await getCurrentUser(request);
	if (!user) {
		return redirect("/login");
	}

	const cookies = request.headers.get("Cookie");
	const sessionMatch = cookies?.match(/session=([^;]+)/);
	const sessionId = sessionMatch?.[1];

	if (sessionId) {
		await deleteSession(sessionId);
	}

	// Clear session cookie
	const headers = new Headers();
	headers.append("Set-Cookie", "session=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0");
	headers.append("Location", "/login");
}

export default function LogoutPage() {
	return (
		<div className="min-h-screen flex items-center justify-center bg-background p-4">
			<Card className="w-full max-w-md">
				<CardHeader>
					<CardTitle className="text-xl font-bold">Log out</CardTitle>
				</CardHeader>
				<CardContent className="flex flex-col gap-4">
					<p>You have been logged out.</p>
					<Link to="/login">
						<Button>Login</Button>
					</Link>
				</CardContent>
			</Card>
		</div>
	);
}
