import { redirect } from "react-router";
import { authenticateUser, createSession } from "@lib/auth";

export async function action({ request }: { request: Request }) {
	const formData = await request.formData();
	const username = formData.get("username") as string;
	const password = formData.get("password") as string;

	if (!username || !password) {
		return { error: "Username and password are required" };
	}

	const user = await authenticateUser(username, password);
	if (!user) {
		return { error: "Invalid username or password" };
	}

	const sessionId = await createSession(user.id);
	
	// Set session cookie
	const headers = new Headers();
	headers.append(
		"Set-Cookie",
		`session=${sessionId}; Path=/; HttpOnly; SameSite=Lax; Max-Age=${30 * 24 * 60 * 60}`
	);
	headers.append("Location", "/");

	return redirect("/", { headers });
}