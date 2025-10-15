import { validateSession } from "./auth";

export async function getCurrentUser(request: Request) {
	const cookies = request.headers.get("Cookie");
	const sessionMatch = cookies?.match(/session=([^;]+)/);
	const sessionId = sessionMatch?.[1];

	if (!sessionId) {
		return null;
	}

	return await validateSession(sessionId);
}

export async function requireAuth(request: Request) {
	const user = await getCurrentUser(request);
	if (!user) {
		throw new Response("Unauthorized", { status: 401 });
	}
	return user;
}