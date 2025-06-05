import { cookies } from "next/headers";
import { getUserBySession } from "@/lib/db/user";
import type { UserResponse } from "@cvsa/backend";

export async function getCurrentUser(): Promise<UserResponse | null> {
	const cookieStore = await cookies();
	const sessionID = cookieStore.get("session_id");

	if (!sessionID) return null;

	try {
		const user = await getUserBySession(sessionID.value);
		return user ?? null;
	} catch (error) {
		return null;
	}
}
