import { cookies } from "next/headers";
import { getUserBySession } from "@/lib/db/user";

export interface User {
	uid: number;
	username: string;
	nickname: string | null;
	role: string;
	createdAt: string;
}

export async function getCurrentUser(): Promise<User | null> {
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
