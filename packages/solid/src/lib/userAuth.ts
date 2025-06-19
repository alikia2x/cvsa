import { cookies } from "next/headers";
import { getUserBySession, queryUserProfile } from "@/lib/db/user";

export interface User {
	uid: number;
	username: string;
	nickname: string | null;
	role: string;
	createdAt: Date;
}

export interface UserProfile extends User {
	isLoggedIn: boolean;
}

export async function getCurrentUser(): Promise<User | null> {
	const cookieStore = await cookies();
	const sessionID = cookieStore.get("session_id");
	if (!sessionID) return null;

	try {
		const user = await getUserBySession(sessionID.value);

		return user ?? null;
	} catch (error) {
		console.log(error);
		return null;
	}
}

export async function getUserProfile(uid: number): Promise<UserProfile | null> {
	const cookieStore = await cookies();
	const sessionID = cookieStore.get("session_id");

	try {
		const user = await queryUserProfile(uid, sessionID?.value);

		return user ?? null;
	} catch (error) {
		console.log(error);
		return null;
	}
}
