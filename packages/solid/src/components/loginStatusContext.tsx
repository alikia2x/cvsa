import { dbCred } from "~db/index";
import { loginSessions, users } from "~db/cred/schema";
import { and, eq, gt, isNull, sql } from "drizzle-orm";
import { SensitiveUserFields, UserType } from "~db/outerSchema";

type ReturnedUser = Omit<UserType, SensitiveUserFields>;

export const getUserLoggedin = async (sessionID?: string): Promise<ReturnedUser | null> => {
	if (!sessionID) {
		return null;
	}
	const session = await dbCred
		.select({
			uid: loginSessions.uid
		})
		.from(loginSessions)
		.where(
			and(
				eq(loginSessions.id, sessionID),
				gt(loginSessions.expireAt, sql`now()`),
				isNull(loginSessions.deactivatedAt)
			)
		)
		.limit(1);
	if (session.length === 0) {
		return null;
	}

	const uid = session[0].uid;
	const user: ReturnedUser[] = await dbCred
		.select({
			id: users.id,
			username: users.username,
			nickname: users.nickname,
			role: users.role,
			createdAt: users.createdAt
		})
		.from(users)
		.where(eq(users.id, uid))
		.limit(1);

	if (user.length === 0) {
		return null;
	}

	console.log("Query for sessionID:", sessionID);
	return user[0];
};
