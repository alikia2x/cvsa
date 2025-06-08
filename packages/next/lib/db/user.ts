import { UserType, sqlCred } from "@cvsa/core";
import { UserProfile } from "../userAuth";

export const getUserBySession = async (sessionID: string) => {
	const users = await sqlCred<UserType[]>`
        SELECT user_id as id, username, nickname, "role", user_created_at as created_at
        FROM get_user_by_session_func(${sessionID});
    `;

	if (users.length === 0) {
		return undefined;
	}

	const user = users[0];
	return {
		uid: user.id,
		username: user.username,
		nickname: user.nickname,
		role: user.role,
		createdAt: user.created_at
	};
};

export const queryUserProfile = async (uid: number, sessionID?: string): Promise<UserProfile | null> => {
	interface Result extends UserType {
		logged_in: boolean;
	}
	const users = await sqlCred<Result[]>`
        SELECT
            u.id, u.username, u.nickname, u."role", u.created_at,
            CASE
                WHEN (ls.uid IS NOT NULL AND ls.deactivated_at IS NULL AND ls.expire_at > NOW()) THEN TRUE
                ELSE FALSE
            END AS logged_in
        FROM
            users u
        LEFT JOIN
            login_sessions ls ON ls.uid = u.id AND ls.id = ${sessionID || ""}
        WHERE
            u.id = ${uid};
    `;

	if (users.length === 0) {
		return null;
	}

	const user = users[0];
	return {
		uid: user.id,
		username: user.username,
		nickname: user.nickname,
		role: user.role,
		createdAt: user.created_at,
		isLoggedIn: user.logged_in
	};
};
