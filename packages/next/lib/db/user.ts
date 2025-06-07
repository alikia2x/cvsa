import { UserType, sqlCred } from "@cvsa/core";

export const getUserBySession = async (sessionID: string) => {
	const users = await sqlCred<UserType[]>`
	    SELECT u.*
        FROM users u
        JOIN login_sessions ls ON u.id = ls.uid
        WHERE ls.id = ${sessionID};
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

export const getUserSessions = async (sessionID: string) => {};
