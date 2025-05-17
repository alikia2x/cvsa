import { ErrorResponse } from "src/schema";

export const getJWTsecret = () => {
	const secret = process.env["JWT_SECRET"];
	if (!secret) {
		const response: ErrorResponse = {
			message: "JWT_SECRET is not set",
			code: "SERVER_ERROR"
		};
		return [response, true];
	}
	return [secret, null];
}