import { createHandlers } from "./utils.ts";
import Argon2id from "@rabbit-company/argon2id";

export const registerHandler = createHandlers(async (c) => {
	try {
		const client = c.get("dbCred");
		const body = await c.req.json();
		const username = body.username;
		const password = body.password;
		const hash = await Argon2id.hashEncoded(password);
		const query = `
            INSERT INTO users (username, password) VALUES ($1, $2)
        `;
		await client.queryObject(query, [username, hash]);
		return c.json({
			success: true,
			message: "Registered",
		});
	} catch (e) {
        if (e instanceof SyntaxError) {
            return c.json({ error: "Invalid JSON" }, 400);
        }
		else {
            return c.json({ error: (e as Error).message }, 500);
        }
	}
});
