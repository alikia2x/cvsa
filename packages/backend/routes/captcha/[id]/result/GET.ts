import { Context } from "hono";
import { Bindings, BlankEnv, BlankInput } from "hono/types";
import { ErrorResponse } from "src/schema";
import { createHandlers } from "src/utils.ts";

const getChallengeVerificationResult = async (id: string, ans: string) => {
	const baseURL = process.env["UCAPTCHA_URL"];
	const url = new URL(baseURL);
	url.pathname = `/challenge/${id}/validation`;
	const res = await fetch(url.toString(), {
		method: "POST",
		headers: {
			"Content-Type": "application/json"
		},
		body: JSON.stringify({
			y: ans
		})
	});
	return res;
};


export const verifyChallengeHandler = createHandlers(
	async (c: Context<BlankEnv & { Bindings: Bindings }, "/captcha/:id/result", BlankInput>) => {
		const id = c.req.param("id");
		const ans = c.req.query("ans");
        if (!ans) {
            const response: ErrorResponse<string> = {
                message: "Missing required query parameter: ans",
                code: "INVALID_QUERY_PARAMS"
            };
            return c.json<ErrorResponse<string>>(response, 400);
        }
        const res = await getChallengeVerificationResult(id, ans);
		return res;
	}
);
