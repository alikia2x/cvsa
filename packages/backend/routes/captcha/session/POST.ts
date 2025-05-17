import { createHandlers } from "src/utils.ts";

const DIFFICULTY = 200000;

const createNewChallenge = async (difficulty: number) => {
    const baseURL = process.env["UCAPTCHA_URL"];
    const url = new URL(baseURL);
    url.pathname = "/challenge";
    return await fetch(url.toString(), {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            difficulty: difficulty,
        })
    });
}

export const createCaptchaSessionHandler = createHandlers(async (_c) => {
    const res = await createNewChallenge(DIFFICULTY);
    return res;
});
