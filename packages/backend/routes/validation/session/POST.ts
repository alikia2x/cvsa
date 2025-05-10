import { createHandlers } from "src/utils.ts";

const DIFFICULTY = 200000;

const createNewChallenge = async (difficulty: number) => {
    const baseURL = process.env["UCAPTCHA_URL"];
    const url = new URL(baseURL);
    url.pathname = "/challenge";
    const res = await fetch(url.toString(), {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            difficulty: difficulty,
        })
    });
    const data = await res.json();
    return data;
}

export const createValidationSessionHandler = createHandlers(async (c) => {
    const challenge = await createNewChallenge(DIFFICULTY);
    return c.json(challenge);
});
