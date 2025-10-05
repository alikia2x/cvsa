export function generateRandomId(length: number): string {
	const characters = "abcdefghijkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789";
	const charactersLength = characters.length;
	const randomBytes = new Uint8Array(length);

	crypto.getRandomValues(randomBytes);

	let result = "";
	for (let i = 0; i < length; i++) {
		const randomIndex = randomBytes[i] % charactersLength;
		result += characters.charAt(randomIndex);
	}

	return result;
}
