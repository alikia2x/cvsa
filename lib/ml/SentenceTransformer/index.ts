import { SentenceTransformer } from "./model.ts"; // Changed import path

async function main() {
	const sentenceTransformer = await SentenceTransformer.from_pretrained(
		"mixedbread-ai/mxbai-embed-large-v1",
	);
	const outputs = await sentenceTransformer.encode([
		"Hello world",
		"How are you guys doing?",
		"Today is Friday!",
	]);

	// @ts-ignore
	console.log(outputs["last_hidden_state"]);

	return outputs;
}

main(); // Keep main function call if you want this file to be runnable directly for testing.
