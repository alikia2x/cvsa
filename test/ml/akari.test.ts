import Akari from "lib/ml/akari.ts";
import { assertEquals, assertGreaterOrEqual } from "jsr:@std/assert";
import { join } from "$std/path/join.ts";
import { SECOND } from "$std/datetime/constants.ts";

Deno.test("Akari AI - normal cases accuracy test", async () => {
	const path = import.meta.dirname!;
	const dataPath = join(path, "akari.json");
	const rawData = await Deno.readTextFile(dataPath);
	const data = JSON.parse(rawData);
	await Akari.init();
	for (const testCase of data.test1) {
		const result = await Akari.classifyVideo(
			testCase.title,
			testCase.desc,
			testCase.tags,
		);
		assertEquals(result, testCase.label);
	}
});

Deno.test("Akari AI - performance test", async () => {
	const path = import.meta.dirname!;
	const dataPath = join(path, "akari.json");
	const rawData = await Deno.readTextFile(dataPath);
	const data = JSON.parse(rawData);
	await Akari.init();
	const N = 200;
	const testCase = data.test1[0];
	const title = testCase.title;
	const desc = testCase.desc;
	const tags = testCase.tags;
	const time = performance.now();
	for (let i = 0; i < N; i++) {
		await Akari.classifyVideo(
			title,
			desc,
			tags,
		);
	}
	const end = performance.now();
	const elapsed = (end - time) / SECOND;
	const throughput = N / elapsed;
	assertGreaterOrEqual(throughput, 100);
	console.log(`Akari AI throughput: ${throughput.toFixed(1)} samples / sec`);
});
