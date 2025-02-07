import { assertEquals } from "jsr:@std/assert";
import { getVideoTags } from "lib/net/getVideoTags.ts";

Deno.test("Get video tags - regular video", async () => {
	const tags = (await getVideoTags(826597951)).sort();
	assertEquals(tags, [
		"纯白P",
		"中华墨水娘",
		"中华少女",
		"中华粘土娘",
		"中华缘木娘",
		"中华少女Project",
		"提糯Tino",
		"中华烛火娘",
		"中华烁金娘",
		"新世代音乐人计划女生季",
	].sort());
});

Deno.test("Get video tags - non-existent video", async () => {
	const tags = (await getVideoTags(8265979511111111));
	assertEquals(tags, []);
});

Deno.test("Get video tags - video with no tag", async () => {
	const tags = (await getVideoTags(981001865));
	assertEquals(tags, []);
});