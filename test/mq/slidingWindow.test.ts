import { assertEquals } from "jsr:@std/assert";
import { SlidingWindow } from "lib/mq/slidingWindow.ts";
import { Redis } from "ioredis";

Deno.test("SlidingWindow - event and count", async () => {
	const redis = new Redis({ maxRetriesPerRequest: null });
	const windowSize = 5000; // 5 seconds
	const slidingWindow = new SlidingWindow(redis, windowSize);
	const eventName = "test_event";
    await slidingWindow.clear(eventName);

	await slidingWindow.event(eventName);
	const count = await slidingWindow.count(eventName);

	assertEquals(count, 1);
    redis.quit();
});

Deno.test("SlidingWindow - multiple events", async () => {
	const redis = new Redis({ maxRetriesPerRequest: null });
	const windowSize = 5000; // 5 seconds
	const slidingWindow = new SlidingWindow(redis, windowSize);
	const eventName = "test_event";
    await slidingWindow.clear(eventName);

	await slidingWindow.event(eventName);
	await slidingWindow.event(eventName);
	await slidingWindow.event(eventName);
	const count = await slidingWindow.count(eventName);

	assertEquals(count, 3);
    redis.quit();
});

Deno.test("SlidingWindow - no events", async () => {
	const redis = new Redis({ maxRetriesPerRequest: null });
	const windowSize = 5000; // 5 seconds
	const slidingWindow = new SlidingWindow(redis, windowSize);
	const eventName = "test_event";
    await slidingWindow.clear(eventName);

	const count = await slidingWindow.count(eventName);

	assertEquals(count, 0);
    redis.quit();
});

Deno.test("SlidingWindow - different event names", async () => {
	const redis = new Redis({ maxRetriesPerRequest: null });
	const windowSize = 5000; // 5 seconds
	const slidingWindow = new SlidingWindow(redis, windowSize);
	const eventName1 = "test_event_1";
	const eventName2 = "test_event_2";
    await slidingWindow.clear(eventName1);
    await slidingWindow.clear(eventName2);

	await slidingWindow.event(eventName1);
	await slidingWindow.event(eventName2);

	const count1 = await slidingWindow.count(eventName1);
	const count2 = await slidingWindow.count(eventName2);

	assertEquals(count1, 1);
	assertEquals(count2, 1);
    redis.quit();
});

Deno.test("SlidingWindow - large number of events", async () => {
	const redis = new Redis({ maxRetriesPerRequest: null });
	const windowSize = 5000; // 5 seconds
	const slidingWindow = new SlidingWindow(redis, windowSize);
	const eventName = "test_event";
    await slidingWindow.clear(eventName);
	const numEvents = 1000;

	for (let i = 0; i < numEvents; i++) {
		await slidingWindow.event(eventName);
	}

	const count = await slidingWindow.count(eventName);

	assertEquals(count, numEvents);
    redis.quit();
});
