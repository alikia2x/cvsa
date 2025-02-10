import { assertEquals } from "jsr:@std/assert";
import { SlidingWindow } from "lib/mq/slidingWindow.ts";
import { Redis } from "ioredis";

Deno.test("SlidingWindow - event and count", async () => {
	const redis = new Redis({ maxRetriesPerRequest: null });
	const windowSize = 5000; // 5 seconds
	const slidingWindow = new SlidingWindow(redis, windowSize);
	const eventName = "test_event";
    slidingWindow.clear(eventName);

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
    slidingWindow.clear(eventName);

	await slidingWindow.event(eventName);
	await slidingWindow.event(eventName);
	await slidingWindow.event(eventName);
	const count = await slidingWindow.count(eventName);

	assertEquals(count, 3);
    redis.quit();
});

Deno.test("SlidingWindow - events outside window", async () => {
	const redis = new Redis({ maxRetriesPerRequest: null });
	const windowSize = 5000; // 5 seconds
	const slidingWindow = new SlidingWindow(redis, windowSize);
	const eventName = "test_event";
    slidingWindow.clear(eventName);

	const now = Date.now();
	await redis.zadd(`cvsa:sliding_window:${eventName}`, now - windowSize - 1000, now - windowSize - 1000); // Event outside the window
	await slidingWindow.event(eventName); // Event inside the window

	const count = await slidingWindow.count(eventName);

	assertEquals(count, 1);
    redis.quit();
});

Deno.test("SlidingWindow - no events", async () => {
	const redis = new Redis({ maxRetriesPerRequest: null });
	const windowSize = 5000; // 5 seconds
	const slidingWindow = new SlidingWindow(redis, windowSize);
	const eventName = "test_event";
    slidingWindow.clear(eventName);

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
    slidingWindow.clear(eventName1);
    slidingWindow.clear(eventName2);

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
    slidingWindow.clear(eventName);
	const numEvents = 1000;

	for (let i = 0; i < numEvents; i++) {
		await slidingWindow.event(eventName);
	}

	const count = await slidingWindow.count(eventName);

	assertEquals(count, numEvents);
    redis.quit();
});
