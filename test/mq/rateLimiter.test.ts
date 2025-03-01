import { assertEquals } from "jsr:@std/assert";
import { SlidingWindow } from "lib/mq/slidingWindow.ts";
import { RateLimiter, RateLimiterConfig } from "lib/mq/rateLimiter.ts";
import { Redis } from "npm:ioredis@5.5.0";

Deno.test("RateLimiter works correctly", async () => {
	const redis = new Redis({ maxRetriesPerRequest: null });
	const windowSize = 5;
	const maxRequests = 10;

	const slidingWindow = new SlidingWindow(redis, windowSize);
	const config: RateLimiterConfig = {
		window: slidingWindow,
		max: maxRequests,
	};
	const rateLimiter = new RateLimiter("test_event", [config]);
	await rateLimiter.clear();

	// Initial availability should be true
	assertEquals(await rateLimiter.getAvailability(), true);

	// Trigger events up to the limit
	for (let i = 0; i < maxRequests; i++) {
		await rateLimiter.trigger();
	}

	// Availability should now be false
	assertEquals(await rateLimiter.getAvailability(), false);

	// Wait for the window to slide
	await new Promise((resolve) => setTimeout(resolve, windowSize * 1000 + 500));

	// Availability should be true again
	assertEquals(await rateLimiter.getAvailability(), true);

	redis.quit();
});

Deno.test("Multiple configs work correctly", async () => {
	const redis = new Redis({ maxRetriesPerRequest: null });
	const windowSize1 = 1;
	const maxRequests1 = 2;
	const windowSize2 = 5;
	const maxRequests2 = 6;

	const slidingWindow1 = new SlidingWindow(redis, windowSize1);
	const config1: RateLimiterConfig = {
		window: slidingWindow1,
		max: maxRequests1,
	};
	const slidingWindow2 = new SlidingWindow(redis, windowSize2);
	const config2: RateLimiterConfig = {
		window: slidingWindow2,
		max: maxRequests2,
	};
	const rateLimiter = new RateLimiter("test_event_multi", [config1, config2]);
	await rateLimiter.clear();

	// Initial availability should be true
	assertEquals(await rateLimiter.getAvailability(), true);

	// Trigger events up to the limit of the first config
	for (let i = 0; i < maxRequests1; i++) {
		await rateLimiter.trigger();
	}

	// Availability should now be false (due to config1)
	assertEquals(await rateLimiter.getAvailability(), false);

	// Wait for the first window to slide
	await new Promise((resolve) => setTimeout(resolve, windowSize1 * 1000 + 500));

	// Availability should now be true (due to config1)
	assertEquals(await rateLimiter.getAvailability(), true);

	// Trigger events up to the limit of the second config
	for (let i = maxRequests1; i < maxRequests2; i++) {
		await rateLimiter.trigger();
	}

	// Availability should still be false (due to config2)
	assertEquals(await rateLimiter.getAvailability(), false);

	// Wait for the second window to slide
	await new Promise((resolve) => setTimeout(resolve, windowSize2 * 1000 + 500));

	// Availability should be true again
	assertEquals(await rateLimiter.getAvailability(), true);

	redis.quit();
});
