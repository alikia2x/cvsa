#!/usr/bin/env bun
/**
 * PM2 wrapper to handle Bun's async module loading
 * Bypasses require-in-the-middle issues with TypeScript files
 */
// When PM2's require-in-the-middle tries to hook into module loading,
// it fails with async modules. This wrapper uses import() which works correctly.
import("./jobAdder.ts").catch((error) => {
	console.error("Failed to start server:", error);
	process.exit(1);
});
