// Define interfaces for input and output
interface VdfProgressCallback {
	(progress: number): void;
}

interface VdfResult {
	result: bigint;
	time: number; // Time taken in milliseconds
}

// The content of the Web Worker script
const workerContent = `addEventListener("message", async (event) => {
    const { g, N, difficulty } = event.data;

    // Although pow is not used in the iterative VDF, it's good to keep the original worker code structure.
    // The iterative computeVDFWithProgress is better for progress reporting.
    function pow(base, exponent, mod) {
        let result = 1n;
        base = base % mod;
        while (exponent > 0n) {
            if (exponent % 2n === 1n) {
                result = (result * base) % mod;
            }
            base = (base * base) % mod;
            exponent = exponent / 2n;
            // Using BigInt division (/) which performs integer division
        }
        return result;
    }

    // Compute VDF iteratively to report progress
    function computeVDFWithProgress(g, N, T, postProgress) {
        let result = g;
        let latestTime = performance.now();
        const totalSteps = T; // T is the difficulty, representing 2^T squaring steps

        for (let i = 0n; i < totalSteps; i++) {
            result = (result * result) % N;
            // Report progress periodically (approx. every 16ms to match typical frame rate)
            if (performance.now() - latestTime > 16) {
                 // Calculate progress as a percentage
                const progress = Number((i + 1n) * 10000n / totalSteps) / 100; // Using 10000 for better precision before dividing by 100
                postProgress(progress);
                latestTime = performance.now();
            }
        }
        // Ensure final progress is reported
        postProgress(100);
        return result;
    }

    const startTime = performance.now();
    // The worker computes g^(2^difficulty) mod N. The loop runs 'difficulty' times, performing squaring.
    const result = computeVDFWithProgress(g, N, difficulty, (progress) => {
        // Post progress back to the main thread
        postMessage({ type: "progress", progress: progress });
    });
    const endTime = performance.now();
    const timeTaken = endTime - startTime;

    // Post the final result and time taken back to the main thread
    postMessage({ type: "result", result: result.toString(), time: timeTaken });
});
`;

/**
 * Computes the Verifiable Delay Function (VDF) result g^(2^difficulty) mod N
 * in a Web Worker and reports progress.
 * @param g - The base (bigint).
 * @param N - The modulus (bigint).
 * @param difficulty - The number of squaring steps (T) (bigint).
 * @param onProgress - Optional callback function to receive progress updates (0-100).
 * @returns A Promise that resolves with the VDF result and time taken.
 */
export function computeVdfInWorker(
	g: bigint,
	N: bigint,
	difficulty: bigint,
	onProgress?: VdfProgressCallback
): Promise<VdfResult> {
	return new Promise((resolve, reject) => {
		// Create a Blob containing the worker script
		const blob = new Blob([workerContent], { type: "text/javascript" });
		// Create a URL for the Blob
		const workerUrl = URL.createObjectURL(blob);
		// Create a new Web Worker
		const worker = new Worker(workerUrl);

		// Handle messages from the worker
		worker.onmessage = (event) => {
			const { type, progress, result, time } = event.data;

			if (type === "progress") {
				if (onProgress) {
					onProgress(progress);
				}
			} else if (type === "result") {
				// Resolve the promise with the result and time
				resolve({ result: BigInt(result), time });
				// Terminate the worker and revoke the URL
				worker.terminate();
				URL.revokeObjectURL(workerUrl);
			}
		};

		// Handle potential errors in the worker
		worker.onerror = (error) => {
			reject(error);
			// Terminate the worker and revoke the URL in case of error
			worker.terminate();
			URL.revokeObjectURL(workerUrl);
		};

		// Post the data to the worker to start the computation
		worker.postMessage({ g, N, difficulty });
	});
}
