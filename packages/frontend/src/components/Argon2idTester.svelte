<script lang="ts">
    import { type computeHash } from "argon2id";
    import setupWasm from "argon2id/lib/setup.js";

    let totalHashes = 0;
    let argon2id: null | computeHash = null;
    let memorySize = 2560;
    let passes = 1;

    function startsWithZeros(uint8Array: Uint8Array, numZeros: number) {
        if (numZeros > uint8Array.length * 8) {
            return false; // More zeros requested than possible in the array
        }

        let zeroCount = 0;
        for (let i = 0; i < uint8Array.length; i++) {
            const byte = uint8Array[i];
            for (let j = 7; j >= 0; j--) {
                if ((byte >> j) & 1) {
                    // Found a '1' bit, so the leading zeros sequence ends
                    return zeroCount >= numZeros;
                }
                zeroCount++;
                if (zeroCount === numZeros) {
                    return true;
                }
            }
        }

        // Reached the end of the array and the required number of zeros were found
        return zeroCount >= numZeros;
    }

    async function init() {
        const SIMD_FILENAME = "/simd.wasm";
        const NON_SIMD_FILENAME = "/non-simd.wasm";

        const setupWasmWithFetch = async (
            wasmUrl: string,
            importObject: WebAssembly.Imports
        ): Promise<WebAssembly.WebAssemblyInstantiatedSource> => {
            try {
                const response = await fetch(wasmUrl);
                if (!response.ok) {
                    throw new Error(`Failed to fetch WASM module from ${wasmUrl}: ${response.status}`);
                }
                const wasmBytes = await response.arrayBuffer();
                const result = await WebAssembly.instantiate(wasmBytes, importObject);
                return result;
            } catch (error) {
                console.error(`Error initializing WASM from ${wasmUrl}:`, error);
                throw error; // Re-throw the error to be caught by the caller
            }
        };

        argon2id = await setupWasm(
            (importObject) => setupWasmWithFetch(SIMD_FILENAME, importObject),
            (importObject) => setupWasmWithFetch(NON_SIMD_FILENAME, importObject)
        );
    }

    // Simple PoW function (very basic for demonstration)
    async function proofOfWork(message: string, difficulty: number) {
        if (argon2id === null) {
            await init();
        }
        let nonce = 0;
        const h = argon2id!;
        while (true) {
            const p = new TextEncoder().encode(`${message}-${nonce}`);
            const hash = h({
                password: new Uint8Array(p),
                salt: crypto.getRandomValues(new Uint8Array(32)),
                memorySize: memorySize,
                parallelism: 1,
                passes: passes,
                tagLength: 32,
            });
            totalHashes++;
            if (startsWithZeros(hash, difficulty)) {
                return nonce;
            }
            nonce++;
            if (nonce > 100000) {
                throw new Error(
                    "Could not find a valid nonce within a reasonable number of attempts. Try a lower difficulty."
                );
            }
            await new Promise((resolve) => setTimeout(resolve, 0)); // Yield control for responsiveness
        }
    }
    const challenge = "challenge";
    let difficulty = 4;
    let tries = 10;
    let solution = "";
    let isSolving = false;
    let errorMessage = "";
    let solveTime = "";

    async function solveChallenge() {
        totalHashes = 0;
        isSolving = true;
        errorMessage = "";
        solution = "";
        solveTime = "";
        const startTime = performance.now();

        try {
            const flooredDifficulty = Math.floor(difficulty);
            const percentage = difficulty - flooredDifficulty;
            const numberOfHiggerDifficultyTries = Math.floor(tries * percentage);
            const numberOfLowerDifficultyTries = tries - numberOfHiggerDifficultyTries;
            for (let i = 0; i < numberOfLowerDifficultyTries; i++) {
                const nonce = await proofOfWork(challenge, flooredDifficulty);
                solution = `${challenge}-${nonce}`;
            }
            for (let i = 0; i < numberOfHiggerDifficultyTries; i++) {
                const nonce = await proofOfWork(challenge, flooredDifficulty + 1);
                solution = `${challenge}-${nonce}`;
            }
            const endTime = performance.now();
            solveTime = (endTime - startTime).toFixed(2);
        } catch (error) {
            console.error("Error solving challenge:", error);
            const e = error as Error;
            errorMessage = e.message || "An error occurred while solving the challenge.";
        } finally {
            isSolving = false;
        }
    }
</script>

<div class="bg-gray-100 dark:bg-gray-800 p-6 rounded-md shadow-md mb-6 mt-8 lg:w-1/2 xl:w-1/3 mx-auto">
    <h2 class="text-xl font-bold mb-4 text-gray-800 dark:text-gray-200">The Challenge</h2>

    <div class="mb-4">
        <label for="difficulty" class="block text-gray-700 dark:text-gray-300 text-sm font-bold mb-2">Memory:</label
        >
        <input
            type="number"
            id="memorySize"
            bind:value={memorySize}
            min="8"
            step="64"
            class="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 dark:text-gray-300 leading-tight focus:outline-none focus:shadow-outline dark:bg-gray-700 dark:border-gray-600"
            placeholder="Number of leading zeros"
        />
        <p class="text-gray-500 dark:text-gray-400 text-xs italic">Memory cost (KiB) in argon2id config.</p>
    </div>

    <div class="mb-4">
        <label for="difficulty" class="block text-gray-700 dark:text-gray-300 text-sm font-bold mb-2">Passes:</label
        >
        <input
            type="number"
            id="passes"
            bind:value={passes}
            min="1"
            step="1"
            class="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 dark:text-gray-300 leading-tight focus:outline-none focus:shadow-outline dark:bg-gray-700 dark:border-gray-600"
            placeholder="Number of leading zeros"
        />
        <p class="text-gray-500 dark:text-gray-400 text-xs italic">Number of passes in argon2id config.</p>
    </div>

    <div class="mb-4">
        <label for="difficulty" class="block text-gray-700 dark:text-gray-300 text-sm font-bold mb-2">Difficulty:</label
        >
        <input
            type="number"
            id="difficulty"
            bind:value={difficulty}
            min="0"
            step="1"
            class="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 dark:text-gray-300 leading-tight focus:outline-none focus:shadow-outline dark:bg-gray-700 dark:border-gray-600"
            placeholder="Number of leading zeros"
        />
        <p class="text-gray-500 dark:text-gray-400 text-xs italic">Higher difficulty requires more computation.</p>
    </div>

    <div class="mb-4">
        <label for="difficulty" class="block text-gray-700 dark:text-gray-300 text-sm font-bold mb-2">Tries:</label
        >
        <input
            type="number"
            id="tries"
            bind:value={tries}
            min="1"
            step="1"
            class="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 dark:text-gray-300 leading-tight focus:outline-none focus:shadow-outline dark:bg-gray-700 dark:border-gray-600"
            placeholder="Number of leading zeros"
        />
        <p class="text-gray-500 dark:text-gray-400 text-xs italic">The number of consecutive successes required to pass the verification.</p>
    </div>

    <button
        on:click={solveChallenge}
        disabled={isSolving}
        class="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline disabled:bg-gray-500 disabled:cursor-not-allowed"
    >
        {#if isSolving}
            Solving...
        {:else}
            Solve Challenge
        {/if}
    </button>

    {#if errorMessage}
        <p class="text-red-500 mt-2">{errorMessage}</p>
    {/if}

    <p class="mt-3 text-sm text-gray-600 dark:text-gray-400">Total hashes calculated: {totalHashes}</p>

    {#if solution}
        <div class="mt-6 p-4 bg-gray-200 dark:bg-gray-700 rounded-md">
            <h3 class="text-lg font-bold mb-2 text-gray-800 dark:text-gray-200">Solution Found!</h3>
            <p class="text-gray-700 dark:text-gray-300">
                The solution (challenge + nonce) is:
                <code class="bg-gray-300 dark:bg-gray-600 text-gray-900 dark:text-gray-100 p-1 rounded font-mono"
                    >{solution}</code
                >
            </p>
            {#if solveTime}
                <p class="text-sm text-gray-600 dark:text-gray-400 mt-1">
                    Time to solve: {solveTime} ms
                </p>
            {/if}
        </div>
    {/if}
</div>
