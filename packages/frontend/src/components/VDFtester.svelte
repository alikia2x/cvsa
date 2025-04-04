<script lang="ts">
    import { N_ARRAY } from "src/const"; // 假设你的常量文件现在导出 N_ARRAY

    function generateRandomBigInt(min: bigint, max: bigint) {
        const range = max - min;
        const bitLength = range.toString(2).length;
        const byteLength = Math.ceil(bitLength / 8);
        const mask = (1n << BigInt(bitLength)) - 1n; // 用于截断的掩码
        let result;
        do {
            const randomBytes = new Uint8Array(byteLength);
            crypto.getRandomValues(randomBytes);
            result = 0n;
            for (let i = 0; i < byteLength; i++) {
                result = (result << 8n) | BigInt(randomBytes[i]);
            }
            result = result & mask; // 确保不超过 bitLength 位
        } while (result > range);
        return min + result;
    }

    function generateValidG(N: bigint) {
        if (N <= 4n) throw new Error("N must be > 4");
        while (true) {
            const r = generateRandomBigInt(2n, N - 1n);
            const g = (r * r) % N;
            if (g !== 1n && g !== 0n && g !== N - 1n) {
                return g;
            }
        }
    }

    const workerContent = `addEventListener("message", async (event) => {
    const { g, N, difficulty } = event.data;
    function pow(base, exponent, mod) {
        let result = 1n;
        base = base % mod;
        while (exponent > 0n) {
            if (exponent % 2n === 1n) {
                result = (result * base) % mod;
            }
            base = (base * base) % mod;
            exponent = exponent / 2n;
        }
        return result;
    }
    function computeVDFWithProgress(g, N, T, postProgress) {
        let result = g;
        let latestTime = performance.now();
        for (let i = 0n; i < T; i++) {
            result = (result * result) % N;
            if (performance.now() - latestTime > 16) {
                postProgress(Number(i * 100n) / Number(T));
                latestTime = performance.now();
            }
        }
        postProgress(100);
        return result;
    }
    const startTime = performance.now();
    const result = computeVDFWithProgress(g, N, difficulty, (progress) => {
        postMessage({ type: "progress", N: N.toString(), difficulty: difficulty.toString(), progress });
    });
    const endTime = performance.now();
    const timeTaken = endTime - startTime;
    postMessage({ type: "result", N: N.toString(), difficulty: difficulty.toString(), time: timeTaken, result });
});
`;

    let isBenchmarking = false;
    interface BenchmarkResult {
        N: bigint;
        difficulty: bigint;
        time: number;
    }
    let benchmarkResults: BenchmarkResult[] = [];
    let currentProgress = 0;
    let currentN: bigint | null = null;
    let currentDifficulty: bigint | null = null;
    let worker: Worker | null = null;
    let currentTestIndex = 0;
    const difficulties = [BigInt(20000), BigInt(200000)];
    const testCombinations: { N: bigint; difficulty: bigint }[] = [];

    // 创建需要测试的 N 和难度的组合
    N_ARRAY.forEach((n) => {
        difficulties.forEach((difficulty) => {
            testCombinations.push({ N: n, difficulty });
        });
    });

    async function startBenchmark() {
        if (testCombinations.length === 0) {
            alert("No N values provided in src/const N_ARRAY.");
            return;
        }
        isBenchmarking = true;
        benchmarkResults = [];
        currentTestIndex = 0;

        const { N, difficulty } = testCombinations[currentTestIndex];
        const g = generateValidG(N);

        let blob = new Blob([workerContent], { type: "text/javascript" });
        worker = new Worker(window.URL.createObjectURL(blob));

        worker.onmessage = (event) => {
            const { type, N: resultNStr, difficulty: resultDifficultyStr, time, progress } = event.data;
            const resultN = BigInt(resultNStr);
            const resultDifficulty = BigInt(resultDifficultyStr);

            if (type === "progress") {
                currentProgress = progress;
                currentN = resultN;
                currentDifficulty = resultDifficulty;
            } else if (type === "result") {
                benchmarkResults = [...benchmarkResults, { N: resultN, difficulty: resultDifficulty, time }];
                currentProgress = 0;
                currentTestIndex++;

                if (currentTestIndex < testCombinations.length) {
                    // 继续下一个测试组合
                    const nextTest = testCombinations[currentTestIndex];
                    const nextG = generateValidG(nextTest.N);
                    worker?.postMessage({ g: nextG, N: nextTest.N, difficulty: nextTest.difficulty });
                } else {
                    // 所有测试完毕
                    isBenchmarking = false;
                    worker?.terminate();
                    worker = null;
                    currentN = null;
                    currentDifficulty = null;
                }
            }
        };

        // 开始第一个测试
        worker.postMessage({ g, N, difficulty });
    }

    function getAccumulatedTime() {
        return benchmarkResults.reduce((acc, result) => acc + result.time, 0);
    }

    function getAccumulatedDifficulty() {
        return benchmarkResults.reduce((acc, result) => acc + Number(result.difficulty), 0);
    }

    function getSpeed() {
        return (getAccumulatedDifficulty() / getAccumulatedTime()) * 1000;
    }
</script>

<div
    class="md:bg-zinc-50 md:dark:bg-zinc-800 p-6 rounded-md md:border dark:border-zinc-700 mb-6 mt-8 md:w-2/3 lg:w-1/2 xl:w-[37%] md:mx-auto"
>
    <h2 class="text-xl font-bold mb-4 text-zinc-800 dark:text-zinc-200">VDF Benchmark</h2>

    {#if !isBenchmarking}
        <button
            class="bg-blue-500 hover:bg-blue-600 duration-100 text-white font-bold py-2 px-4 rounded"
            on:click={startBenchmark}
        >
            Start Benchmark
        </button>
    {/if}

    {#if isBenchmarking}
        <p class="mb-8 text-zinc-700 dark:text-zinc-300">
            Benchmarking in progress... ({currentTestIndex + 1}/{testCombinations.length})
        </p>
        {#if currentN !== null && currentDifficulty !== null}
            <p class="mb-2 text-zinc-700 dark:text-zinc-300">N Bits: {currentN.toString(2).length}</p>
            <p class="mb-2 text-zinc-700 dark:text-zinc-300">Difficulty: {currentDifficulty}</p>
            <div class="w-full bg-zinc-300 dark:bg-neutral-700 rounded-full h-1 relative overflow-hidden">
                <div
                    class="bg-black dark:bg-white h-full rounded-full relative"
                    style="width: {currentProgress}%"
                ></div>
            </div>
        {/if}
    {/if}

    {#if benchmarkResults.length > 0 && !isBenchmarking}
        <h3 class="text-lg font-bold mt-4 mb-2 text-zinc-800 dark:text-zinc-200">Benchmark Results</h3>
        <p class="mb-4 text-zinc-700 dark:text-zinc-300 text-sm">
            <b>Summary:</b>
            {getAccumulatedDifficulty()}
            calculations done in {getAccumulatedTime().toFixed(1)}ms,
            speed: {getSpeed().toFixed(2)} op/s
        </p>
        <table class="w-full text-sm text-left rtl:text-right text-zinc-500 dark:text-zinc-400">
            <thead
                class="text-xs text-zinc-700 uppercase dark:text-zinc-400 border-b border-zinc-400 dark:border-zinc-500"
            >
                <tr>
                    <th scope="col" class="px-6 py-3">Time (ms)</th>
                    <th scope="col" class="px-6 py-3">N (bits)</th>
                    <th scope="col" class="px-6 py-3">T (log10)</th>
                </tr>
            </thead>
            <tbody>
                {#each benchmarkResults as result}
                    <tr class="border-b dark:border-zinc-700 border-zinc-200">
                        <td class="px-6 py-4 font-medium text-zinc-900 whitespace-nowrap dark:text-white"
                            >{result.time.toFixed(2)}</td
                        >
                        <td class="px-6 py-4 font-medium text-zinc-900 whitespace-nowrap dark:text-white"
                            >{result.N.toString(2).length}</td
                        >
                        <td class="px-6 py-4 font-medium text-zinc-900 whitespace-nowrap dark:text-white"
                            >{Math.log10(Number(result.difficulty)).toFixed(2)}</td
                        >
                    </tr>
                {/each}
            </tbody>
        </table>
    {/if}
</div>
