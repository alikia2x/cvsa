<script lang="ts">
    import { N_ARRAY } from "src/const";
    import { fade } from "svelte/transition";

	let bigintSupported = typeof BigInt !== 'undefined';

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

    const preferredBits = 1024;
	let closetBits = 0;
	let speedSample: BenchmarkResult;

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
			if (Math.abs(Number(resultDifficultyStr) - preferredBits) < Math.abs(Number(resultDifficultyStr) - closetBits)) {
				closetBits = Number(resultDifficultyStr);
			}

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

	function getSpeed() {
		speedSample = benchmarkResults.filter((result) => result.difficulty === BigInt(closetBits)).sort((a, b) => a.time - b.time)[0];
		if (!speedSample) {
			return 0;
		}
		return Number(speedSample.difficulty) / speedSample.time * 1000;
	}
</script>

<div
    class="relative mt-8 md:mt-20 md:bg-surface-container-high md:dark:bg-dark-surface-container-high
    p-6 rounded-md mb-6 md:w-2/3 lg:w-1/2 xl:w-[37%] md:mx-auto"
>
    <h2 class="text-xl font-[500] mb-4">VDF 基准测试</h2>

	{#if !bigintSupported}
		<p class="text-error dark:text-dark-error">
			⚠️ 您的浏览器不支持 BigInt，无法运行基准测试。
		</p>
	{:else if !isBenchmarking}
		<button
				class="bg-primary dark:bg-dark-primary duration-100 text-on-primary dark:text-dark-on-primary
                font-medium py-2 px-4 rounded hover:brightness-90"
				on:click={startBenchmark}
				disabled={!bigintSupported}
		>
			开始测试
		</button>
	{/if}

    {#if isBenchmarking}
        <p class="mb-8">
            正在测试: {currentTestIndex + 1}/{testCombinations.length}
        </p>
        {#if currentN !== null && currentDifficulty !== null}
            <p class="mb-2">密钥长度: {currentN.toString(2).length} 比特</p>
            <p class="mb-2">难度: {currentDifficulty.toLocaleString()}</p>
            <div class="w-full rounded-full h-1 relative overflow-hidden">
                <div
                    class="bg-primary dark:bg-dark-primary h-full rounded-full absolute"
                    style="width: {currentProgress}%"
                ></div>
                <div
                    class="bg-secondary-container dark:bg-dark-secondary-container h-full rounded-full absolute right-0"
                    style="width: calc({100 - currentProgress}% - 0.25rem)"
                ></div>
                <div class="bg-primary dark:bg-dark-primary h-full w-1 rounded-full absolute right-0"></div>
            </div>
        {/if}
    {/if}

    {#if benchmarkResults.length > 0 && !isBenchmarking}
        <h3 class="text-lg font-medium mt-4 mb-2">测试结果</h3>
        <p class="mb-4 text-sm">
            测试在 {(getAccumulatedTime() / 1000).toFixed(3)} 秒内完成. <br/>
			速度: {Math.round(getSpeed()).toLocaleString()} 迭代 / 秒. <br/>
			<span class="text-sm text-on-surface-variant dark:text-dark-on-surface-variant">
				速度是在 N = {preferredBits} bits, T = {speedSample.difficulty} 的测试中测量的.
			</span>
        </p>
        <table class="w-full text-sm text-left rtl:text-right mt-4">
            <thead class="text-sm uppercase font-medium border-b border-outline dark:border-dark-outline">
                <tr>
                    <th scope="col" class="px-6 py-3">耗时 (ms)</th>
                    <th scope="col" class="px-6 py-3">N (bits)</th>
                    <th scope="col" class="px-6 py-3">T (迭代)</th>
                </tr>
            </thead>
            <tbody>
                {#each benchmarkResults as result}
                    <tr class="border-b border-outline-variant dark:border-dark-outline-variant">
                        <td class="px-6 py-4 whitespace-nowrap">
                            {result.time.toFixed(2)}
                        </td>
                        <td class="px-6 py-4 whitespace-nowrap">
                            {result.N.toString(2).length}
                        </td>
                        <td class="px-6 py-4 whitespace-nowrap">
                            {Number(result.difficulty)}
                        </td>
                    </tr>
                {/each}
            </tbody>
        </table>
    {/if}
</div>

{#if !isBenchmarking}
    <div
        class={"md:w-2/3 lg:w-1/2 xl:w-[37%] md:mx-auto mx-6 mb-12 " +
            (benchmarkResults.length > 0 && !isBenchmarking ? "" : "absolute left-1/2 -translate-x-1/2 top-72")}
        transition:fade={{ duration: 200 }}
    >
        <h2 class="text-lg font-medium">关于本页</h2>
        <div class="text-sm text-on-surface-variant dark:text-dark-on-surface-variant">
            <p>
                这是一个性能测试页面，<br />
                旨在测试我们的一个 VDF (Verifiable Delayed Function, 可验证延迟函数) 实现的性能。<br />
                这是一个数学函数，它驱动了整个网站的验证码（CAPTCHA）。<br />
                通过使用该函数，我们可以让您无需通过点选图片或滑动滑块既可完成验证， 同时防御我们的网站，使其免受自动程序的攻击。
                <br />
            </p>
            <p>
                点击 <i>Start Benchmark</i> 按钮，会自动测试并展示结果。<br />
            </p>
            <p>
                你可以将结果发送至邮箱: <a href="mailto:contact@alikia2x.com">contact@alikia2x.com</a>
                或 QQ：<a href="https://qm.qq.com/q/WS8zyhlcEU">1559913735</a>，并附上自己的设备信息
                （例如，手机型号、电脑的 CPU 型号等）。<br />
                我们会根据测试结果，优化我们的实现，使性能更优。<br />
                感谢你的支持！<br />
            </p>
        </div>
    </div>
{/if}

<style lang="postcss">
    @reference "tailwindcss";
    p {
        @apply my-2;
    }
</style>
