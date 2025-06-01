"use client";

import { N_ARRAY } from "@/lib/const";
import { UAParser } from "ua-parser-js";
import { useEffect, useState } from "react";
import { computeVdfInWorker } from "@/lib/vdf";
import { FilledButton } from "@/components/ui/Buttons/FilledButton";

let bigintSupported = typeof BigInt !== "undefined";

function generateRandomBigInt(min: bigint, max: bigint) {
	const range = max - min;
	const bitLength = range.toString(2).length;
	const byteLength = Math.ceil(bitLength / 8);
	const mask = (1n << BigInt(bitLength)) - 1n;
	let result;
	do {
		const randomBytes = new Uint8Array(byteLength);
		crypto.getRandomValues(randomBytes);
		result = 0n;
		for (let i = 0; i < byteLength; i++) {
			result = (result << 8n) | BigInt(randomBytes[i]);
		}
		result = result & mask;
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

export const VDFtestCard = () => {
	const [browserInfo, setBrowserInfo] = useState<string | null>(null);
	const [isBenchmarking, setIsBenchmarking] = useState(false);
	const [benchmarkResults, setBenchmarkResults] = useState<{ N: bigint; difficulty: bigint; time: number }[]>([]);
	const [currentProgress, setCurrentProgress] = useState(0);
	const [currentN, setCurrentN] = useState<bigint | null>(null);
	const [currentDifficulty, setCurrentDifficulty] = useState<bigint | null>(null);
	const [currentTestIndex, setCurrentTestIndex] = useState(0);
	const difficulties = [BigInt(20000), BigInt(200000)];
	const [testCombinations, setTestCombinations] = useState<{ N: bigint; difficulty: bigint }[]>([]);
	const speedSampleIndex = 1;
	const [speedSample, setSpeedSample] = useState<{ N: bigint; difficulty: bigint; time: number } | undefined>(
		undefined
	);

	useEffect(() => {
		// 创建需要测试的 N 和难度的组合
		const combinations: { N: bigint; difficulty: bigint }[] = [];
		N_ARRAY.forEach((n) => {
			difficulties.forEach((difficulty) => {
				combinations.push({ N: n, difficulty });
			});
		});
		setTestCombinations(combinations);

		const ua = navigator ? navigator.userAgent : "";
		const { browser } = UAParser(ua);
		setBrowserInfo(browser.name + " " + browser.version);
	}, []);

	async function startBenchmark() {
		if (testCombinations.length === 0) {
			alert("No N values provided in src/const N_ARRAY.");
			return;
		}
		setIsBenchmarking(true);
		setBenchmarkResults([]);
		setCurrentTestIndex(0);

		async function runTest(index: number) {
			if (index >= testCombinations.length) {
				setIsBenchmarking(false);
				setCurrentN(null);
				setCurrentDifficulty(null);
				return;
			}

			const { N, difficulty } = testCombinations[index];
			const g = generateValidG(N);

			try {
				const { time } = await computeVdfInWorker(g, N, difficulty, (progress) => {
					setCurrentProgress(progress);
					setCurrentN(N);
					setCurrentDifficulty(difficulty);
				});
				setBenchmarkResults((prevResults) => [...prevResults, { N, difficulty, time }]);
				setCurrentProgress(0);
				setCurrentTestIndex((prevIndex) => prevIndex + 1);
				runTest(index + 1);
			} catch (error) {
				setIsBenchmarking(false);
				setCurrentN(null);
				setCurrentDifficulty(null);
			}
		}

		runTest(0);
	}

	function getAccumulatedTime() {
		return benchmarkResults.reduce((acc, result) => acc + result.time, 0);
	}

	function calculateSpeed() {
		const sample = benchmarkResults[speedSampleIndex];
		if (!sample) return 0;
		return (Number(sample.difficulty) / sample.time) * 1000;
	}
	useEffect(() => {
		if (benchmarkResults.length > speedSampleIndex) {
			setSpeedSample(benchmarkResults[speedSampleIndex]);
		}
	}, [benchmarkResults]);

	return (
		<div className="relative mt-8 mb-12 h-auto duration-300">
			<h2 className="text-2xl font-medium mb-5">VDF 基准测试</h2>

			{!bigintSupported ? (
				<p className="text-red-500 dark:text-red-400">⚠️ 您的浏览器不支持 BigInt，无法运行基准测试。</p>
			) : !isBenchmarking ? (
				<FilledButton onClick={startBenchmark} disabled={!bigintSupported} shape="square">
					开始测试
				</FilledButton>
			) : null}

			{isBenchmarking && (
				<>
					<p className="mb-8">
						正在测试: {currentTestIndex + 1}/{testCombinations.length}
					</p>
					{currentN !== null && currentDifficulty !== null && (
						<>
							<p className="mb-2">密钥长度: {currentN.toString(2).length} 比特</p>
							<p className="mb-2">难度: {currentDifficulty.toLocaleString()}</p>
							<div className="w-full rounded-full h-1 relative overflow-hidden">
								<div
									className="bg-primary dark:bg-dark-primary h-full rounded-full absolute"
									style={{ width: `${currentProgress}%` }}
								></div>
								<div
									className="bg-secondary-container dark:bg-dark-secondary-container h-full rounded-full absolute right-0"
									style={{ width: `calc(${100 - currentProgress}% - 0.25rem)` }}
								></div>
								<div className="bg-primary dark:bg-dark-primary h-full w-1 rounded-full absolute right-0"></div>
							</div>
						</>
					)}
				</>
			)}

			{benchmarkResults.length > 0 && !isBenchmarking && (
				<>
					<h3 className="text-lg font-medium mt-4 mb-2">测试结果</h3>
					<p className="mb-4 text-sm">
						测试在 {(getAccumulatedTime() / 1000).toFixed(3)} 秒内完成. <br />
						速度: {Math.round(calculateSpeed()).toLocaleString()} 迭代 / 秒. <br />
						<span className="text-sm text-on-surface-variant dark:text-dark-on-surface-variant">
							速度是在 N = {speedSample?.N.toString(2).length} bits, T = {speedSample?.difficulty}{" "}
							的测试中测量的.
						</span>
						<br />
						{browserInfo && <>浏览器版本：{browserInfo}</>}
					</p>
					<table className="w-full text-sm text-left rtl:text-right mt-4">
						<thead className="text-sm uppercase font-medium border-b border-outline dark:border-dark-outline">
							<tr>
								<th scope="col" className="px-6 py-3">
									耗时 (ms)
								</th>
								<th scope="col" className="px-6 py-3">
									N (bits)
								</th>
								<th scope="col" className="px-6 py-3">
									T (迭代)
								</th>
							</tr>
						</thead>
						<tbody>
							{benchmarkResults.map((result) => (
								<tr
									key={`${result.N}-${result.difficulty}-${result.time}`}
									className="border-b border-outline-variant dark:border-dark-outline-variant"
								>
									<td className="px-6 py-4 whitespace-nowrap">{result.time.toFixed(2)}</td>
									<td className="px-6 py-4 whitespace-nowrap">{result.N.toString(2).length}</td>
									<td className="px-6 py-4 whitespace-nowrap">{Number(result.difficulty)}</td>
								</tr>
							))}
						</tbody>
					</table>
				</>
			)}
		</div>
	);
};
