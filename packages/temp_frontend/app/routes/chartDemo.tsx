import { TimeSeriesChart, type TimeRange } from "@/components/Chart";
import React, { useEffect, useState } from "react";
import useSWR from "swr";
import useDarkTheme from "@alikia/dark-theme-hook";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

const API_URL = "https://api.projectcvsa.com";

const App = () => {
	const isDarkMode = useDarkTheme();
	const [range, setRange] = useState<TimeRange>("7d");
	const [currentData, setCurrentData] = useState("");
	const [currentDate, setCurrentDate] = useState("");
	const [outside, setOutside] = useState(false);
	const ref = React.useRef<HTMLDivElement>(null);
	const { data, error, isLoading } = useSWR(`${API_URL}/video/av285205499/snapshots?ps=1300`, async (url) => {
		const response = await fetch(url);
		if (!response.ok) {
			throw new Error("Failed to fetch song info");
		}
		return response.json();
	});

	function generateSampleData() {
		if (!data || data.length === 0) return [];
		const d = [];
		for (let i = data.length - 1; i >= 0; i--) {
			d.push({
				timestamp: new Date(data[i].created_at).getTime(),
				value: data[i].views,
			});
		}
		return d;
	}

	const [chartData, setChartData] = useState(generateSampleData());

	useEffect(() => {
		if (data) {
			setChartData(generateSampleData());
		}
	}, [data]);

	return (
		<div
			className="p-3 pt-10 mx-auto sm:max-w-lg md:max-w-xl lg:max-w-2xl xl:max-w-3xl min-h-screen"
			onPointerDown={(e: React.PointerEvent) => {
				const targetElement = e.target;
				if (ref.current && ref.current.contains(targetElement as Node)) return;
				if (e.pointerType !== "touch") return;
				setOutside(true);
				setTimeout(() => {
					setOutside(false);
				}, 100);
			}}
		>
			<div>
				<div className="w-full flex justify-between items-end">
					<span className="text-xl md:text-2xl font-bold [font-feature-settings:'tnum']">{currentData}</span>
					<span className="text-base text-neutral-700 dark:text-neutral-400 md:text-2xl font-bold [font-feature-settings:'tnum']">
						{currentDate}
					</span>
				</div>
				<TimeSeriesChart
					className="h-70 lg:h-100"
					data={chartData}
					width="100%"
					accentColor={isDarkMode ? "#ecedef" : "#141517"}
					smoothInterpolation={true}
					timeRange={range}
					outside={outside}
					ref={ref}
					setCurrentData={setCurrentData}
					setCurrentDate={setCurrentDate}
				/>
			</div>
			<Tabs defaultValue={range} onValueChange={(v) => setRange(v as TimeRange)} className="w-full mt-4">
				<TabsList>
					<TabsTrigger value="6h">6小时</TabsTrigger>
					<TabsTrigger value="1d">1天</TabsTrigger>
					<TabsTrigger value="7d">1周</TabsTrigger>
					<TabsTrigger value="30d">30天</TabsTrigger>
					<TabsTrigger value="6mo">3个月</TabsTrigger>
					<TabsTrigger value="1y">1年</TabsTrigger>
					<TabsTrigger value="all">全部</TabsTrigger>
				</TabsList>
			</Tabs>
		</div>
	);
};

export default App;
