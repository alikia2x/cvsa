import { TimeSeriesChart } from "@/components/Chart";
import { useEffect, useState } from "react";
import useSWR from "swr";

const API_URL = "https://api.projectcvsa.com";

const App = () => {
	const { data, error, isLoading } = useSWR(`${API_URL}/video/av285205499/snapshots`, async (url) => {
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
				timestamp: data[i].created_at,
				value: data[i].views
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
			style={{
				padding: "20px",
				maxWidth: "500px",
				margin: "0 auto",
				fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
			}}
		>
			<h2 className="mb-4">健康数据趋势</h2>

			<TimeSeriesChart
				data={chartData}
				height={280}
				accentColor="#007AFF"
				smoothInterpolation={true}
				timeRange="30d"
			/>
		</div>
	);
};

export default App;
