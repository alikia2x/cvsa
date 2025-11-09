"use client";

import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts";
import { useDarkMode } from "usehooks-ts";
import { type ChartConfig, ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";

const chartConfigLight = {
	views: {
		label: "播放",
		color: "#111417",
	},
	likes: {
		label: "点赞",
	},
} satisfies ChartConfig;

const chartConfigDark = {
	views: {
		label: "播放",
		color: "#EEEEF0",
	},
	likes: {
		label: "点赞",
	},
} satisfies ChartConfig;

interface ChartData {
	createdAt: string;
	views: number;
}

function formatDate(dateStr: string, showYear = false): string {
	const date = new Date(dateStr);
	const year = date.getFullYear();
	const month = String(date.getMonth() + 1).padStart(2, "0");
	const day = String(date.getDate()).padStart(2, "0");
	const hours = String(date.getHours()).padStart(2, "0");
	const minutes = String(date.getMinutes()).padStart(2, "0");
	const yearStr = showYear ? ` ${year}-` : "";
	return `${yearStr}${month}-${day} ${hours}:${minutes}`;
}

const formatYAxisLabel = (value: number, minMax: number) => {
	if (minMax >= 40000) {
		return (value / 10000).toFixed() + " 万";
	}
	return value.toLocaleString();
}

export function ViewsChart({ chartData }: { chartData: ChartData[] }) {
	const { isDarkMode } = useDarkMode();
	const minMax = chartData[chartData.length - 1].views - chartData[0].views;
	if (!chartData || chartData.length === 0) return <></>;
	return (
		<ChartContainer config={isDarkMode ? chartConfigDark : chartConfigLight} className="min-h-[200px] w-full">
			<LineChart accessibilityLayer data={chartData}>
				<CartesianGrid vertical={false} />
				<XAxis
					dataKey="createdAt"
					tickLine={false}
					tickMargin={10}
					axisLine={true}
					tickFormatter={(e) => formatDate(e)}
					minTickGap={30}
					className="stat-num"
				/>
				<YAxis
					dataKey="views"
					tickLine={false}
					tickMargin={5}
					domain={["auto", "auto"]}
					className="stat-num"
					tickFormatter={(value) => formatYAxisLabel(value, minMax)}
					allowDecimals={false}
				/>
				<ChartTooltip
					content={<ChartTooltipContent hideIndicator={true} labelFormatter={(e) => formatDate(e, true)} />}
				/>
				<Line dataKey="views" stroke="var(--color-views)" strokeWidth={2} dot={false} />
				<Line dataKey="likes" stroke="var(--color-likes)" strokeWidth={2} dot={false} />
			</LineChart>
		</ChartContainer>
	);
}
