"use client";

import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts";
import { useDarkMode } from "usehooks-ts";
import { formatDateTime } from "@/components/SearchResults";
import {
	type ChartConfig,
	ChartContainer,
	ChartTooltip,
	ChartTooltipContent,
} from "@/components/ui/chart";

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

const formatDate = (dateStr: string, showYear = false) =>
	formatDateTime(new Date(dateStr), showYear);

const formatYAxisLabel = (value: number) => {
	if (value >= 1000000) {
		return `${(value / 10000).toPrecision(4)}万`;
	} else if (value >= 10000) {
		return `${(value / 10000).toPrecision(3)}万`;
	}
	return value.toLocaleString();
};

export function ViewsChart({ chartData }: { chartData: ChartData[] }) {
	const { isDarkMode } = useDarkMode();
	if (!chartData || chartData.length === 0) return;
	return (
		<ChartContainer
			config={isDarkMode ? chartConfigDark : chartConfigLight}
			className="min-h-[200px] w-full"
		>
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
					tickMargin={0}
					domain={["auto", "auto"]}
					className="stat-num"
					tickFormatter={formatYAxisLabel}
					allowDecimals={false}
				/>
				<ChartTooltip
					content={
						<ChartTooltipContent
							hideIndicator={true}
							labelFormatter={(e) => formatDate(e, true)}
						/>
					}
				/>
				<Line
					dataKey="views"
					stroke="var(--color-views)"
					strokeWidth={2}
					dot={false}
					animationDuration={300}
				/>
				<Line
					dataKey="likes"
					stroke="var(--color-likes)"
					strokeWidth={2}
					dot={false}
					animationDuration={300}
				/>
			</LineChart>
		</ChartContainer>
	);
}
