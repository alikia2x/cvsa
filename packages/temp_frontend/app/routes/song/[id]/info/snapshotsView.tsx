import { HOUR } from "@core/lib";
import { memo, useMemo, useState } from "react";
import { When } from "react-if";
import { formatDateTime } from "@/components/SearchResults";
import { Button } from "@/components/ui/button";
import {
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { columns, type Snapshot } from "@/routes/song/[id]/info/columns";
import { DataTable } from "@/routes/song/[id]/info/data-table";
import {
	addHoursToNow,
	type EtaInfo,
	formatHours,
	type Snapshots,
} from "@/routes/song/[id]/info/index";
import { detectMilestoneAchievements, processSnapshots } from "@/routes/song/[id]/info/lib";
import { ViewsChart } from "@/routes/song/[id]/info/views-chart";

const StatsTable = ({ snapshots }: { snapshots: Snapshots | null }) => {
	if (!snapshots || snapshots.length === 0) {
		return null;
	}

	const tableData: Snapshot[] = snapshots.map((snapshot) => ({
		createdAt: snapshot.createdAt,
		views: snapshot.views,
		likes: snapshot.likes || 0,
		favorites: snapshot.favorites || 0,
		coins: snapshot.coins || 0,
		danmakus: snapshot.danmakus || 0,
		shares: snapshot.shares || 0,
	}));

	return <DataTable columns={columns} data={tableData} />;
};

const getMileStoneName = (views: number) => {
	if (views < 100000) return "殿堂";
	if (views < 1000000) return "传说";
	return "神话";
};

export const SnapshotsView = ({
	snapshots,
	etaData,
	publishedAt,
}: {
	snapshots: Snapshots | null;
	etaData: EtaInfo | null;
	publishedAt: string;
}) => {
	const [timeRange, setTimeRange] = useState<string>("7d");
	const [timeOffsetHours, setTimeOffsetHours] = useState(0);

	// Calculate time range in hours
	const timeRangeHours = useMemo(() => {
		switch (timeRange) {
			case "6h":
				return 6;
			case "24h":
				return 24;
			case "7d":
				return 7 * 24;
			case "14d":
				return 14 * 24;
			case "30d":
				return 30 * 24;
			case "90d":
				return 90 * 24;
			case "365d":
				return 365 * 24;
			default:
				return undefined; // "all"
		}
	}, [timeRange]);

	const sortedSnapshots = useMemo(() => {
		if (!snapshots) return null;
		return [
			{
				id: 0,
				createdAt: publishedAt,
				views: 0,
				coins: 0,
				likes: 0,
				favorites: 0,
				shares: 0,
				danmakus: 0,
				aid: 0,
				replies: 0,
			},
			...snapshots,
		]
			.sort((a, b) => new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime())
			.map((s) => ({
				...s,
				timestamp: new Date(s.createdAt).getTime(),
			}));
	}, [snapshots, publishedAt]);

	const canGoBack = useMemo(() => {
		if (!sortedSnapshots || !timeRangeHours || timeRangeHours <= 0) return false;
		const oldestTimestamp = sortedSnapshots[0].timestamp;
		const newestTimestamp = sortedSnapshots[sortedSnapshots.length - 1].timestamp;
		const timeDiff = newestTimestamp - oldestTimestamp;
		return timeOffsetHours * HOUR + timeRangeHours * HOUR < timeDiff;
	}, [timeRangeHours, timeOffsetHours, sortedSnapshots]);

	const canGoForward = useMemo(() => {
		return !(
			!sortedSnapshots ||
			!timeRangeHours ||
			timeRangeHours <= 0 ||
			timeOffsetHours <= 0
		);
	}, [timeRangeHours, timeOffsetHours, sortedSnapshots]);

	const processedData = useMemo(
		() => processSnapshots(sortedSnapshots, timeRangeHours, timeOffsetHours),
		[timeRangeHours, timeOffsetHours, sortedSnapshots]
	);

	if (!snapshots) {
		return <Skeleton className="w-full h-50 rounded-lg mt-4" />;
	}

	if (snapshots.length === 0) {
		return (
			<div className="mt-4">
				<p>暂无数据</p>
			</div>
		);
	}

	const milestoneAchievements = detectMilestoneAchievements(snapshots, publishedAt);

	const handleBack = () => {
		if (timeRangeHours && timeRangeHours > 0) {
			setTimeOffsetHours((prev) => prev + timeRangeHours);
		}
	};

	const handleForward = () => {
		if (timeRangeHours && timeRangeHours > 0) {
			setTimeOffsetHours((prev) => Math.max(0, prev - timeRangeHours));
		}
	};

	return (
		<div className="mt-4 stat-num">
			<p>
				播放: {snapshots[0].views.toLocaleString()}
				<span className="text-secondary-foreground">
					{" "}
					更新于 {formatDateTime(new Date(snapshots[0].createdAt))}
				</span>
			</p>
			<When condition={etaData != null}>
				<When condition={etaData?.views && etaData.views <= 10000000}>
					<p>
						下一个成就：{getMileStoneName(etaData?.views || 0)}
						<span className="text-secondary-foreground">
							{" "}
							预计 {formatHours(etaData?.eta || 0)} 后（
							{addHoursToNow(etaData?.eta || 0)}
							）达成
						</span>
					</p>
				</When>

				<When condition={milestoneAchievements.length > 0}>
					<div className="mt-2">
						<p className="text-sm text-secondary-foreground">成就达成时间：</p>
						{milestoneAchievements.map((achievement) => (
							<p
								key={achievement.milestone}
								className="text-sm text-secondary-foreground ml-2"
							>
								{achievement.milestoneName}（
								{achievement.milestone.toLocaleString()}） -{" "}
								{formatDateTime(new Date(achievement.achievedAt), true, false)}
								{achievement.timeTaken && ` - 用时 ${achievement.timeTaken}`}
							</p>
						))}
					</div>
				</When>
			</When>

			<Tabs defaultValue="chart" className="mt-4">
				<div className="flex justify-between items-center mb-4">
					<h2 className="text-2xl font-medium">数据</h2>
					<TabsList>
						<TabsTrigger value="chart">图表</TabsTrigger>
						<TabsTrigger value="table">表格</TabsTrigger>
					</TabsList>
				</div>

				<TabsContent value="chart">
					<div className="flex flex-col gap-2">
						<div className="flex justify-between items-center">
							<Button
								variant="outline"
								size="sm"
								onClick={handleBack}
								disabled={!canGoBack}
							>
								上一页
							</Button>
							<Select value={timeRange} onValueChange={setTimeRange}>
								<SelectTrigger className="w-32">
									<SelectValue placeholder="时间范围" />
								</SelectTrigger>
								<SelectContent>
									<SelectItem value="6h">6小时</SelectItem>
									<SelectItem value="24h">24小时</SelectItem>
									<SelectItem value="7d">7天</SelectItem>
									<SelectItem value="14d">14天</SelectItem>
									<SelectItem value="30d">30天</SelectItem>
									<SelectItem value="90d">90天</SelectItem>
									<SelectItem value="365d">1年</SelectItem>
									<SelectItem value="all">全部</SelectItem>
								</SelectContent>
							</Select>
							<Button
								variant="outline"
								size="sm"
								onClick={handleForward}
								disabled={!canGoForward}
							>
								下一页
							</Button>
						</div>
						<ViewsChart chartData={processedData} />
					</div>
				</TabsContent>
				<TabsContent value="table">
					<StatsTable snapshots={snapshots} />
				</TabsContent>
			</Tabs>
		</div>
	);
};

export const MemoizedSnapshotsView = memo(SnapshotsView);
