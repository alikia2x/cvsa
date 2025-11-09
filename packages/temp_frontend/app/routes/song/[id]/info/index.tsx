import type { Route } from "./+types/index";
import { treaty } from "@elysiajs/eden";
import type { App } from "@elysia/src";
import { memo, useEffect, useState, useMemo } from "react";
import { Skeleton } from "@/components/ui/skeleton";
import { TriangleAlert } from "lucide-react";
import { Title } from "@/components/Title";
import { toast } from "sonner";
import { Error } from "@/components/Error";
import { Layout } from "@/components/Layout";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { formatDateTime } from "@/components/SearchResults";
import { ViewsChart } from "./views-chart";
import { processSnapshots, detectMilestoneAchievements, type MilestoneAchievement } from "./lib";
import { DataTable } from "./data-table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { If, Then } from "react-if";
import {
	AlertDialog,
	AlertDialogAction,
	AlertDialogCancel,
	AlertDialogContent,
	AlertDialogDescription,
	AlertDialogFooter,
	AlertDialogHeader,
	AlertDialogTitle,
	AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { av2bv } from "@elysia/lib/bilibiliID";
import { columns, type Snapshot } from "./columns";

// @ts-ignore idk
const app = treaty<App>(import.meta.env.VITE_API_URL!);

type SongInfo = Awaited<ReturnType<ReturnType<typeof app.song>["info"]["get"]>>["data"];
type EtaInfo = Awaited<ReturnType<ReturnType<typeof app.video>["eta"]["get"]>>["data"];
export type Snapshots = Awaited<ReturnType<ReturnType<typeof app.video>["snapshots"]["get"]>>["data"];
type SongInfoError = Awaited<ReturnType<ReturnType<typeof app.song>["info"]["get"]>>["error"];
type SnapshotsError = Awaited<ReturnType<ReturnType<typeof app.video>["snapshots"]["get"]>>["error"];
type EtaInfoError = Awaited<ReturnType<ReturnType<typeof app.video>["eta"]["get"]>>["error"];

export async function clientLoader({ params }: Route.LoaderArgs) {
	return { id: params.id };
}

function formatHours(hours: number): string {
	if (hours >= 24 * 14) return `${Math.floor(hours / 24)} 天`;
	if (hours >= 24) return `${Math.floor(hours / 24)} 天 ${hours % 24} 小时`;
	if (hours >= 1) return `${Math.floor(hours)} 时 ${Math.round((hours % 1) * 60)} 分`;
	return `${Math.round(hours * 60)} 分钟`;
}

function addHoursToNow(hours: number): string {
	const d = new Date();
	d.setHours(d.getHours() + hours);
	return `${d.getFullYear()}-${(d.getMonth() + 1).toString().padStart(2, "0")}-${d.getDate().toString().padStart(2, "0")} ${d.getHours().toString().padStart(2, "0")}:${d.getMinutes().toString().padStart(2, "0")}`;
}

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

const SnapshotsView = ({
	snapshots,
	etaData,
	publishedAt,
}: {
	snapshots: Snapshots | null;
	etaData: EtaInfo | null;
	publishedAt?: string;
}) => {
	const [timeRange, setTimeRange] = useState<string>("all");
	const [currentPage, setCurrentPage] = useState(1);
	const [pageSize] = useState(10);
	const [timeOffsetHours, setTimeOffsetHours] = useState(0);

	// Calculate time range in hours
	const timeRangeHours = useMemo(() => {
		switch (timeRange) {
			case "6h":
				return 6;
			case "12h":
				return 12;
			case "24h":
				return 24;
			case "3d":
				return 72;
			case "7d":
				return 168;
			case "14d":
				return 336;
			case "30d":
				return 720;
			default:
				return undefined; // "all"
		}
	}, [timeRange]);

	// Pagination for table data
	const paginatedSnapshots = useMemo(() => {
		if (!snapshots) return [];
		const startIndex = (currentPage - 1) * pageSize;
		const endIndex = startIndex + pageSize;
		return snapshots.slice(startIndex, endIndex);
	}, [snapshots, currentPage, pageSize]);

	const totalPages = snapshots ? Math.ceil(snapshots.length / pageSize) : 0;

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

	const processedData = processSnapshots(snapshots, timeRangeHours, timeOffsetHours);
	const milestoneAchievements = detectMilestoneAchievements(snapshots, publishedAt);

	// Handle time range navigation
	const totalDataHours =
		snapshots && snapshots.length > 0
			? (new Date(snapshots[snapshots.length - 1].createdAt).getTime() -
					new Date(snapshots[0].createdAt).getTime()) /
				(1000 * 60 * 60)
			: 0;

	// Simplified logic: always allow navigation if we have a time range
	const canGoBack = timeRangeHours !== undefined && timeRangeHours > 0;
	const canGoForward = timeRangeHours !== undefined && timeRangeHours > 0;

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
		<div className="mt-4">
			<p>
				播放: {snapshots[0].views.toLocaleString()}
				<span className="text-secondary-foreground">
					{" "}
					更新于 {formatDateTime(new Date(snapshots[0].createdAt))}
				</span>
			</p>
			{etaData && (
				<>
					{etaData!.views <= 10000000 && (
						<p>
							下一个成就：{getMileStoneName(etaData!.views)}
							<span className="text-secondary-foreground">
								{" "}
								预计 {formatHours(etaData!.eta)} 后（{addHoursToNow(etaData!.eta)}）达成
							</span>
						</p>
					)}

					{milestoneAchievements.length > 0 && (
						<div className="mt-2">
							<p className="text-sm text-secondary-foreground">成就达成时间：</p>
							{milestoneAchievements.map((achievement) => (
								<p key={achievement.milestone} className="text-sm text-secondary-foreground ml-2">
									{achievement.milestoneName}（{achievement.milestone.toLocaleString()}） -{" "}
									{formatDateTime(new Date(achievement.achievedAt))}
									{achievement.timeTaken && ` - 用时 ${achievement.timeTaken}`}
								</p>
							))}
						</div>
					)}
				</>
			)}

			<Tabs defaultValue="chart" className="mt-4">
				<div className="flex justify-between items-center mb-4">
					<h2 className="text-2xl font-medium">数据</h2>
					<div className="flex items-center gap-4">
						<TabsList>
							<TabsTrigger value="chart">图表</TabsTrigger>
							<TabsTrigger value="table">表格</TabsTrigger>
						</TabsList>
						<Select value={timeRange} onValueChange={setTimeRange}>
							<SelectTrigger className="w-32">
								<SelectValue placeholder="时间范围" />
							</SelectTrigger>
							<SelectContent>
								<SelectItem value="6h">6小时</SelectItem>
								<SelectItem value="12h">12小时</SelectItem>
								<SelectItem value="24h">24小时</SelectItem>
								<SelectItem value="3d">3天</SelectItem>
								<SelectItem value="7d">7天</SelectItem>
								<SelectItem value="14d">14天</SelectItem>
								<SelectItem value="30d">30天</SelectItem>
								<SelectItem value="all">全部</SelectItem>
							</SelectContent>
						</Select>
					</div>
				</div>

				<TabsContent value="chart">
					<div className="flex flex-col gap-2">
						<div className="flex justify-between items-center">
							<Button variant="outline" size="sm" onClick={handleBack} disabled={!canGoBack}>
								上一个时间段
							</Button>
							<span className="text-sm text-secondary-foreground">
								{timeRangeHours ? `${timeRangeHours}小时范围` : "全部数据"}
							</span>
							<Button variant="outline" size="sm" onClick={handleForward} disabled={!canGoForward}>
								下一个时间段
							</Button>
						</div>
						<ViewsChart chartData={processedData} />
					</div>
				</TabsContent>
				<TabsContent value="table">
					<StatsTable snapshots={paginatedSnapshots} />
					{totalPages > 1 && (
						<div className="flex justify-center items-center gap-2 mt-4">
							<Button
								variant="outline"
								size="sm"
								onClick={() => setCurrentPage((prev) => Math.max(prev - 1, 1))}
								disabled={currentPage === 1}
							>
								上一页
							</Button>
							<span className="text-sm text-secondary-foreground">
								第 {currentPage} 页，共 {totalPages} 页
							</span>
							<Button
								variant="outline"
								size="sm"
								onClick={() => setCurrentPage((prev) => Math.min(prev + 1, totalPages))}
								disabled={currentPage === totalPages}
							>
								下一页
							</Button>
						</div>
					)}
				</TabsContent>
			</Tabs>
		</div>
	);
};

const MemoizedSnapshotsView = memo(SnapshotsView);

export default function SongInfo({ loaderData }: Route.ComponentProps) {
	const [songInfo, setData] = useState<SongInfo | null>(null);
	const [snapshots, setSnapshots] = useState<Snapshots | null>(null);
	const [etaData, setEtaData] = useState<EtaInfo | null>(null);
	const [error, setError] = useState<SongInfoError | SnapshotsError | EtaInfoError | null>(null);
	const [isDialogOpen, setIsDialogOpen] = useState(false);
	const [songName, setSongName] = useState("");
	const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false);
	const [isDeleting, setIsDeleting] = useState(false);
	const [isSaving, setIsSaving] = useState(false);

	const getEta = async (aid: number) => {
		const { data, error } = await app.video({ id: `av${aid}` }).eta.get();
		if (error) {
			console.log(error);
			return;
		}
		setEtaData(data);
	};

	const getSnapshots = async (aid: number) => {
		const { data, error } = await app.video({ id: `av${aid}` }).snapshots.get();
		if (error) {
			console.log(error);
			return;
		}
		setSnapshots(data);
	};

	const getInfo = async () => {
		const { data, error } = await app.song({ id: loaderData.id }).info.get();
		if (error) {
			console.log(error);
			setError(error);
			return;
		}
		setData(data);
	};

	useEffect(() => {
		getInfo();
	}, []);

	useEffect(() => {
		if (!songInfo) return;
		const aid = songInfo.aid;
		if (!aid) return;
		getSnapshots(aid);
		getEta(aid);
	}, [songInfo]);

	useEffect(() => {
		if (songInfo?.name) {
			setSongName(songInfo.name);
		}
	}, [songInfo?.name]);

	if (!songInfo && !error) {
		return (
			<Layout>
				<Title title="加载中" />
				<Skeleton className="mt-6 w-full aspect-video rounded-lg" />
				<div className="mt-6 flex justify-between items-baseline">
					<Skeleton className="w-60 h-10 rounded-sm" />
					<Skeleton className="w-25 h-10 rounded-sm" />
				</div>
			</Layout>
		);
	}

	if (error?.status === 404) {
		return (
			<div className="w-screen min-h-screen flex items-center justify-center">
				<Title title="未找到曲目" />
				<div className="max-w-md w-full bg-gray-100 dark:bg-neutral-900 rounded-2xl shadow-lg p-6 flex flex-col gap-4 items-center text-center">
					<div className="w-16 h-16 flex items-center justify-center rounded-full bg-red-500 text-white text-3xl">
						<TriangleAlert size={34} className="-translate-y-0.5" />
					</div>
					<h1 className="text-3xl font-semibold text-neutral-900 dark:text-neutral-100">无法找到曲目</h1>
					<a href={`/song/${loaderData.id}/add`} className="text-secondary-foreground">
						点此收录
					</a>
				</div>
			</div>
		);
	}

	if (error) {
		return <Error error={error} />;
	}

	const formatDuration = (duration: number) => {
		return `${Math.floor(duration / 60)}:${(duration % 60).toString().padStart(2, "0")}`;
	};

	const handleSongNameChange = async () => {
		if (songName.trim() === "") return;
		setIsSaving(true);
		const { data, error } = await app.song({ id: loaderData.id }).info.patch(
			{ name: songName },
			{
				headers: {
					Authorization: `Bearer ${localStorage.getItem("sessionID") || ""}`,
				},
			},
		);
		setIsDialogOpen(false);
		setIsSaving(false);
		if (error || !data) {
			toast.error(`无法更新：${error.value.message || "未知错误"}`);
		}
		getInfo();
	};

	const handleDeleteSong = async () => {
		if (!songInfo) return;
		setIsDeleting(true);
		try {
			const { data, error } = await app.song({ id: songInfo.id }).delete(undefined, {
				headers: {
					Authorization: `Bearer ${localStorage.getItem("sessionID") || ""}`,
				},
			});

			if (error) {
				toast.error(`删除失败：${error.value.message || "未知错误"}`);
				return;
			}

			toast.success("歌曲删除成功");
			// Redirect to home page after successful deletion
			setTimeout(() => {
				window.location.href = "/";
			}, 1000);
		} catch (err) {
			toast.error("删除失败：网络错误");
		} finally {
			setIsDeleting(false);
			setIsDeleteDialogOpen(false);
		}
	};

	return (
		<Layout>
			<Title title={songInfo!.name ? songInfo!.name : "未知歌曲名"} />
			{songInfo!.cover && (
				<img
					src={songInfo!.cover}
					referrerPolicy="no-referrer"
					className="w-full aspect-video object-cover rounded-lg mt-6"
				/>
			)}
			<div className="mt-6 flex items-center gap-2">
				<h1 className="text-4xl font-medium" onDoubleClick={() => setIsDialogOpen(true)}>
					{songInfo!.name ? songInfo!.name : "未知歌曲名"}
				</h1>
				<Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
					<DialogContent>
						<DialogHeader>
							<DialogTitle>编辑歌曲名称</DialogTitle>
						</DialogHeader>
						<div className="space-y-4">
							<Input
								value={songName}
								onChange={(e) => setSongName(e.target.value)}
								placeholder="请输入歌曲名称"
								className="w-full"
							/>
							<div className="flex justify-end gap-2">
								<Button variant="outline" onClick={() => setIsDialogOpen(false)}>
									取消
								</Button>
								<Button onClick={handleSongNameChange}>{isSaving ? "保存中..." : "保存"}</Button>
							</div>
						</div>
					</DialogContent>
				</Dialog>
			</div>
			<div className="flex justify-between mt-3">
				<div>
					<If condition={songInfo!.aid}>
						<Then>
							<p>
								<span>av{songInfo!.aid}</span> · <span>{av2bv(songInfo!.aid!)}</span>
							</p>
						</Then>
					</If>
					<p>
						<If condition={songInfo!.duration}>
							<Then>
								<span>
									时长：
									{formatDuration(songInfo!.duration!)}
								</span>
							</Then>
						</If>
						<span> · </span>
						<If condition={songInfo!.publishedAt}>
							<Then>
								<span>发布于 {formatDateTime(new Date(songInfo!.publishedAt!))}</span>
							</Then>
						</If>
					</p>

					<span>
						P主：
						{songInfo!.producer ? songInfo!.producer : "未知P主"}
					</span>
				</div>
				<div className="flex flex-col gap-3">
					{songInfo!.aid && (
						<Button className="bg-pink-400">
							<a href={`https://www.bilibili.com/video/${av2bv(songInfo!.aid)}`}>哔哩哔哩</a>
						</Button>
					)}
					<AlertDialog open={isDeleteDialogOpen} onOpenChange={setIsDeleteDialogOpen}>
						<AlertDialogTrigger asChild>
							<Button variant="destructive">删除</Button>
						</AlertDialogTrigger>
						<AlertDialogContent>
							<AlertDialogHeader>
								<AlertDialogTitle>确认删除</AlertDialogTitle>
								<AlertDialogDescription>你确定要删除本歌曲吗？</AlertDialogDescription>
							</AlertDialogHeader>
							<AlertDialogFooter>
								<AlertDialogCancel>取消</AlertDialogCancel>
								<AlertDialogAction
									onClick={handleDeleteSong}
									disabled={isDeleting}
									className="bg-red-600 hover:bg-red-700"
								>
									{isDeleting ? "删除中..." : "确认删除"}
								</AlertDialogAction>
							</AlertDialogFooter>
						</AlertDialogContent>
					</AlertDialog>
				</div>
			</div>
			<MemoizedSnapshotsView
				snapshots={snapshots}
				etaData={etaData}
				publishedAt={songInfo!.publishedAt || undefined}
			/>
		</Layout>
	);
}
