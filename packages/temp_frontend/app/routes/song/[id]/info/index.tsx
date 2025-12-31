import { av2bv } from "@backend/lib/bilibiliID";
import type { App } from "@backend/src";
import { treaty } from "@elysiajs/eden";
import { TriangleAlert } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { When } from "react-if";
import { toast } from "sonner";
import { ErrorPage } from "@/components/Error";
import { Layout } from "@/components/Layout";
import { formatDateTime } from "@/components/SearchResults";
import { Title } from "@/components/Title";
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
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { MemoizedSnapshotsView } from "@/routes/song/[id]/info/snapshotsView";
import type { Route } from "./+types/index";

const app = treaty<App>(import.meta.env.VITE_API_URL || "https://api.projectcvsa.com/");

type SongInfo = Awaited<ReturnType<ReturnType<typeof app.song>["info"]["get"]>>["data"];
export type EtaInfo = Awaited<ReturnType<ReturnType<typeof app.song>["eta"]["get"]>>["data"];
export type Snapshots = Awaited<
	ReturnType<ReturnType<typeof app.video>["snapshots"]["get"]>
>["data"];
type SongInfoError = Awaited<ReturnType<ReturnType<typeof app.song>["info"]["get"]>>["error"];
type SnapshotsError = Awaited<
	ReturnType<ReturnType<typeof app.video>["snapshots"]["get"]>
>["error"];
type EtaInfoError = Awaited<ReturnType<ReturnType<typeof app.video>["eta"]["get"]>>["error"];

// noinspection JSUnusedGlobalSymbols
export async function clientLoader({ params }: Route.LoaderArgs) {
	return { id: params.id };
}

export function formatHours(hours: number): string {
	if (hours >= 24 * 14) return `${Math.floor(hours / 24)} 天`;
	if (hours >= 24) return `${Math.floor(hours / 24)} 天 ${Math.round(hours) % 24} 小时`;
	if (hours >= 1) return `${Math.floor(hours)} 时 ${Math.round((hours % 1) * 60)} 分`;
	return `${Math.round(hours * 60)} 分钟`;
}

export function addHoursToNow(hours: number): string {
	const d = new Date();
	d.setSeconds(d.getSeconds() + hours * 3600);
	return formatDateTime(d, true);
}

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

	const getEta = useCallback(async () => {
		const { data, error } = await app.song({ id: loaderData.id }).eta.get();
		if (error) {
			console.log(error);
			return;
		}
		setEtaData(data);
	}, [loaderData.id]);

	const getSnapshots = useCallback(async () => {
		const { data, error } = await app.song({ id: loaderData.id }).snapshots.get();
		if (error) {
			console.log(error);
			return;
		}
		setSnapshots(data);
	}, [loaderData.id]);

	const getInfo = useCallback(async () => {
		const { data, error } = await app.song({ id: loaderData.id }).info.get();
		if (error) {
			console.log(error);
			setError(error);
			return;
		}
		setData(data);
	}, [loaderData.id]);

	useEffect(() => {
		getInfo().then(() => {});
		getSnapshots().then(() => {});
		getEta().then(() => {});
	}, [getEta, getInfo, getSnapshots]);

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
					<h1 className="text-3xl font-semibold text-neutral-900 dark:text-neutral-100">
						无法找到曲目
					</h1>
					<a href={`/song/${loaderData.id}/add`} className="text-secondary-foreground">
						点此收录
					</a>
				</div>
			</div>
		);
	}

	if (error) {
		return <ErrorPage error={error} />;
	}

	// Type guard: songInfo is guaranteed to be non-null at this point
	if (!songInfo) {
		throw new Error("Invariant violation: songInfo should not be null here");
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
			}
		);
		setIsDialogOpen(false);
		setIsSaving(false);
		if (error || !data) {
			toast.error(`无法更新：${error.value.message || "未知错误"}`);
		}
		getInfo().then(() => {});
	};

	const handleDeleteSong = async () => {
		if (!songInfo) return;
		setIsDeleting(true);
		try {
			const { error } = await app.song({ id: songInfo.id }).delete(undefined, {
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
		} catch {
			toast.error("删除失败：网络错误");
		} finally {
			setIsDeleting(false);
			setIsDeleteDialogOpen(false);
		}
	};

	return (
		<Layout>
			<Title title={songInfo.name ? songInfo.name : "未知歌曲名"} />
			<When condition={songInfo.cover}>
				<img
					src={songInfo?.cover}
					alt="封面图片"
					referrerPolicy="no-referrer"
					className="w-full aspect-video object-cover rounded-lg mt-6"
				/>
			</When>
			<div className="mt-6 flex items-center gap-2">
				<h1 className="text-4xl font-medium" onDoubleClick={() => setIsDialogOpen(true)}>
					{songInfo.name ? songInfo.name : "未知歌曲名"}
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
								<Button onClick={handleSongNameChange}>
									{isSaving ? "保存中..." : "保存"}
								</Button>
							</div>
						</div>
					</DialogContent>
				</Dialog>
			</div>
			<div className="flex justify-between mt-3  stat-num">
				<div>
					<When condition={songInfo.aid}>
						<p>
							<span>av{songInfo.aid}</span> · <span>{av2bv(songInfo.aid || 0)}</span>
						</p>
					</When>
					<p>
						<When condition={songInfo.duration}>
							<span>
								时长：
								{formatDuration(songInfo.duration || 0)}
							</span>
						</When>
						<span> · </span>
						<When condition={songInfo.publishedAt}>
							<span>
								发布于 {formatDateTime(new Date(songInfo.publishedAt || 0))}
							</span>
						</When>
					</p>

					<span>
						P主：
						{songInfo.producer ? songInfo.producer : "未知P主"}
					</span>
				</div>
				<div className="flex flex-col gap-3">
					{songInfo.aid && (
						<Button className="bg-pink-400">
							<a href={`https://www.bilibili.com/video/${av2bv(songInfo.aid)}`}>
								哔哩哔哩
							</a>
						</Button>
					)}
					<AlertDialog open={isDeleteDialogOpen} onOpenChange={setIsDeleteDialogOpen}>
						<AlertDialogTrigger asChild>
							<Button variant="destructive">删除</Button>
						</AlertDialogTrigger>
						<AlertDialogContent>
							<AlertDialogHeader>
								<AlertDialogTitle>确认删除</AlertDialogTitle>
								<AlertDialogDescription>
									你确定要删除本歌曲吗？
								</AlertDialogDescription>
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
				publishedAt={songInfo.publishedAt || ""}
			/>
		</Layout>
	);
}
