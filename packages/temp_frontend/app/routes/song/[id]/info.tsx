import type { Route } from "./+types/info";
import { treaty } from "@elysiajs/eden";
import type { App } from "@elysia/src";
import { useEffect, useState } from "react";
import { Skeleton } from "@/components/ui/skeleton";
import { TriangleAlert } from "lucide-react";
import { Title } from "@/components/Title";
import { Search } from "@/components/Search";
import { Error } from "@/components/Error";
import { Layout } from "@/components/Layout";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

const app = treaty<App>(import.meta.env.VITE_API_URL!);

type SongInfo = Awaited<ReturnType<ReturnType<typeof app.song>["info"]["get"]>>["data"];
type Snapshots = Awaited<ReturnType<ReturnType<typeof app.video>["snapshots"]["get"]>>["data"];
type SongInfoError = Awaited<ReturnType<ReturnType<typeof app.song>["info"]["get"]>>["error"];
type SnapshotsError = Awaited<ReturnType<ReturnType<typeof app.video>["snapshots"]["get"]>>["error"];

export async function clientLoader({ params }: Route.LoaderArgs) {
	return { id: params.id };
}

const SnapshotsView = ({ snapshots }: { snapshots: Snapshots | null }) => {
	if (!snapshots) {
		return (
			<>
				<h2 className="mt-6 text-2xl font-medium mb-4">历史快照</h2>
				<Skeleton className="w-full h-20 rounded-lg" />
			</>
		);
	}
	return (
		<div>
			<h2 className="mt-6 text-2xl font-medium mb-4">历史快照</h2>
			<table>
				<thead>
					<tr>
						<th className="text-left pr-4">日期</th>
						<th className="text-left pr-4">播放量</th>
						<th className="text-left pr-4">弹幕数</th>
						<th className="text-left pr-4">点赞数</th>
						<th className="text-left pr-4">收藏数</th>
						<th className="text-left pr-4">硬币数</th>
					</tr>
				</thead>
				<tbody>
					{snapshots.map((snapshot: Exclude<Snapshots, null>[number]) => (
						<tr key={snapshot.id}>
							<td className="pr-4">{new Date(snapshot.createdAt).toLocaleDateString()}</td>
							<td className="pr-4">{snapshot.views}</td>
							<td className="pr-4">{snapshot.danmakus}</td>
							<td className="pr-4">{snapshot.likes}</td>
							<td className="pr-4">{snapshot.favorites}</td>
							<td className="pr-4">{snapshot.coins}</td>
						</tr>
					))}
				</tbody>
			</table>
		</div>
	);
};

export default function SongInfo({ loaderData }: Route.ComponentProps) {
	const [data, setData] = useState<SongInfo | null>(null);
	const [snapshots, setSnapshots] = useState<Snapshots | null>(null);
	const [error, setError] = useState<SongInfoError | SnapshotsError | null>(null);
	const [isDialogOpen, setIsDialogOpen] = useState(false);
	const [songName, setSongName] = useState("");

	const getSnapshots = async (aid: number) => {
		const { data, error } = await app.video({ id: `av${aid}` }).snapshots.get();
		if (error) {
			console.log(error);
			setError(error);
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
		if (!data) return;
		const aid = data.aid;
		if (!aid) return;
		getSnapshots(aid);
	}, [data]);

	// Update local song name when data changes
	useEffect(() => {
		if (data?.name) {
			setSongName(data.name);
		}
	}, [data?.name]);

	if (!data && !error) {
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
		const minutes = Math.floor(duration / 60);
		const seconds = duration % 60;
		return `${minutes}:${seconds}`;
	};

	const handleSongNameChange = async () => {
		if (songName.trim() === "") return;

		await app.song({ id: loaderData.id }).info.patch({ name: songName });
		setIsDialogOpen(false);
		// Refresh the data to show the updated name
		getInfo();
	};

	return (
		<div className="w-screen min-h-screen relative left-0 top-0 flex justify-center">
			<Title title={data!.name ? data!.name : "未知歌曲名"} />
			<main className="w-full max-sm:mx-6 pt-14 sm:w-xl xl:w-2xl">
				<a href="/">
					<h1 className="text-4xl mb-5">中V档案馆</h1>
				</a>
				<Search />
				{data!.cover && (
					<img
						src={data!.cover}
						referrerPolicy="no-referrer"
						className="w-full aspect-video object-cover rounded-lg mt-6"
					/>
				)}
				<div className="mt-6 flex justify-between">
					<div className="flex items-center gap-2">
						<h1 className="text-4xl font-medium" onDoubleClick={() => setIsDialogOpen(true)}>
							{data!.name ? data!.name : "未知歌曲名"}
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
										<Button onClick={handleSongNameChange}>保存</Button>
									</div>
								</div>
							</DialogContent>
						</Dialog>
					</div>
					<div className="flex flex-col items-end h-10 whitespace-nowrap">
						<span className="leading-5 text-neutral-800 dark:text-neutral-200">
							{data!.duration ? formatDuration(data!.duration) : "未知时长"}
						</span>
						<span className="text-lg leading-5 text-neutral-800 dark:text-neutral-200 font-bold">
							{data!.producer ? data!.producer : "未知P主"}
						</span>
					</div>
				</div>
				<SnapshotsView snapshots={snapshots} />
			</main>
		</div>
	);
}
