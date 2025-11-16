import type { Route } from "./+types/add";
import { treaty } from "@elysiajs/eden";
import type { App } from "@backend/src";
import { useEffect, useState } from "react";
import { Skeleton } from "@/components/ui/skeleton";
import { TriangleAlert, CheckCircle, Clock, AlertCircle } from "lucide-react";
import { Title } from "@/components/Title";
import { Search } from "@/components/Search";
import { Error } from "@/components/Error";
import { Layout } from "@/components/Layout";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { toast } from "sonner";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

// @ts-ignore idk
const app = treaty<App>(import.meta.env.VITE_API_URL!);

type SongInfo = Awaited<ReturnType<ReturnType<typeof app.song>["info"]["get"]>>["data"];
type SongInfoError = Awaited<ReturnType<ReturnType<typeof app.song>["info"]["get"]>>["error"];
type ImportStatus = {
	id: string;
	state: string;
	result?: any;
	failedReason?: string;
};

export async function clientLoader({ params }: Route.LoaderArgs) {
	return { id: params.id };
}

export default function SongInfo({ loaderData }: Route.ComponentProps) {
	const [isImporting, setIsImporting] = useState(false);
	const [importStatus, setImportStatus] = useState<ImportStatus | null>(null);
	const [importInterval, setImportInterval] = useState<NodeJS.Timeout | null>(null);

	const importSong = async () => {
		const response = await app.song.import.bilibili.post(
			{ id: loaderData.id },
			{
				headers: {
					Authorization: `Bearer ${localStorage.getItem("sessionID") || ""}`,
				},
			},
		);

		if (response.error) {
			toast.error(`导入失败：${response.error.value.message || "未知错误"}`);
			setIsImporting(false);
			return;
		}

		// @ts-ignore - Type issues with Eden treaty
		const jobID = response.data?.jobID;
		if (!jobID) {
			toast.error("导入失败：未收到任务ID");
			setIsImporting(false);
			return;
		}
		toast.success("歌曲导入任务已提交，正在处理中...");
		// Start polling for import status
		const interval = setInterval(async () => {
			const { data, error } = await app.song.import({ id: jobID }).status.get({
				headers: {
					Authorization: `Bearer ${localStorage.getItem("sessionID") || ""}`,
				},
			});
			if (error) {
				toast.error(`导入失败：${error.value.message || "未知错误"}`);
				setIsImporting(false);
				clearInterval(interval);
				return;
			}
			if (!data) {
				return;
			}
			setImportStatus(data);
			if (data.state !== "completed" && data.state !== "failed") {
				return;
			}
			clearInterval(interval);
			setIsImporting(false);
			if (data.state !== "completed") {
				toast.error(`导入失败：${data.failedReason || "未知错误"}`);
				return;
			}
			toast.success("歌曲导入成功！");
			// Redirect to song info page after successful import
			setTimeout(() => {
				window.location.href = `/song/${loaderData.id}/info`;
			}, 2000);
		}, 2000);
		setImportInterval(interval);
	};

	const handleImportSong = async () => {
		setIsImporting(true);
		try {
			await importSong();
		} catch (err) {
			toast.error("导入失败：网络错误");
			setIsImporting(false);
		}
	};

	useEffect(() => {
		return () => {
			if (importInterval) {
				clearInterval(importInterval);
			}
		};
	}, [importInterval]);

	const getStatusIcon = (state: string) => {
		switch (state) {
			case "completed":
				return <CheckCircle className="h-6 w-6 text-green-500" />;
			case "failed":
				return <AlertCircle className="h-6 w-6 text-red-500" />;
			case "active":
				return <Clock className="h-6 w-6 text-blue-500 animate-spin" />;
			default:
				return <Clock className="h-6 w-6 text-gray-500" />;
		}
	};

	const getStatusText = (state: string) => {
		switch (state) {
			case "completed":
				return "导入完成";
			case "failed":
				return "导入失败";
			case "active":
				return "正在导入";
			case "waiting":
				return "等待处理";
			case "delayed":
				return "延迟处理";
			default:
				return "未知状态";
		}
	};

	return (
		<Layout>
			<Title title="收录歌曲" />
			<Card className="mx-auto mt-8">
				<CardHeader>
					<CardTitle>收录歌曲</CardTitle>
					<CardDescription>将 Bilibili 视频 ID "{loaderData.id}" 收录为歌曲</CardDescription>
				</CardHeader>
				<CardContent className="space-y-6">
					{!importStatus ? (
						<div className="text-center space-y-4">
							<p className="text-lg">
								您将要收录视频 ID: <strong>{loaderData.id}</strong>
							</p>
							<Button onClick={handleImportSong} disabled={isImporting} size="lg">
								{isImporting ? "提交中..." : "开始收录"}
							</Button>
						</div>
					) : (
						<div className="space-y-4">
							<div className="flex items-center gap-3 p-4 border rounded-lg">
								{getStatusIcon(importStatus.state)}
								<div className="flex-1">
									<p className="font-medium">{getStatusText(importStatus.state)}</p>
									<p className="text-sm text-gray-500">任务 ID: {importStatus.id}</p>
									{importStatus.failedReason && (
										<p className="text-sm text-red-500 mt-1">
											失败原因: {importStatus.failedReason}
										</p>
									)}
								</div>
							</div>

							{importStatus.state === "completed" && (
								<div className="text-center">
									<p className="text-green-600 mb-4">歌曲收录成功！正在跳转到歌曲页面...</p>
									<Button
										onClick={() => (window.location.href = `/song/${loaderData.id}/info`)}
										variant="outline"
									>
										立即查看
									</Button>
								</div>
							)}

							{importStatus.state === "failed" && (
								<div className="text-center">
									<Button onClick={handleImportSong} disabled={isImporting}>
										{isImporting ? "重新提交中..." : "重新尝试"}
									</Button>
								</div>
							)}
						</div>
					)}
				</CardContent>
			</Card>
		</Layout>
	);
}
