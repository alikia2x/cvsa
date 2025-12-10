import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from "@/components/ui/select";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { RefreshCw, Play, Pause, CheckCircle, XCircle, Clock } from "lucide-react";
import { apiClient } from "@/lib/api";
import type { TasksResponse } from "@/types/api";
import { Spinner } from "@/components/ui/spinner"

export function TaskMonitor() {
	const [statusFilter, setStatusFilter] = useState<string>("all");

	// Fetch tasks
	const {
		data: tasksData,
		isLoading: tasksLoading,
		refetch: refetchTasks
	} = useQuery<TasksResponse>({
		queryKey: ["tasks", statusFilter],
		queryFn: () => {
			const params = statusFilter === "all" ? {} : { status: statusFilter };
			return apiClient.getTasks(params.status, 50);
		},
		refetchInterval: 5000 // Refresh every 5 seconds
	});

	const getStatusIcon = (status: string) => {
		switch (status) {
			case "running":
				return <Play className="h-4 w-4 text-blue-500" />;
			case "completed":
				return <CheckCircle className="h-4 w-4 text-green-500" />;
			case "failed":
				return <XCircle className="h-4 w-4 text-red-500" />;
			case "pending":
				return <Clock className="h-4 w-4 text-yellow-500" />;
			default:
				return <Pause className="h-4 w-4 text-gray-500" />;
		}
	};

	const getStatusBadgeVariant = (status: string) => {
		switch (status) {
			case "running":
				return "default";
			case "completed":
				return "secondary";
			case "failed":
				return "destructive";
			case "pending":
				return "outline";
			default:
				return "outline";
		}
	};

	const formatDate = (dateString: string) => {
		return new Date(dateString).toLocaleString("en-US");
	};

	const formatDuration = (start: string, end?: string) => {
		const startTime = new Date(start).getTime();
		const endTime = end ? new Date(end).getTime() : Date.now();
		const duration = Math.floor((endTime - startTime) / 1000);

		if (duration < 60) return `${duration}s`;
		if (duration < 3600) return `${Math.floor(duration / 60)}m ${duration % 60}s`;
		return `${Math.floor(duration / 3600)}h ${Math.floor((duration % 3600) / 60)}m`;
	};

	if (tasksLoading) {
		return (
			<div className="flex items-center justify-center h-64">
				<Spinner/>
			</div>
		);
	}

	return (
		<div className="space-y-6">
			{/* Controls */}
			<div className="flex items-center justify-between">
				<div className="flex items-center space-x-4">
					<Select value={statusFilter} onValueChange={setStatusFilter}>
						<SelectTrigger className="w-40">
							<SelectValue placeholder="Task Status" />
						</SelectTrigger>
						<SelectContent>
							<SelectItem value="all">All Status</SelectItem>
							<SelectItem value="pending">Pending</SelectItem>
							<SelectItem value="running">Running</SelectItem>
							<SelectItem value="completed">Completed</SelectItem>
							<SelectItem value="failed">Failed</SelectItem>
						</SelectContent>
					</Select>
				</div>

				<Button variant="outline" size="sm" onClick={() => refetchTasks()}>
					<RefreshCw className="h-4 w-4 mr-2" />
					Refresh
				</Button>
			</div>

			{/* Tasks List */}
			<div className="space-y-3">
				{tasksData?.tasks && tasksData.tasks.length > 0 ? (
					tasksData.tasks.map((task: any) => (
						<Card key={task.task_id}>
							<CardContent className="p-4">
								<div className="flex items-start justify-between mb-3">
									<div className="flex items-center space-x-2">
										{getStatusIcon(task.status)}
										<span className="font-mono text-sm">
											{task.task_id.slice(0, 8)}...
										</span>
										<Badge variant={getStatusBadgeVariant(task.status)}>
											{task.status}
										</Badge>
									</div>
									<div className="text-sm text-muted-foreground">
										{formatDate(task.created_at)}
									</div>
								</div>

								{task.progress && (
									<div className="space-y-2 mb-3">
										<div className="flex items-center justify-between text-sm">
											<span>{task.progress.message}</span>
											<span>{task.progress.percentage.toFixed(1)}%</span>
										</div>
										<Progress
											value={task.progress.percentage}
											className="h-2"
										/>
										<div className="text-xs text-muted-foreground">
											Step {task.progress.completed_steps}/
											{task.progress.total_steps}:{" "}
											{task.progress.current_step}
										</div>
									</div>
								)}

								<div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
									{task.started_at && (
										<div>
											<span className="text-muted-foreground">Start Time:</span>
											<br />
											{formatDate(task.started_at)}
										</div>
									)}
									{task.completed_at && (
										<div>
											<span className="text-muted-foreground">Complete Time:</span>
											<br />
											{formatDate(task.completed_at)}
										</div>
									)}
									<div>
										<span className="text-muted-foreground">Duration:</span>
										<br />
										{formatDuration(
											task.started_at || task.created_at,
											task.completed_at
										)}
									</div>
									{task.result && (
										<div>
											<span className="text-muted-foreground">Result:</span>
											<br />
											{task.result.dataset_id
												? `Dataset: ${task.result.dataset_id.slice(0, 8)}...`
												: "None"}
										</div>
									)}
								</div>

								{task.error && (
									<div className="mt-3 p-2 bg-red-50 border border-red-200 rounded text-sm text-red-700">
										<strong>Error:</strong> {task.error}
									</div>
								)}
							</CardContent>
						</Card>
					))
				) : (
					<Card>
						<CardContent className="flex flex-col items-center justify-center py-12">
							<Clock className="h-12 w-12 text-muted-foreground mb-4" />
							<h3 className="text-lg font-medium mb-2">No Tasks</h3>
							<p className="text-sm text-muted-foreground text-center">
								{statusFilter === "all"
									? "No tasks found"
									: `No ${statusFilter} tasks found`}
							</p>
						</CardContent>
					</Card>
				)}
			</div>
		</div>
	);
}
