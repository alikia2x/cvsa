import  { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
	Dialog,
	DialogContent,
	DialogDescription,
	DialogFooter,
	DialogHeader,
	DialogTitle,
	DialogTrigger
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import {
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Trash2, Plus, Database, FileText, Calendar, Activity } from "lucide-react";
import { apiClient } from "@/lib/api";
import { toast } from "sonner";
import { Spinner } from "@/components/ui/spinner"

export function DatasetManager() {
	const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
	const [createFormData, setCreateFormData] = useState({
		strategy: "all",
		limit: "",
		embeddingModel: "",
		description: "",
		forceRegenerate: false
	});

	const queryClient = useQueryClient();

	// Fetch datasets
	const { data: datasetsData, isLoading: datasetsLoading } = useQuery({
		queryKey: ["datasets"],
		queryFn: () => apiClient.getDatasets(),
		refetchInterval: 30000 // Refresh every 30 seconds
	});

	// Fetch embedding models
	const { data: modelsData, isLoading: modelsLoading } = useQuery({
		queryKey: ["embedding-models"],
		queryFn: () => apiClient.getEmbeddingModels()
	});

	// Create dataset mutation
	const createDatasetMutation = useMutation({
		mutationFn: (data: any) => apiClient.createDatasetWithSampling(data),
		onSuccess: () => {
			toast.success("Dataset creation task started");
			setIsCreateDialogOpen(false);
			setCreateFormData({
				strategy: "all",
				limit: "",
				embeddingModel: "",
				description: "",
				forceRegenerate: false
			});
			queryClient.invalidateQueries({ queryKey: ["datasets"] });
			queryClient.invalidateQueries({ queryKey: ["tasks"] });
		},
		onError: (error: Error) => {
			toast.error(`Creation failed: ${error.message}`);
		}
	});

	// Delete dataset mutation
	const deleteDatasetMutation = useMutation({
		mutationFn: (datasetId: string) => apiClient.deleteDataset(datasetId),
		onSuccess: () => {
			toast.success("Dataset deleted");
			queryClient.invalidateQueries({ queryKey: ["datasets"] });
		},
		onError: (error: Error) => {
			toast.error(`Delete failed: ${error.message}`);
		}
	});

	const handleCreateDataset = () => {
		if (!createFormData.embeddingModel) {
			toast.error("Please select an embedding model");
			return;
		}

		const requestData = {
			sampling: {
				strategy: createFormData.strategy,
				...(createFormData.limit && { limit: parseInt(createFormData.limit) })
			},
			embedding_model: createFormData.embeddingModel,
			force_regenerate: createFormData.forceRegenerate,
			description: createFormData.description || undefined
		};

		createDatasetMutation.mutate(requestData);
	};

	const handleDeleteDataset = (datasetId: string) => {
		if (window.confirm("Are you sure you want to delete this dataset?")) {
			deleteDatasetMutation.mutate(datasetId);
		}
	};

	const formatDate = (dateString: string) => {
		return new Date(dateString).toLocaleString("en-US");
	};

	const formatFileSize = (bytes: number) => {
		if (bytes === 0) return "0 Bytes";
		const k = 1024;
		const sizes = ["Bytes", "KB", "MB", "GB"];
		const i = Math.floor(Math.log(bytes) / Math.log(k));
		return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
	};

	if (datasetsLoading) {
		return (
			<div className="flex items-center justify-center h-64">
				<Spinner />
			</div>
		);
	}

	return (
		<div className="space-y-6">
			{/* Create Dataset Button */}
			<div className="flex justify-between items-center">
				<div>
					<h3 className="text-lg font-medium">Dataset List</h3>
					<p className="text-sm text-muted-foreground">
						{datasetsData?.datasets?.length || 0} datasets created
					</p>
				</div>

				<Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
					<DialogTrigger asChild>
						<Button>
							<Plus className="h-4 w-4 mr-2" />
							Create Dataset
						</Button>
					</DialogTrigger>
					<DialogContent className="sm:max-w-[500px]">
						<DialogHeader>
							<DialogTitle>Create New Dataset</DialogTitle>
							<DialogDescription>
								Select sampling strategy and configuration parameters to create a new dataset
							</DialogDescription>
						</DialogHeader>

						<div className="grid gap-4 py-4">
							<div className="grid gap-2">
								<Label htmlFor="strategy">Sampling Strategy</Label>
								<Select
									value={createFormData.strategy}
									onValueChange={(value) =>
										setCreateFormData((prev) => ({ ...prev, strategy: value }))
									}
								>
									<SelectTrigger>
										<SelectValue placeholder="Select sampling strategy" />
									</SelectTrigger>
									<SelectContent>
										<SelectItem value="all">All Videos</SelectItem>
										<SelectItem value="random">Random Sampling</SelectItem>
									</SelectContent>
								</Select>
							</div>

							{createFormData.strategy === "random" && (
								<div className="grid gap-2">
									<Label htmlFor="limit">Sample Count</Label>
									<Textarea
										id="limit"
										placeholder="Enter number of samples, e.g., 1000"
										value={createFormData.limit}
										onChange={(e) =>
											setCreateFormData((prev) => ({
												...prev,
												limit: e.target.value
											}))
										}
									/>
								</div>
							)}

							<div className="grid gap-2">
								<Label htmlFor="model">Embedding Model</Label>
								<Select
									value={createFormData.embeddingModel}
									onValueChange={(value) =>
										setCreateFormData((prev) => ({
											...prev,
											embeddingModel: value
										}))
									}
								>
									<SelectTrigger>
										<SelectValue placeholder="Select embedding model" />
									</SelectTrigger>
									<SelectContent>
										{modelsData?.models &&
											Object.keys(modelsData.models).map((modelName) => (
												<SelectItem key={modelName} value={modelName}>
													{modelName} (
													{modelsData.models[modelName].dimensions}D)
												</SelectItem>
											))}
									</SelectContent>
								</Select>
							</div>

							<div className="grid gap-2">
								<Label htmlFor="description">Description (Optional)</Label>
								<Textarea
									id="description"
									placeholder="Enter dataset description"
									value={createFormData.description}
									onChange={(e) =>
										setCreateFormData((prev) => ({
											...prev,
											description: e.target.value
										}))
									}
								/>
							</div>

							<div className="flex items-center space-x-2">
								<input
									type="checkbox"
									id="forceRegenerate"
									checked={createFormData.forceRegenerate}
									onChange={(e) =>
										setCreateFormData((prev) => ({
											...prev,
											forceRegenerate: e.target.checked
										}))
									}
								/>
								<Label htmlFor="forceRegenerate">Force Regenerate Embeddings</Label>
							</div>
						</div>

						<DialogFooter>
							<Button variant="outline" onClick={() => setIsCreateDialogOpen(false)}>
								Cancel
							</Button>
							<Button
								onClick={handleCreateDataset}
								disabled={createDatasetMutation.isPending}
							>
								{createDatasetMutation.isPending ? "Creating..." : "Create Dataset"}
							</Button>
						</DialogFooter>
					</DialogContent>
				</Dialog>
			</div>

			{/* Datasets List */}
			<div className="grid gap-4">
				{datasetsData?.datasets && datasetsData.datasets.length > 0 ? (
					datasetsData.datasets.map((dataset: any) => (
						<Card key={dataset.dataset_id}>
							<CardHeader className="pb-3">
								<div className="flex items-start justify-between">
									<div className="flex items-center space-x-2">
										<CardTitle className="text-base">
											{dataset.dataset_id.slice(0, 8)}...{dataset.dataset_id.slice(-8)}
										</CardTitle>
									</div>
									<Button
										variant="ghost"
										size="sm"
										onClick={() => handleDeleteDataset(dataset.dataset_id)}
										disabled={deleteDatasetMutation.isPending}
									>
										<Trash2 className="h-4 w-4" />
									</Button>
								</div>
								{dataset.description && (
									<CardDescription>{dataset.description}</CardDescription>
								)}
							</CardHeader>
							<CardContent>
								<div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 text-sm">
									<div className="flex items-center space-x-2">
										<span>{dataset.stats.total_records} records</span>
									</div>
									<div className="flex items-center space-x-2">
										<span>{dataset.stats.embedding_model}</span>
									</div>
									<div className="flex items-center space-x-2">
										<span>{formatDate(dataset.created_at)}</span>
									</div>
									<div className="flex items-center space-x-2">
										<span className="text-muted-foreground">
											New: {dataset.stats.new_embeddings}
										</span>
									</div>
									
								</div>
							</CardContent>
						</Card>
					))
				) : (
					<Card>
						<CardContent className="flex flex-col items-center justify-center py-12">
							<Database className="h-12 w-12 text-muted-foreground mb-4" />
							<h3 className="text-lg font-medium mb-2">No Datasets</h3>
							<p className="text-sm text-muted-foreground text-center mb-4">
								Start by creating your first dataset
							</p>
							<Button onClick={() => setIsCreateDialogOpen(true)}>
								<Plus className="h-4 w-4 mr-2" />
								Create Dataset
							</Button>
						</CardContent>
					</Card>
				)}
			</div>
		</div>
	);
}
