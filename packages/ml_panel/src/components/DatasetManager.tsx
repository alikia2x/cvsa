import { useState } from "react";
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
import { Trash2, Plus, Database, Upload } from "lucide-react";
import { apiClient } from "@/lib/api";
import { toast } from "sonner";
import { Spinner } from "@/components/ui/spinner";

export function DatasetManager() {
	const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
	const [createFormData, setCreateFormData] = useState({
		creationMode: "sampling", // "sampling" or "aidList"
		strategy: "all",
		limit: "",
		embeddingModel: "",
		description: "",
		forceRegenerate: false,
		aidListFile: null as File | null,
		aidList: [] as number[]
	});

	const queryClient = useQueryClient();

	// Fetch datasets
	const { data: datasetsData, isLoading: datasetsLoading } = useQuery({
		queryKey: ["datasets"],
		queryFn: () => apiClient.getDatasets(),
		refetchInterval: 30000 // Refresh every 30 seconds
	});

	// Fetch embedding models
	const { data: modelsData } = useQuery({
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
				creationMode: "sampling",
				strategy: "all",
				limit: "",
				embeddingModel: "",
				description: "",
				forceRegenerate: false,
				aidListFile: null,
				aidList: []
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

	// Build dataset mutation
	const buildDatasetMutation = useMutation({
		mutationFn: (data: {
			aid_list: number[];
			embedding_model: string;
			force_regenerate?: boolean;
			description?: string;
		}) => apiClient.buildDataset(data),
		onSuccess: () => {
			toast.success("Dataset build task started");
			setIsCreateDialogOpen(false);
			setCreateFormData({
				creationMode: "sampling",
				strategy: "all",
				limit: "",
				embeddingModel: "",
				description: "",
				forceRegenerate: false,
				aidListFile: null,
				aidList: []
			});
			queryClient.invalidateQueries({ queryKey: ["datasets"] });
			queryClient.invalidateQueries({ queryKey: ["tasks"] });
		},
		onError: (error: Error) => {
			toast.error(`Build failed: ${error.message}`);
		}
	});

	const handleCreateDataset = () => {
		if (!createFormData.embeddingModel) {
			toast.error("Please select an embedding model");
			return;
		}

		if (createFormData.creationMode === "sampling") {
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
		} else if (createFormData.creationMode === "aidList") {
			if (createFormData.aidList.length === 0) {
				toast.error("Please upload an aid list file");
				return;
			}

			const requestData = {
				aid_list: createFormData.aidList,
				embedding_model: createFormData.embeddingModel,
				force_regenerate: createFormData.forceRegenerate,
				description: createFormData.description || undefined
			};

			buildDatasetMutation.mutate(requestData);
		}
	};

	const handleDeleteDataset = (datasetId: string) => {
		if (window.confirm("Are you sure you want to delete this dataset?")) {
			deleteDatasetMutation.mutate(datasetId);
		}
	};

	// Parse aid list file
	const parseAidListFile = (file: File): Promise<number[]> => {
		return new Promise((resolve, reject) => {
			const reader = new FileReader();
			reader.onload = (e) => {
				try {
					const content = e.target?.result as string;
					const lines = content.split("\n").filter((line) => line.trim());
					const aidList: number[] = [];

					for (const line of lines) {
						const trimmed = line.trim();
						if (trimmed) {
							const aid = parseInt(trimmed, 10);
							if (!isNaN(aid)) {
								aidList.push(aid);
							}
						}
					}

					resolve(aidList);
				} catch (error) {
					reject(new Error("Failed to parse file"));
				}
			};
			reader.onerror = () => reject(new Error("Failed to read file"));
			reader.readAsText(file);
		});
	};

	// Handle file upload
	const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
		const file = event.target.files?.[0];
		if (!file) return;

		if (!file.name.endsWith(".txt") && !file.name.endsWith(".csv")) {
			toast.error("Please upload a .txt or .csv file");
			return;
		}

		try {
			const aidList = await parseAidListFile(file);
			if (aidList.length === 0) {
				toast.error("No valid AIDs found in the file");
				return;
			}

			setCreateFormData((prev) => ({
				...prev,
				aidListFile: file,
				aidList: aidList
			}));

			toast.success(`Loaded ${aidList.length} AIDs from file`);
		} catch (error) {
			toast.error("Failed to parse aid list file");
		}
	};

	const formatDate = (dateString: string) => {
		return new Date(dateString).toLocaleString("en-US");
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
								Select sampling strategy and configuration parameters to create a
								new dataset
							</DialogDescription>
						</DialogHeader>

						<div className="grid gap-4 py-4">
							<div className="grid gap-2">
								<Label htmlFor="creationMode">Creation Mode</Label>
								<Select
									value={createFormData.creationMode}
									onValueChange={(value) =>
										setCreateFormData((prev) => ({
											...prev,
											creationMode: value,
											// Reset aid list when switching modes
											aidListFile: null,
											aidList: []
										}))
									}
								>
									<SelectTrigger>
										<SelectValue placeholder="Select creation mode" />
									</SelectTrigger>
									<SelectContent>
										<SelectItem value="sampling">Sampling Strategy</SelectItem>
										<SelectItem value="aidList">Upload Aid List</SelectItem>
									</SelectContent>
								</Select>
							</div>

							{createFormData.creationMode === "sampling" && (
								<div className="grid gap-2">
									<Label htmlFor="strategy">Sampling Strategy</Label>
									<Select
										value={createFormData.strategy}
										onValueChange={(value) =>
											setCreateFormData((prev) => ({
												...prev,
												strategy: value
											}))
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
							)}

							{createFormData.creationMode === "sampling" &&
								createFormData.strategy === "random" && (
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

							{createFormData.creationMode === "aidList" && (
								<div className="grid gap-2">
									<Label htmlFor="aidListFile">Aid List File</Label>
									<div
										className="border-2 border-dashed rounded-lg p-4 cursor-pointer"
										onClick={() =>
											document.getElementById("aidListFile")?.click()
										}
									>
										<div className="flex flex-col items-center space-y-2">
											<Upload className="h-8 w-8text-secondary-foreground" />
											<div className="text-sm text-secondary-foreground text-center">
												{createFormData.aidListFile
													? `${createFormData.aidListFile.name} (${createFormData.aidList.length} AIDs loaded)`
													: "Click to upload a .txt or .csv file containing AIDs (one per line)"}
											</div>
											<Button
												type="button"
												variant="outline"
												size="sm"
												className="mt-2"
												onClick={(e) => {
													e.stopPropagation();
													document.getElementById("aidListFile")?.click();
												}}
											>
												Choose File
											</Button>
										</div>
									</div>
									<input
										id="aidListFile"
										type="file"
										accept=".txt,.csv"
										onChange={handleFileUpload}
										className="hidden"
									/>
									{createFormData.aidList.length > 0 && (
										<div className="text-sm text-green-600">
											✓ Loaded {createFormData.aidList.length} AIDs from{" "}
											{createFormData.aidListFile?.name}
										</div>
									)}
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
								disabled={
									createDatasetMutation.isPending ||
									buildDatasetMutation.isPending ||
									(createFormData.creationMode === "aidList" &&
										createFormData.aidList.length === 0)
								}
							>
								{createDatasetMutation.isPending || buildDatasetMutation.isPending
									? "Creating..."
									: createFormData.creationMode === "sampling"
										? "Create Dataset"
										: "Build Dataset"}
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
										<CardTitle className="text-base line-clamp-1">
											{dataset.dataset_id}
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
								<div className="flex flex-wrap gap-5 text-sm leading-1">
									<span>{dataset.stats.total_records} records</span>
									<span>{dataset.stats.embedding_model}</span>
									<span>{formatDate(dataset.created_at)}</span>
									<span className="text-muted-foreground">
										New: {dataset.stats.new_embeddings}
									</span>
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
