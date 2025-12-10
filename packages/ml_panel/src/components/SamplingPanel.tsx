import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Database, Play, TestTube, Settings, BarChart3 } from "lucide-react";
import { apiClient } from "@/lib/api";
import type { SamplingResponse, DatasetCreateResponse } from "@/types/api";

interface SamplingConfig {
	strategy: string;
	limit?: number;
}

export function SamplingPanel() {
	const [samplingConfig, setSamplingConfig] = useState<SamplingConfig>({
		strategy: "all",
		limit: undefined,
	});

	const [embeddingModel, setEmbeddingModel] = useState<string>("");
	const [description, setDescription] = useState<string>("");


	// Test sampling mutation
	const testSamplingMutation = useMutation({
		mutationFn: (config: SamplingConfig) => apiClient.sampleDataset(config),
		onSuccess: (data: SamplingResponse) => {
			console.log("Sampling test successful:", data);
		},
		onError: (error: Error) => {
			console.error("Sampling test failed:", error);
		}
	});

	// Create dataset with sampling mutation
	const createDatasetMutation = useMutation({
		mutationFn: (config: {
			sampling: SamplingConfig;
			embedding_model: string;
			description?: string;
		}) => apiClient.createDatasetWithSampling(config),
		onSuccess: (data: DatasetCreateResponse) => {
			console.log("Dataset created successfully:", data);
		},
		onError: (error: Error) => {
			console.error("Dataset creation failed:", error);
		}
	});

	const handleStrategyChange = (strategy: string) => {
		setSamplingConfig((prev) => ({ ...prev, strategy }));
	};

	const handleLimitChange = (limit: string) => {
		setSamplingConfig((prev) => ({
			...prev,
			limit: limit ? parseInt(limit) : undefined
		}));
	};

	const handleTestSampling = () => {
		testSamplingMutation.mutate(samplingConfig);
	};

	const handleCreateDataset = () => {
		if (!embeddingModel) {
			alert("Please select an embedding model");
			return;
		}

		createDatasetMutation.mutate({
			sampling: samplingConfig,
			embedding_model: embeddingModel,
			description: description || undefined
		});
	};

	const getStrategyDescription = (strategy: string) => {
		switch (strategy) {
			case "all":
				return "Sample all labeled videos";
			case "random":
				return "Randomly sample specified number of labeled videos";
			default:
				return "Unknown strategy";
		}
	};

	return (
		<div className="space-y-6">
			<Tabs defaultValue="configure" className="w-full">
				<TabsList className="w-full mb-4">
					<TabsTrigger value="configure">
						<Settings className="h-4 w-4 mr-2" />
						Configure Sampling
					</TabsTrigger>
					<TabsTrigger value="test">
						<TestTube className="h-4 w-4 mr-2" />
						Test Sampling
					</TabsTrigger>
				</TabsList>

				<TabsContent value="configure" className="space-y-4">
					<Card>
						<CardHeader>
							<CardTitle>Sampling Strategy Configuration</CardTitle>
							<CardDescription>Select data sampling strategy and parameters</CardDescription>
						</CardHeader>
						<CardContent className="space-y-4">
							<div className="grid grid-cols-2 gap-4">
								<div className="space-y-2">
									<Label htmlFor="strategy">Sampling Strategy</Label>
									<Select
										value={samplingConfig.strategy}
										onValueChange={handleStrategyChange}
									>
										<SelectTrigger>
											<SelectValue placeholder="Select strategy" />
										</SelectTrigger>
										<SelectContent>
											<SelectItem value="all">All Labeled Videos</SelectItem>
											<SelectItem value="random">Random Sampling</SelectItem>
										</SelectContent>
									</Select>
									<p className="text-sm text-muted-foreground">
										{getStrategyDescription(samplingConfig.strategy)}
									</p>
								</div>

								{samplingConfig.strategy === "random" && (
									<div className="space-y-2">
										<Label htmlFor="limit">Sample Count</Label>
										<Input
											id="limit"
											type="number"
											placeholder="e.g., 1000"
											value={samplingConfig.limit || ""}
											onChange={(e) => handleLimitChange(e.target.value)}
										/>
									</div>
								)}
							</div>
						</CardContent>
					</Card>
				</TabsContent>

				<TabsContent value="test" className="space-y-4">
					<Card>
						<CardHeader>
							<CardTitle>Test Sampling</CardTitle>
							<CardDescription>
								Test sampling strategy and view data statistics for sampling
							</CardDescription>
						</CardHeader>
						<CardContent className="space-y-4">
							<div className="flex space-x-4">
								<Button
									onClick={handleTestSampling}
									disabled={testSamplingMutation.isPending}
									className="flex-1"
								>
									<Play className="h-4 w-4 mr-2" />
									{testSamplingMutation.isPending ? "Testing..." : "Start Test"}
								</Button>
							</div>

							{testSamplingMutation.isSuccess && testSamplingMutation.data && (
								<Alert>
									<BarChart3 className="h-4 w-4" />
									<AlertDescription>
										<div className="space-y-2">
											<div className="flex items-center justify-between">
												<span>Total available data:</span>
												<Badge variant="outline">
													{(
														testSamplingMutation.data as SamplingResponse
													).total_available.toLocaleString()}
												</Badge>
											</div>
											<div className="flex items-center justify-between">
												<span>Will sample:</span>
												<Badge>
													{(
														testSamplingMutation.data as SamplingResponse
													).sampled_count.toLocaleString()}
												</Badge>
											</div>
											<div className="flex items-center justify-between">
												<span>Sampling ratio:</span>
												<Badge variant="secondary">
													{(
														((
															testSamplingMutation.data as SamplingResponse
														).sampled_count /
															(
																testSamplingMutation.data as SamplingResponse
															).total_available) *
														100
													).toFixed(1)}
													%
												</Badge>
											</div>
										</div>
									</AlertDescription>
								</Alert>
							)}

							{testSamplingMutation.isError && (
								<Alert variant="destructive">
									<AlertDescription>
										Test failed: {(testSamplingMutation.error as Error).message}
									</AlertDescription>
								</Alert>
							)}
						</CardContent>
					</Card>
				</TabsContent>
			</Tabs>
		</div>
	);
}
