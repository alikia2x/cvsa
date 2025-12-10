import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { CardDescription, CardTitle } from "@/components/ui/card";
import { DatasetManager } from "@/components/DatasetManager";
import { TaskMonitor } from "@/components/TaskMonitor";
import { SamplingPanel } from "@/components/SamplingPanel";
import { Database, Activity, Settings } from "lucide-react";

const queryClient = new QueryClient({
	defaultOptions: {
		queries: {
			retry: 3,
			refetchOnWindowFocus: false
		}
	}
});

function App() {
	return (
		<QueryClientProvider client={queryClient}>
			<div className="min-h-screen flex justify-center">
				<div className="container lg:max-w-3xl xl:max-w-4xl bg-background py-8 px-3">
					<div className="mb-8">
						<h1 className="text-3xl font-bold tracking-tight">ML Dataset Management Panel</h1>
						<p className="text-muted-foreground">
							Create and manage machine learning datasets with multiple sampling strategies and task monitoring
						</p>
					</div>

					<Tabs defaultValue="datasets" className="space-y-4">
						<TabsList className="grid w-full grid-cols-3">
							<TabsTrigger value="datasets" className="flex items-center gap-2">
								<Database className="h-4 w-4" />
								Datasets
							</TabsTrigger>
							<TabsTrigger value="sampling" className="flex items-center gap-2">
								<Settings className="h-4 w-4" />
								Sampling
							</TabsTrigger>
							<TabsTrigger value="monitor" className="flex items-center gap-2">
								<Activity className="h-4 w-4" />
								Tasks
							</TabsTrigger>
						</TabsList>

						<TabsContent value="datasets" className="space-y-4">
							<CardTitle>Dataset Management</CardTitle>
							<CardDescription>View, create and manage your machine learning datasets</CardDescription>
							<DatasetManager />
						</TabsContent>

						<TabsContent value="sampling" className="space-y-4">
							<CardTitle>Sampling Strategy Configuration</CardTitle>
							<CardDescription>
								Configure different data sampling strategies to create balanced datasets
							</CardDescription>
							<SamplingPanel />
						</TabsContent>

						<TabsContent value="monitor" className="space-y-4">
							<CardTitle>Task Monitor</CardTitle>
							<CardDescription>Monitor real-time status and progress of dataset building tasks</CardDescription>
							<TaskMonitor />
						</TabsContent>
					</Tabs>
				</div>
			</div>
		</QueryClientProvider>
	);
}

export default App;
