import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { CardTitle } from "@/components/ui/card";
import { DatasetManager } from "@/components/DatasetManager";
import { TaskMonitor } from "@/components/TaskMonitor";
import { Database, Activity } from "lucide-react";

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
						<h1 className="text-3xl font-bold tracking-tight">
							CVSA Machine Learning Panel
						</h1>
					</div>

					<Tabs defaultValue="datasets" className="space-y-4">
						<TabsList className="grid w-full grid-cols-2">
							<TabsTrigger value="datasets" className="flex items-center gap-2">
								<Database className="h-4 w-4" />
								Datasets
							</TabsTrigger>
							<TabsTrigger value="monitor" className="flex items-center gap-2">
								<Activity className="h-4 w-4" />
								Tasks
							</TabsTrigger>
						</TabsList>

						<TabsContent value="datasets" className="space-y-4">
							<DatasetManager />
						</TabsContent>

						<TabsContent value="monitor" className="space-y-4">
							<CardTitle>Task Monitor</CardTitle>
							<TaskMonitor />
						</TabsContent>
					</Tabs>
				</div>
			</div>
		</QueryClientProvider>
	);
}

export default App;
