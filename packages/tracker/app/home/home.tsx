import type { Route } from "./+types/home";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Plus, Kanban, LogOut } from "lucide-react";
import { Link, Form, redirect } from "react-router";
import { db } from "@lib/db";
import { projects as projectsTable, tasks, users } from "@lib/db/schema";
import { count, eq } from "drizzle-orm";
import Layout from "@/components/layout";
import { getCurrentUser } from "@lib/auth-utils";
import { getUserProjects } from "@lib/auth";

export function meta({}: Route.MetaArgs) {
	return [{ title: "Projects - FramSpor" }];
}

export async function loader({ request }: { request: Request }) {
	// Check if there are any users
	const existingUsers = await db.select().from(users).limit(1);

	// If users exist, redirect to login
	if (existingUsers.length === 0) {
		return redirect("/setup");
	}

	const user = await getCurrentUser(request);
	if (!user) {
		return redirect("/login");
	}

	// Fetch user's accessible projects
	const projects = await getUserProjects(user.id);

	// For each project, count the number of tasks
	const projectsWithTaskCount = await Promise.all(
		projects.map(async (project) => {
			const taskCountResult = await db
				.select({ count: count() })
				.from(tasks)
				.where(eq(tasks.projectId, project.id));

			return {
				...project,
				taskCount: taskCountResult[0]?.count || 0
			};
		})
	);

	return { projects: projectsWithTaskCount, user };
}

export default function Home({ loaderData }: Route.ComponentProps) {
	const { projects, user } = loaderData as { projects: any[]; user: any };

	return (
		<Layout>
			{/* Header */}
			<div className="max-sm:flex-col max-sm:gap-6 flex justify-between mb-8">
				<div>
					<h1 className="text-3xl font-bold tracking-tight">Projects</h1>
					<p className="text-muted-foreground mt-2">
						Welcome, <Link to="/profile" className="text-blue-500">{user.username}</Link>! You have {projects.length} project
						{projects.length === 1 ? "" : "s"}.
						{user.isAdmin && <Link to="/admin/users"><span className="ml-2 text-blue-500">(Admin)</span></Link>}
					</p>
				</div>
				<div className="flex gap-2">
					<Link to="/logout">
						<Button type="submit" variant="outline">
							<LogOut className="size-4 mr-1" />
							Logout
						</Button>
					</Link>
					<Button asChild>
						<Link to="/project/new">
							<Plus className="size-4.5 mr-1" />
							New Project
						</Link>
					</Button>
				</div>
			</div>

			{/* Projects Grid */}
			<div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
				{projects.map((project: any) => (
					<Card key={project.id} className="hover:shadow-lg transition-shadow">
						<CardHeader>
							<div className="flex items-center gap-3">
								<div>
									<CardTitle className="text-xl">{project.name}</CardTitle>
									<CardDescription>{project.description}</CardDescription>
								</div>
							</div>
						</CardHeader>
						<CardContent>
							<div className="flex items-center justify-between">
								<span className="text-sm text-muted-foreground">
									{project.taskCount} tasks
								</span>
								<Button asChild variant="outline" size="sm">
									<Link to={`/project/${project.id}`}>Open Board</Link>
								</Button>
							</div>
						</CardContent>
					</Card>
				))}
			</div>

			{/* Empty State */}
			{projects.length === 0 && (
				<Card className="text-center py-12">
					<CardContent>
						<Kanban className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
						<h3 className="text-lg font-semibold mb-2">No projects yet</h3>
						<p className="text-muted-foreground mb-4">
							Create your first project to get started with task management
						</p>
						<Button asChild>
							<Link to="/project/new">
								<Plus className="size-4.5 mr-1" />
								Create Project
							</Link>
						</Button>
					</CardContent>
				</Card>
			)}
		</Layout>
	);
}
