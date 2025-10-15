import type { Route } from "./+types/newProject";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { ArrowLeft } from "lucide-react";
import { Link, Form, redirect } from "react-router";
import { db } from "@lib/db";
import { projects, columns } from "@lib/db/schema";
import { generate as generateId } from "@alikia/random-key";
import Layout from "@/components/layout";
import { getCurrentUser } from "@lib/auth-utils";

export function meta({}: Route.MetaArgs) {
	return [
		{ title: "Create New Project" },
		{ name: "description", content: "Create a new project for task management" }
	];
}

export async function action({ request }: Route.ActionArgs) {
	const user = await getCurrentUser(request);
	if (!user) {
		throw new Response("Unauthorized", { status: 401 });
	}

	const formData = await request.formData();
	const name = formData.get("name") as string;
	const description = formData.get("description") as string;

	if (!name) {
		return { error: "Project name is required" };
	}

	try {
		const projectId = await generateId(6);
		const now = new Date();

		// Create the project
		await db.insert(projects).values({
			id: projectId,
			ownerId: user.id,
			name,
			description,
			createdAt: now,
			updatedAt: now
		});

		// Create default columns for the project
		const defaultColumns = [
			{ name: "To Do", position: 0 },
			{ name: "In Progress", position: 1 },
			{ name: "Done", position: 2 }
		];

		for (const column of defaultColumns) {
			await db.insert(columns).values({
				id: await generateId(6),
				projectId,
				name: column.name,
				position: column.position,
				createdAt: now,
				updatedAt: now
			});
		}

		return redirect(`/project/${projectId}`);
	} catch (error) {
		console.error("Failed to create project:", error);
		return { error: "Failed to create project. Please try again." };
	}
}

export default function NewProject() {
	return (
		<Layout>
			{/* Header */}
			<div className="flex items-center gap-4 mb-8">
				<Button variant="ghost" size="icon" asChild>
					<Link to="/">
						<ArrowLeft className="w-4 h-4" />
					</Link>
				</Button>
				<div>
					<h1 className="text-3xl font-bold tracking-tight">Create A New Project</h1>
				</div>
			</div>

			{/* Project Creation Form */}
			<Card>
				<CardHeader>
					<CardTitle>Project Details</CardTitle>
					<CardDescription>
						Enter the basic information for your new project
					</CardDescription>
				</CardHeader>
				<CardContent>
					<Form method="post" className="space-y-4">
						<div className="space-y-2">
							<label htmlFor="name" className="text-sm font-medium">
								Project Name *
							</label>
							<Input
								id="name"
								name="name"
								placeholder="Enter project name"
								required
								className="w-full"
							/>
						</div>

						<div className="space-y-2">
							<label htmlFor="description" className="text-sm font-medium">
								Description
							</label>
							<Textarea
								id="description"
								name="description"
								placeholder="Describe your project (optional)"
								rows={4}
								className="w-full"
							/>
						</div>

						<div className="flex gap-2 pt-4">
							<Button variant="outline" asChild>
								<Link to="/">Cancel</Link>
							</Button>
							<Button type="submit">Create Project</Button>
						</div>
					</Form>
				</CardContent>
			</Card>
		</Layout>
	);
}
