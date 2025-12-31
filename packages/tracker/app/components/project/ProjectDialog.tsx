import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
	Dialog,
	DialogContent,
	DialogDescription,
	DialogFooter,
	DialogHeader,
	DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";

interface ProjectDialogProps {
	open: boolean;
	onOpenChange: (open: boolean) => void;
	onSubmit: (data: { name: string; description: string; isPublic: boolean }) => void;
	onDelete?: () => void;
	initialData?: {
		name: string;
		description: string;
		isPublic?: boolean;
	};
	isEditing?: boolean;
}

export function ProjectDialog({
	open,
	onOpenChange,
	onSubmit,
	onDelete,
	initialData,
	isEditing = false,
}: ProjectDialogProps) {
	const [name, setName] = useState(initialData?.name || "");
	const [description, setDescription] = useState(initialData?.description || "");
	const [isPublic, setIsPublic] = useState(initialData?.isPublic || false);

	const handleSubmit = (e: React.FormEvent) => {
		e.preventDefault();
		onSubmit({
			description,
			isPublic,
			name,
		});
		onOpenChange(false);
		// Reset form
		if (!isEditing) {
			setName("");
			setDescription("");
		}
	};

	const handleOpenChange = (open: boolean) => {
		onOpenChange(open);
		if (!open && !isEditing) {
			// Reset form when closing dialog in create mode
			setName("");
			setDescription("");
		}
	};

	return (
		<Dialog open={open} onOpenChange={handleOpenChange}>
			<DialogContent className="sm:max-w-[425px]">
				<form onSubmit={handleSubmit}>
					<DialogHeader>
						<DialogTitle>{isEditing ? "Edit Project" : "Create Project"}</DialogTitle>
						<DialogDescription>
							{isEditing
								? "Update your project details."
								: "Create a new project to organize your tasks."}
						</DialogDescription>
					</DialogHeader>
					<div className="grid gap-4 py-4">
						<div className="grid gap-2">
							<Label htmlFor="name">Project Name</Label>
							<Input
								id="name"
								value={name}
								onChange={(e) => setName(e.target.value)}
								placeholder="Enter project name"
								required
							/>
						</div>
						<div className="grid gap-2">
							<Label htmlFor="description">Description</Label>
							<Textarea
								id="description"
								value={description}
								onChange={(e) => setDescription(e.target.value)}
								placeholder="Enter project description (optional)"
								rows={3}
							/>
						</div>
						<div className="flex items-center gap-2">
							<Checkbox
								id="isPublic"
								checked={isPublic}
								onCheckedChange={(c: boolean) => setIsPublic(c)}
								className="rounded border-gray-300"
							/>
							<Label htmlFor="isPublic">Public Project</Label>
						</div>
					</div>
					<DialogFooter className="flex justify-between">
						<div>
							{isEditing && onDelete && (
								<Button
									type="button"
									variant="destructive"
									onClick={() => {
										onDelete();
										onOpenChange(false);
									}}
								>
									Delete Project
								</Button>
							)}
						</div>
						<div className="flex gap-2">
							<Button
								type="button"
								variant="outline"
								onClick={() => onOpenChange(false)}
							>
								Cancel
							</Button>
							<Button type="submit">{isEditing ? "Update" : "Create"}</Button>
						</div>
					</DialogFooter>
				</form>
			</DialogContent>
		</Dialog>
	);
}
