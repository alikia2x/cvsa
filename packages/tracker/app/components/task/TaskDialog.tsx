import { useState } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { TaskForm } from "./TaskForm";

interface TaskDialogProps {
	open: boolean;
	onOpenChange: (open: boolean) => void;
	projectId: string;
	columns: Array<{ id: string; name: string }>;
	onDelete: () => Promise<void>;
	onSubmit: (data: {
		title: string;
		description: string;
		columnId: string;
		priority: "low" | "medium" | "high";
		dueDate?: Date;
	}) => Promise<void>;
	initialData?: {
		title?: string;
		description?: string;
		columnId?: string;
		priority?: "low" | "medium" | "high";
		dueDate?: Date;
	};
	isEditing?: boolean;
	canEdit?: boolean;
}

export function TaskDialog({
	open,
	onOpenChange,
	columns,
	onSubmit,
	onDelete,
	initialData,
	isEditing = false,
	canEdit = true
}: TaskDialogProps) {
	const handleSubmit = async (data: {
		title: string;
		description: string;
		columnId: string;
		priority: "low" | "medium" | "high";
		dueDate?: Date;
	}) => {
		try {
			await onSubmit(data);
			onOpenChange(false);
		} finally {
		}
	};

	const handleDelete = async () => {
		try {
			await onDelete();
			onOpenChange(false);
		} finally {
		}
	};

	return (
		<Dialog open={open} onOpenChange={onOpenChange}>
			<DialogContent className="sm:max-w-[425px]">
				<DialogHeader>
					<DialogTitle>
						{!canEdit ? "View Task" : isEditing ? "Edit Task" : "Create New Task"}
					</DialogTitle>
				</DialogHeader>
				<TaskForm
					columns={columns}
					onSubmit={handleSubmit}
					onDelete={handleDelete}
					initialData={initialData}
					isEditing={isEditing}
					canEdit={canEdit}
				/>
			</DialogContent>
		</Dialog>
	);
}
