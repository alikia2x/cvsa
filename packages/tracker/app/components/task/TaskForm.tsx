import { format } from "date-fns";
import { CalendarIcon, Flag, Trash2, X } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Calendar } from "@/components/ui/calendar";
import { Input } from "@/components/ui/input";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import {
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";

interface TaskFormProps {
	columns: Array<{ id: string; name: string }>;
	onSubmit: (data: {
		title: string;
		description: string;
		columnId: string;
		priority: "low" | "medium" | "high";
		dueDate?: Date;
	}) => Promise<void>;
	onDelete: () => void;
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

export function TaskForm({
	columns,
	onSubmit,
	onDelete,
	initialData,
	isEditing = false,
	canEdit = true,
}: TaskFormProps) {
	const [title, setTitle] = useState(initialData?.title || "");
	const [description, setDescription] = useState(initialData?.description || "");
	const [columnId, setColumnId] = useState(initialData?.columnId || columns[0]?.id || "");
	const [priority, setPriority] = useState<"low" | "medium" | "high">(
		initialData?.priority || "medium"
	);
	const [dueDate, setDueDate] = useState<Date | undefined>(initialData?.dueDate);
	const [isSubmitting, setIsSubmitting] = useState(false);

	const currentColumn = columns.find((col) => col.id === columnId);
	const priorityLabel = { high: "High", low: "Low", medium: "Medium" }[priority];
	const priorityColor = {
		high: "bg-red-100 text-red-800",
		low: "bg-green-100 text-green-800",
		medium: "bg-yellow-100 text-yellow-800",
	}[priority];

	const handleSubmit = async (e: React.FormEvent) => {
		e.preventDefault();

		if (!title.trim() || !columnId) {
			return;
		}

		setIsSubmitting(true);
		try {
			await onSubmit({
				columnId,
				description: description.trim(),
				dueDate,
				priority,
				title: title.trim(),
			});
		} finally {
			setIsSubmitting(false);
		}
	};

	return (
		<form onSubmit={handleSubmit} className="space-y-4">
			<div className="space-y-2">
				<label htmlFor="title" className="text-sm font-medium">
					Task Title *
				</label>
				{canEdit ? (
					<Input
						id="title"
						value={title}
						onChange={(e) => setTitle(e.target.value)}
						placeholder="Enter task title"
						required
						className="w-full"
					/>
				) : (
					<div className="p-3 bg-background border rounded-md text-foreground">
						{title || <span className="text-muted-foreground">No title</span>}
					</div>
				)}
			</div>

			<div className="space-y-2">
				<label htmlFor="description" className="text-sm font-medium">
					Description
				</label>
				{canEdit ? (
					<Textarea
						id="description"
						value={description}
						onChange={(e) => setDescription(e.target.value)}
						placeholder="Describe the task (optional)"
						rows={3}
						className="w-full max-h-60"
					/>
				) : (
					<div className="p-3 bg-background border rounded-md text-foreground min-h-[80px]">
						{description || (
							<span className="text-muted-foreground">No description</span>
						)}
					</div>
				)}
			</div>

			<div className="grid grid-cols-2 gap-4">
				<div className="space-y-2">
					<label htmlFor="column" className="text-sm font-medium">
						Column *
					</label>
					{canEdit ? (
						<Select value={columnId} onValueChange={setColumnId}>
							<SelectTrigger>
								<SelectValue placeholder="Select column" />
							</SelectTrigger>
							<SelectContent>
								{columns.map((column) => (
									<SelectItem key={column.id} value={column.id}>
										{column.name}
									</SelectItem>
								))}
							</SelectContent>
						</Select>
					) : (
						<div className="h-12 px-4 flex items-center bg-background border rounded-md text-foreground">
							{currentColumn?.name || (
								<span className="text-muted-foreground">No column</span>
							)}
						</div>
					)}
				</div>

				<div className="space-y-2">
					<label htmlFor="priority" className="text-sm font-medium">
						Priority
					</label>
					{canEdit ? (
						<Select
							value={priority}
							onValueChange={(value: "low" | "medium" | "high") => setPriority(value)}
						>
							<SelectTrigger>
								<SelectValue placeholder="Select priority" />
							</SelectTrigger>
							<SelectContent>
								<SelectItem value="low">Low</SelectItem>
								<SelectItem value="medium">Medium</SelectItem>
								<SelectItem value="high">High</SelectItem>
							</SelectContent>
						</Select>
					) : (
						<div className="h-12 px-4 justify-between bg-background border rounded-md flex items-center gap-2">
							<Flag className="w-4 h-4" />
							<span
								className={`px-2 py-0.5 rounded-md text-xs font-medium ${priorityColor}`}
							>
								{priorityLabel}
							</span>
						</div>
					)}
				</div>
			</div>

			<div className="space-y-2">
				<label className="text-sm font-medium">Due Date</label>
				{canEdit ? (
					<Popover>
						<PopoverTrigger asChild>
							<Button
								variant="outline"
								className={cn(
									"w-full justify-start text-left font-normal",
									!dueDate && "text-muted-foreground"
								)}
							>
								<CalendarIcon className="mr-2 h-4 w-4" />
								{dueDate ? format(dueDate, "PPP") : "Pick a date"}
							</Button>
						</PopoverTrigger>
						<PopoverContent className="w-auto p-0" align="start">
							<Calendar mode="single" selected={dueDate} onSelect={setDueDate} />
							{dueDate && (
								<div className="p-3 border-t">
									<Button
										variant="ghost"
										size="sm"
										className="w-full"
										onClick={() => setDueDate(undefined)}
									>
										<X className="w-4 h-4 mr-2" />
										Clear date
									</Button>
								</div>
							)}
						</PopoverContent>
					</Popover>
				) : (
					<div className="p-3 bg-background border rounded-md flex items-center text-foreground">
						<CalendarIcon className="mr-2 h-4 w-4 text-muted-foreground" />
						{dueDate ? (
							format(dueDate, "PPP")
						) : (
							<span className="text-muted-foreground">No due date</span>
						)}
					</div>
				)}
			</div>
			{canEdit && (
				<div className="flex gap-2 pt-4">
					{isEditing && (
						<Button
							type="button"
							variant="destructive"
							onClick={onDelete}
							disabled={isSubmitting}
						>
							<Trash2 className="w-4 h-4 mr-2" />
							Delete
						</Button>
					)}
					<Button type="submit" disabled={isSubmitting || !title.trim() || !columnId}>
						{isSubmitting ? "Saving..." : isEditing ? "Update Task" : "Create Task"}
					</Button>
				</div>
			)}
		</form>
	);
}
