import { Trash2 } from "lucide-react";
import { use, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from "@/components/ui/select";

interface ColumnDialogProps {
	open: boolean;
	onOpenChange: (open: boolean) => void;
	onSubmit: (data: { name: string; position?: number }) => Promise<void>;
	onDelete?: () => void;
	columns: number;
	initialData?: {
		name?: string;
		position?: number;
	};
	isEditing?: boolean;
}

export function ColumnDialog({
	open,
	onOpenChange,
	onSubmit,
	onDelete,
	initialData,
	columns,
	isEditing = false,
}: ColumnDialogProps) {
	const [name, setName] = useState(initialData?.name || "");
	const [position, setPosition] = useState(initialData?.position?.toString() || "0");
	const [isSubmitting, setIsSubmitting] = useState(false);

	useEffect(() => {
		if (initialData?.name) {
			setName(initialData.name);
		}
		if (initialData?.position !== undefined) {
			setPosition(initialData.position.toString());
		}
	}, [initialData?.name, initialData?.position, setName]);

	const handleSubmit = async (e: React.FormEvent) => {
		e.preventDefault();
		if (!name.trim()) return;

		setIsSubmitting(true);
		try {
			await onSubmit({
				name: name.trim(),
				position: parseInt(position) || 0,
			});
			onOpenChange(false);
			// Reset form
			setName("");
		} finally {
			setIsSubmitting(false);
		}
	};

	const handleDelete = () => {
		if (!onDelete) {
			return;
		}
		setIsSubmitting(true);
		try {
			onDelete();
			onOpenChange(false);
		} finally {
			setIsSubmitting(false);
		}
	};

	return (
		<Dialog open={open} onOpenChange={onOpenChange}>
			<DialogContent className="sm:max-w-[425px]">
				<DialogHeader>
					<DialogTitle>{isEditing ? "Edit Column" : "Create New Column"}</DialogTitle>
				</DialogHeader>
				<form onSubmit={handleSubmit} className="space-y-4">
					<div className="space-y-2">
						<Label htmlFor="name">Column Name</Label>
						<Input
							id="name"
							value={name}
							onChange={(e) => setName(e.target.value)}
							placeholder="Enter column name"
							required
						/>
					</div>
					{isEditing && (
						<div className="space-y-2">
							<Label htmlFor="position">Position</Label>
							<Select value={position} onValueChange={setPosition}>
								<SelectTrigger>
									<SelectValue placeholder="Select column" />
								</SelectTrigger>
								<SelectContent>
									{Array.from({ length: columns }, (_, index) => (
										<SelectItem key={index} value={index.toString()}>
											{index + 1}
										</SelectItem>
									))}
								</SelectContent>
							</Select>
						</div>
					)}

					<div className="max-sm: flex justify-between pt-4">
						<div>
							{isEditing && onDelete && (
								<Button
									type="button"
									variant="destructive"
									onClick={handleDelete}
									disabled={isSubmitting}
								>
									<Trash2 className="w-4 h-4 mr-2" />
									Delete
								</Button>
							)}
						</div>
						<div className="flex gap-2">
							<Button type="submit" disabled={isSubmitting || !name.trim()}>
								{isSubmitting
									? "Saving..."
									: isEditing
										? "Save Changes"
										: "Create Column"}
							</Button>
						</div>
					</div>
				</form>
			</DialogContent>
		</Dialog>
	);
}
