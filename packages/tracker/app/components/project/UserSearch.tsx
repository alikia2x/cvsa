import { Search, UserPlus, X } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
	Dialog,
	DialogContent,
	DialogDescription,
	DialogHeader,
	DialogTitle,
	DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export function UserSearchModal({
	availableUsers,
	projectId,
}: {
	availableUsers: Array<{ id: string; username: string }>;
	projectId: string;
}) {
	const [isOpen, setIsOpen] = useState(false);
	const [searchTerm, setSearchTerm] = useState("");
	const [selectedUserId, setSelectedUserId] = useState<string>("");
	const [canEdit, setCanEdit] = useState(false);

	const filteredUsers = availableUsers.filter((user) =>
		user.username.toLowerCase().includes(searchTerm.toLowerCase())
	);

	const handleSubmit = async (e: React.FormEvent) => {
		e.preventDefault();
		if (!selectedUserId) return;

		const formData = new FormData();
		formData.append("intent", "addUser");
		formData.append("userId", selectedUserId);
		formData.append("canEdit", canEdit ? "on" : "");

		try {
			const response = await fetch(`/project/${projectId}/settings`, {
				body: formData,
				method: "POST",
			});

			if (response.ok) {
				setIsOpen(false);
				setSelectedUserId("");
				setSearchTerm("");
				setCanEdit(false);
				// Reload the page to show updated permissions
				window.location.reload();
			} else {
				const result = await response.json();
				alert(result.error || "Failed to add user");
			}
		} catch (error) {
			console.error("Error adding user:", error);
			alert("Failed to add user");
		}
	};

	return (
		<Dialog open={isOpen} onOpenChange={setIsOpen}>
			<DialogTrigger asChild>
				<Button variant="outline" className="w-80 justify-start">
					<Search className="size-4 mr-2" />
					{selectedUserId
						? availableUsers.find((u) => u.id === selectedUserId)?.username
						: "Search and select user..."}
				</Button>
			</DialogTrigger>
			<DialogContent className="sm:max-w-md">
				<DialogHeader>
					<DialogTitle>Add User to Project</DialogTitle>
					<DialogDescription>Search for a user to add to this project</DialogDescription>
				</DialogHeader>

				<form onSubmit={handleSubmit} className="space-y-4">
					<div className="space-y-2">
						<Label htmlFor="search">Search Users</Label>
						<div className="flex gap-2">
							<Input
								id="search"
								type="text"
								placeholder="Type to search users..."
								value={searchTerm}
								onChange={(e) => setSearchTerm(e.target.value)}
								className="flex-1"
							/>
							{searchTerm && (
								<Button
									type="button"
									variant="ghost"
									size="icon"
									onClick={() => setSearchTerm("")}
								>
									<X className="size-4" />
								</Button>
							)}
						</div>
					</div>

					{filteredUsers.length > 0 && (
						<div className="space-y-2 max-h-60 overflow-y-auto">
							<Label>Select User</Label>
							<div className="space-y-1">
								{filteredUsers.map((user) => (
									<div
										key={user.id}
										className={`flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-colors ${
											selectedUserId === user.id
												? "bg-primary/10 border-primary"
												: "hover:bg-muted"
										}`}
										onClick={() => setSelectedUserId(user.id)}
									>
										<div className="flex-1">
											<div className="font-medium">{user.username}</div>
										</div>
										{selectedUserId === user.id && (
											<div className="text-primary">✓</div>
										)}
									</div>
								))}
							</div>
						</div>
					)}

					{filteredUsers.length === 0 && searchTerm && (
						<div className="text-center text-muted-foreground py-4">
							No users found matching "{searchTerm}"
						</div>
					)}

					{selectedUserId && (
						<div className="space-y-4 pt-4 border-t">
							<div className="flex items-center gap-2">
								<Checkbox
									id="canEdit"
									checked={canEdit}
									onCheckedChange={(checked) => setCanEdit(checked === true)}
								/>
								<Label htmlFor="canEdit">Can Edit</Label>
							</div>

							<div className="flex gap-2">
								<Button type="submit" className="flex-1" disabled={!selectedUserId}>
									<UserPlus className="size-4 mr-2" />
									Add User
								</Button>
								<Button
									type="button"
									variant="outline"
									onClick={() => setIsOpen(false)}
								>
									Cancel
								</Button>
							</div>
						</div>
					)}
				</form>
			</DialogContent>
		</Dialog>
	);
}
