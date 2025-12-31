"use client";

import type { ColumnDef } from "@tanstack/react-table";
import { ArrowUpDown } from "lucide-react";
import { formatDateTime } from "@/components/SearchResults";
import { Button } from "@/components/ui/button";

export type Snapshot = {
	createdAt: string;
	views: number;
	likes: number;
	favorites: number;
	coins: number;
	danmakus: number;
	shares: number;
};

export const columns: ColumnDef<Snapshot>[] = [
	{
		accessorKey: "createdAt",
		cell: ({ row }) => {
			const createdAt = row.getValue("createdAt") as string;
			return <div>{formatDateTime(new Date(createdAt))}</div>;
		},
		header: ({ column }) => {
			return (
				<Button
					variant="ghost"
					onClick={() => column.toggleSorting(column.getIsSorted() === "asc")}
				>
					时间
					<ArrowUpDown className="ml-2 h-4 w-4" />
				</Button>
			);
		},
	},
	{
		accessorKey: "views",
		cell: ({ row }) => {
			const views = row.getValue("views") as number;
			return <div>{views.toLocaleString()}</div>;
		},
		header: "播放",
	},
	{
		accessorKey: "likes",
		cell: ({ row }) => {
			const likes = row.getValue("likes") as number;
			return <div>{likes.toLocaleString()}</div>;
		},
		header: "点赞",
	},
	{
		accessorKey: "favorites",
		cell: ({ row }) => {
			const favorites = row.getValue("favorites") as number;
			return <div>{favorites.toLocaleString()}</div>;
		},
		header: "收藏",
	},
	{
		accessorKey: "coins",
		cell: ({ row }) => {
			const coins = row.getValue("coins") as number;
			return <div>{coins.toLocaleString()}</div>;
		},
		header: "硬币",
	},
	{
		accessorKey: "danmakus",
		cell: ({ row }) => {
			const danmakus = row.getValue("danmakus") as number;
			return <div>{danmakus.toLocaleString()}</div>;
		},
		header: "弹幕",
	},
	{
		accessorKey: "shares",
		cell: ({ row }) => {
			const shares = row.getValue("shares") as number;
			return <div>{shares.toLocaleString()}</div>;
		},
		header: "转发",
	},
];
