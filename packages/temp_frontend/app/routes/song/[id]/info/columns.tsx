"use client";

import type { ColumnDef } from "@tanstack/react-table";
import { ArrowUpDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import { formatDateTime } from "@/components/SearchResults";

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
		header: ({ column }) => {
			return (
				<Button variant="ghost" onClick={() => column.toggleSorting(column.getIsSorted() === "asc")}>
					时间
					<ArrowUpDown className="ml-2 h-4 w-4" />
				</Button>
			);
		},
		cell: ({ row }) => {
			const createdAt = row.getValue("createdAt") as string;
			return <div>{formatDateTime(new Date(createdAt))}</div>;
		},
	},
	{
		accessorKey: "views",
		header: "播放",
		cell: ({ row }) => {
			const views = row.getValue("views") as number;
			return <div>{views.toLocaleString()}</div>;
		},
	},
	{
		accessorKey: "likes",
		header: "点赞",
		cell: ({ row }) => {
			const likes = row.getValue("likes") as number;
			return <div>{likes.toLocaleString()}</div>;
		},
	},
	{
		accessorKey: "favorites",
		header: "收藏",
		cell: ({ row }) => {
			const favorites = row.getValue("favorites") as number;
			return <div>{favorites.toLocaleString()}</div>;
		},
	},
	{
		accessorKey: "coins",
		header: "硬币",
		cell: ({ row }) => {
			const coins = row.getValue("coins") as number;
			return <div>{coins.toLocaleString()}</div>;
		},
	},
	{
		accessorKey: "danmakus",
		header: "弹幕",
		cell: ({ row }) => {
			const danmakus = row.getValue("danmakus") as number;
			return <div>{danmakus.toLocaleString()}</div>;
		},
	},
	{
		accessorKey: "shares",
		header: "转发",
		cell: ({ row }) => {
			const shares = row.getValue("shares") as number;
			return <div>{shares.toLocaleString()}</div>;
		},
	},
];
