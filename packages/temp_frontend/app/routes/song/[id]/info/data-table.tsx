"use client";

import * as React from "react";
import {
	flexRender,
	getCoreRowModel,
	getPaginationRowModel,
	getSortedRowModel,
	useReactTable,
	type ColumnDef,
	type SortingState,
} from "@tanstack/react-table";

import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

interface DataTableProps<TData, TValue> {
	columns: ColumnDef<TData, TValue>[];
	data: TData[];
}

export function DataTable<TData, TValue>({ columns, data }: DataTableProps<TData, TValue>) {
	const [sorting, setSorting] = React.useState<SortingState>([]);

	const table = useReactTable({
		data,
		columns,
		getCoreRowModel: getCoreRowModel(),
		getPaginationRowModel: getPaginationRowModel(),
		onSortingChange: setSorting,
		getSortedRowModel: getSortedRowModel(),
		state: {
			sorting,
		},
	});

	return (
		<div className="max-w-[calc(100vw-1.5rem)]">
			<div className="rounded-md border">
				<Table>
					<TableHeader>
						{table.getHeaderGroups().map((headerGroup) => (
							<TableRow key={headerGroup.id}>
								{headerGroup.headers.map((header) => {
									return (
										<TableHead key={header.id}>
											{header.isPlaceholder
												? null
												: flexRender(header.column.columnDef.header, header.getContext())}
										</TableHead>
									);
								})}
							</TableRow>
						))}
					</TableHeader>
					<TableBody>
						{table.getRowModel().rows?.length ? (
							table.getRowModel().rows.map((row) => (
								<TableRow key={row.id} data-state={row.getIsSelected() && "selected"}>
									{row.getVisibleCells().map((cell) => (
										<TableCell key={cell.id} className="stat-num">
											{flexRender(cell.column.columnDef.cell, cell.getContext())}
										</TableCell>
									))}
								</TableRow>
							))
						) : (
							<TableRow>
								<TableCell colSpan={columns.length} className="h-24 text-center">
									暂无数据
								</TableCell>
							</TableRow>
						)}
					</TableBody>
				</Table>
			</div>
			<div className="flex items-center justify-between py-4 stat-num">
				<div className="text-sm text-muted-foreground">
					{table.getState().pagination.pageIndex + 1} / {table.getPageCount()} 页
				</div>
				<div className="flex items-center space-x-2">
					<Button
						variant="outline"
						size="sm"
						onClick={() => table.previousPage()}
						disabled={!table.getCanPreviousPage()}
					>
						上一页
					</Button>
					<div className="flex items-center space-x-2">
						<span className="text-sm">跳转到</span>
						<Input
							type="number"
							min="1"
							max={table.getPageCount()}
							defaultValue={table.getState().pagination.pageIndex + 1}
							className="w-15 h-8 text-center"
							onKeyDown={(e) => {
								if (e.key === "Enter") {
									const page = e.currentTarget.value ? Number(e.currentTarget.value) - 1 : 0;
									table.setPageIndex(Math.max(0, Math.min(page, table.getPageCount() - 1)));
								}
							}}
							onBlur={(e) => {
								const page = e.currentTarget.value ? Number(e.currentTarget.value) - 1 : 0;
								table.setPageIndex(Math.max(0, Math.min(page, table.getPageCount() - 1)));
							}}
						/>
						<span className="text-sm">页</span>
					</div>
					<Button
						variant="outline"
						size="sm"
						onClick={() => table.nextPage()}
						disabled={!table.getCanNextPage()}
					>
						下一页
					</Button>
				</div>
			</div>
		</div>
	);
}
