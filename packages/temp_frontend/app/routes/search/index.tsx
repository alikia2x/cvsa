import { treaty } from "@elysiajs/eden";
import type { App } from "@elysia/src";
import { useEffect, useState } from "react";
import { Skeleton } from "@/components/ui/skeleton";
import { Error } from "@/components/Error";
import { useSearchParams } from "react-router";
import { SearchBox } from "@/components/Search";
import { SearchResults } from "@/components/SearchResults";
import { Title } from "@/components/Title";
import { Layout, LayoutWithouSearch } from "@/components/Layout";

const app = treaty<App>(import.meta.env.VITE_API_URL!);

export type SearchResult = Awaited<ReturnType<typeof app.search.result.get>>["data"];
type SearchError = Awaited<ReturnType<typeof app.search.result.get>>["error"];

const Search = ({
	query,
	setQuery,
	onSearch,
	...props
}: {
	query: string;
	setQuery: (value: string) => void;
	onSearch: () => void;
} & React.ComponentProps<"div">) => <SearchBox query={query} setQuery={setQuery} onSearch={onSearch} {...props} />;

export default function SearchResult() {
	const [searchParams, setSearchParams] = useSearchParams();
	const [query, setQuery] = useState(searchParams.get("q") || "");
	const [data, setData] = useState<SearchResult | null>(null);
	const [error, setError] = useState<SearchError | null>(null);

	const search = async () => {
		setData(null);
		setError(null);
		const { data, error } = await app.search.result.get({
			query: {
				query: searchParams.get("q") || "",
			},
		});
		if (error) {
			console.log(error);
			setError(error);
			return;
		}
		setData(data);
	};

	useEffect(() => {
		if (!searchParams.get("q")) return;
		search();
		setQuery(searchParams.get("q") || "");
	}, [searchParams]);

	const handleSearch = () => setSearchParams({ q: query });

	if (!searchParams.get("q")) {
		return (
			<LayoutWithouSearch>
				<Search query={query} setQuery={setQuery} onSearch={handleSearch} />
			</LayoutWithouSearch>
		);
	}

	if (!data && !error) {
		return (
			<LayoutWithouSearch>
				<Title title={searchParams.get("q") || "搜索"} />
				<Search query={query} setQuery={setQuery} onSearch={handleSearch} className="mb-6" />
				<Skeleton className="w-full h-24 mb-2" />
			</LayoutWithouSearch>
		);
	}

	if (error) {
		return <Error error={error} />;
	}

	return (
		<LayoutWithouSearch>
			<Title title={searchParams.get("q") || ""} />
			<Search query={query} setQuery={setQuery} onSearch={handleSearch} className="mb-6" />
			<SearchResults results={data} />
		</LayoutWithouSearch>
	);
}
