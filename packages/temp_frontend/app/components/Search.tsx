import { SearchIcon } from "@/components/icons/search";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { useState } from "react";
import { useNavigate } from "react-router";

interface SearchBoxProps extends React.ComponentProps<"div"> {
	query: string;
	setQuery: (q: string) => void;
	onSearch: () => void;
}

export function SearchBox({ query = "", setQuery, onSearch, className, ...rest }: SearchBoxProps) {
	return (
		<div className={"flex h-12 gap-2 relative w-full " + (className ? ` ${className}` : "")} {...rest}>
			<Input
				className="h-full pl-5 pr-12 rounded-full"
				type="search"
				placeholder="搜索"
				value={query}
				autoComplete="off"
				autoCorrect="off"
				onChange={(e) => setQuery(e.target.value)}
				onKeyDown={(e) => {
					if (e.key === "Enter") {
						onSearch();
					}
				}}
				id="search-input"
			/>
			<Button variant="ghost" className="absolute rounded-full size-10 top-1 right-1" onClick={onSearch}>
				<SearchIcon className="size-6" />
			</Button>
		</div>
	);
}

export function Search(props: React.ComponentProps<"div">) {
	const [query, setQuery] = useState("");
	let navigate = useNavigate();
	return <SearchBox query={query} setQuery={setQuery} onSearch={() => navigate(`/search?q=${query}`)} {...props} />;
}
