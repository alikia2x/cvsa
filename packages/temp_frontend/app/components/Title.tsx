import { useEffect } from "react";
import { useLocation } from "react-router";

export const Title = ({ title }: { title: string }) => {
	const location = useLocation();
	useEffect(() => {
		document.title = title + " - 中V档案馆";
	}, [title, location]);

	return null;
};
