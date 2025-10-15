export default function Layout({ children }: { children: React.ReactNode }) {
	return (
		<div className="min-h-screen bg-background">
			<div className="container mx-auto px-6 my-16 xl:px-15">{children}</div>
		</div>
	);
}
