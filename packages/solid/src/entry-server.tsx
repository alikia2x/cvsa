// @refresh reload
import { createHandler, StartServer } from "@solidjs/start/server";
import { MetaProvider } from "@solidjs/meta";
import { RequestContextProvider } from "~/components/requestContext";

export default createHandler(() => (
	<RequestContextProvider>
		<StartServer
			document={({ assets, children, scripts }) => (

				<html lang="en">
					<head>
						<meta charset="utf-8" />
						<meta name="viewport" content="width=device-width, initial-scale=1" />
						{/*<link rel="icon" href="/favicon.ico" />*/}
						<MetaProvider></MetaProvider>
						<title>中V档案馆</title>
						{assets}
					</head>
					<body>
						<div id="app">
							{children}
						</div>
						{scripts}
					</body>
				</html>

			)}
		/>
	</RequestContextProvider>
));
