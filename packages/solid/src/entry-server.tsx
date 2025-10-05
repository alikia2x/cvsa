// @refresh reload
import { createHandler, StartServer } from "@solidjs/start/server";
import { MetaProvider } from "@solidjs/meta";
import { RequestContextProvider } from "~/components/requestContext";

export default createHandler(() => (
	<RequestContextProvider>
		<MetaProvider>
			<StartServer
				document={({ assets, children, scripts }) => (
					<html lang="zh-CN">
						<head>
							<meta charset="utf-8" />
							<meta name="viewport" content="width=device-width, initial-scale=1" />
							{assets}
						</head>
						<body>
							<div id="app">{children}</div>
							<div id="modal"></div>
							{scripts}
						</body>
					</html>
				)}
			/>
		</MetaProvider>
	</RequestContextProvider>
));
