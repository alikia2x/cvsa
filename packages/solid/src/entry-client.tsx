// @refresh reload
import { mount, StartClient } from "@solidjs/start/client";
import { RequestContextProvider } from "./components/requestContext";
import { MetaProvider } from "@solidjs/meta";

mount(
	() => (
		<RequestContextProvider>
			<MetaProvider>
				<StartClient />
			</MetaProvider>
		</RequestContextProvider>
	),
	document.getElementById("app")!
);
