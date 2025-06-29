// @refresh reload
import { mount, StartClient } from "@solidjs/start/client";
import { RequestContextProvider } from "./components/requestContext";

mount(
	() => (
		<RequestContextProvider>
			<StartClient />
		</RequestContextProvider>
	),
	document.getElementById("app")!
);
