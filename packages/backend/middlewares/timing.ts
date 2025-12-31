import { type Context, Elysia, type MapResponse, type TraceEvent, type TraceProcess } from "elysia";

type MaybePromise<T> = T | Promise<T>;

class TimeLogger {
	private durations: Map<string, number>;
	private totalStartTime: number | null;

	constructor() {
		this.durations = new Map();
		this.totalStartTime = null;
	}

	getCompletedDurations() {
		return Array.from(this.durations.entries()).map(([name, duration]) => ({
			duration,
			name,
		}));
	}

	startTotal(): void {
		this.totalStartTime = performance.now();
	}

	endTotal(): number | null {
		if (this.totalStartTime === null) return null;
		return performance.now() - this.totalStartTime;
	}
}

export interface ServerTimingOptions {
	/**
	 * Should Elysia report data back to client via 'Server-Sent-Event'
	 */
	report?: boolean;
	/**
	 * Allow Server Timing to log specified life-cycle events
	 */
	trace?: {
		/**
		 * Capture duration from request
		 *
		 * @default true
		 */
		request?: boolean;
		/**
		 * Capture duration from parse
		 *
		 * @default true
		 */
		parse?: boolean;
		/**
		 * Capture duration from transform
		 *
		 * @default true
		 */
		transform?: boolean;
		/**
		 * Capture duration from beforeHandle
		 *
		 * @default true
		 */
		beforeHandle?: boolean;
		/**
		 * Capture duration from handle
		 *
		 * @default true
		 */
		handle?: boolean;
		/**
		 * Capture duration from afterHandle
		 *
		 * @default true
		 */
		afterHandle?: boolean;
		/**
		 * Capture duration from mapResponse
		 *
		 * @default true
		 */
		error?: boolean;
		/**
		 * Capture duration from mapResponse
		 *
		 * @default true
		 */
		mapResponse?: boolean;
		/**
		 * Capture total duration from start to finish
		 *
		 * @default true
		 */
		total?: boolean;
	};
	/**
	 * Determine whether Server Timing should be enabled
	 *
	 * @default NODE_ENV !== 'production'
	 */
	enabled?: boolean;
	/**
	 * A condition whether server timing should be log
	 *
	 * @default undefined
	 */
	allow?: MaybePromise<boolean> | ((context: Omit<Context, "path">) => MaybePromise<boolean>);
	/**
	 * A custom mapResponse provided by user
	 *
	 * @default undefined
	 */
	mapResponse?: MapResponse;
}

const getLabel = (
	event: TraceEvent,
	listener: (callback: (process: TraceProcess<"begin">) => unknown) => unknown,
	write: (value: string) => void
) => {
	listener(async ({ onStop, onEvent, total }) => {
		let label = "";

		if (total === 0) return;

		await onEvent(({ name, index, onStop }) => {
			onStop(({ elapsed }) => {
				label += `${event}.${index}.${name || "anon"};dur=${elapsed},`;
			});
		});

		await onStop(({ elapsed }) => {
			label += `${event};dur=${elapsed},`;

			write(label);
		});
	});
};

export const serverTiming = ({
	allow,
	enabled = process.env.NODE_ENV !== "production",
	trace: {
		request: traceRequest = true,
		parse: traceParse = true,
		transform: traceTransform = true,
		beforeHandle: traceBeforeHandle = true,
		handle: traceHandle = true,
		afterHandle: traceAfterHandle = true,
		error: traceError = true,
		mapResponse: traceMapResponse = true,
		total: traceTotal = true,
	} = {},
}: ServerTimingOptions = {}) => {
	return new Elysia()
		.decorate("timeLog", new TimeLogger())
		.trace(
			{ as: "global" },
			async ({
				onRequest,
				onParse,
				onTransform,
				onBeforeHandle,
				onHandle,
				onAfterHandle,
				onMapResponse,
				onError,
				set,
				context,
			}) => {
				if (!enabled) return;
				let label = "";

				const write = (nextValue: string) => {
					label += nextValue;
				};

				await onRequest(() => {
					context.timeLog.startTotal();
				});

				if (traceRequest) getLabel("request", onRequest, write);
				if (traceParse) getLabel("parse", onParse, write);
				if (traceTransform) getLabel("transform", onTransform, write);
				if (traceBeforeHandle) getLabel("beforeHandle", onBeforeHandle, write);
				if (traceAfterHandle) getLabel("afterHandle", onAfterHandle, write);
				if (traceError) getLabel("error", onError, write);
				if (traceMapResponse) getLabel("mapResponse", onMapResponse, write);

				if (traceHandle)
					await onHandle(({ name, onStop }) => {
						onStop(({ elapsed }) => {
							label += `handle.${name};dur=${elapsed},`;
						});
					});

				await onMapResponse(({ onStop }) => {
					onStop(async () => {
						const completedDurations = context.timeLog.getCompletedDurations();
						if (completedDurations.length > 0) {
							label += `${completedDurations
								.map(({ name, duration }) => `${name};dur=${duration}`)
								.join(", ")},`;
						}
						const elapsed = context.timeLog.endTotal();

						let allowed = allow;
						if (allowed instanceof Promise) allowed = await allowed;

						if (traceTotal) label += `total;dur=${elapsed}`;
						else label = label.slice(0, -1);

						// ? Must wait until request is reported
						switch (typeof allowed) {
							case "boolean":
								if (!allowed) delete set.headers["Server-Timing"];

								set.headers["Server-Timing"] = label;

								break;

							case "function":
								if ((await allowed(context)) === false)
									delete set.headers["Server-Timing"];

								set.headers["Server-Timing"] = label;

								break;

							default:
								set.headers["Server-Timing"] = label;
						}
					});
				});
			}
		);
};

export default serverTiming;
