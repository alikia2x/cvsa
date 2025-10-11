import { useState, useRef, useMemo, useCallback, useEffect } from "react";
import { HOUR, DAY, WEEK } from "@core/lib";

export type TimeRange = "6h" | "1d" | "7d" | "30d" | "3mo" | "6mo" | "1y" | "all";

const TIME_RANGES = {
	"6h": 6 * HOUR,
	"1d": DAY,
	"7d": 7 * DAY,
	"30d": 30 * DAY,
	"3mo": 90 * DAY,
	"6mo": 180 * DAY,
	"1y": 365 * DAY,
	all: (maxTime: number, minTime: number) => maxTime - minTime,
};

function getWindowSize(timeRange: TimeRange, { maxTime, minTime }: { maxTime: number; minTime: number }) {
	const val = TIME_RANGES[timeRange];
	if (typeof val === "function") return val(maxTime, minTime);
	return val ?? 7 * DAY;
}

const leftPad = 10;
const rightPad = 10;

function calculateXAxisAverageAcceleration(lastPointerPositions: { x: number; time: number }[]) {
	if (lastPointerPositions.length < 3) {
		return 0;
	}

	const accelerations = [];

	for (let i = 2; i < lastPointerPositions.length; i++) {
		const point1 = lastPointerPositions[i - 2];
		const point2 = lastPointerPositions[i - 1];
		const point3 = lastPointerPositions[i];

		const deltaTime1 = (point2.time - point1.time) / 1000;
		const deltaTime2 = (point3.time - point2.time) / 1000;
		if (deltaTime1 === 0 || deltaTime2 === 0) {
			continue;
		}

		const velocity1 = (point2.x - point1.x) / deltaTime1;
		const velocity2 = (point3.x - point2.x) / deltaTime2;

		const averageDeltaTime = (deltaTime1 + deltaTime2) / 2;
		const acceleration = (velocity2 - velocity1) / averageDeltaTime;

		accelerations.push(acceleration);
	}

	if (accelerations.length === 0) {
		return 0;
	}

	const sum = accelerations.reduce((acc, val) => acc + val, 0);
	const averageAcceleration = sum / accelerations.length;

	return averageAcceleration;
}

interface CharProps extends React.HTMLAttributes<HTMLDivElement> {
	data: { timestamp: number; value: number }[];
	width?: string;
	height?: number;
	accentColor?: string;
	smoothInterpolation?: boolean;
	timeRange?: "6h" | "1d" | "7d" | "30d" | "3mo" | "6mo" | "1y" | "all";
	outside?: boolean;
	ref?: React.Ref<HTMLDivElement>;
	setCurrentData?: (data: string) => void;
	setCurrentDate?: (date: string) => void;
}

export const TimeSeriesChart = ({
	data = [],
	width = "100%",
	height = 300,
	accentColor = "#007AFF",
	smoothInterpolation = true,
	timeRange = "7d",
	outside = false,
	ref = null,
	setCurrentData,
	setCurrentDate,
	...rest
}: CharProps) => {
	const [yLabelWidth, setYLabelWidth] = useState(0);
	const widthProbeRef = useRef<SVGTextElement | null>(null);
	const sortedData = useMemo(() => [...data].sort((a, b) => a.timestamp - b.timestamp), [data]);
	const globalMaxValue = useMemo(
		() => (sortedData.length > 0 ? sortedData[sortedData.length - 1].value : 0),
		[sortedData],
	);
	// Y-axis range state for lazy adjustment
	const [yAxisRange, setYAxisRange] = useState({ min: 0, max: 0 });
	const [yAxisAnimation, setYAxisAnimation] = useState<{
		isAnimating: boolean;
		startMin: number;
		startMax: number;
		targetMin: number;
		targetMax: number;
		startTimestamp: number;
	} | null>(null);
	const yAxisAnimationRef = useRef<number | null>(null);
	const svgRef = useRef<SVGSVGElement>(null);
	const containerRef = useRef<HTMLDivElement>(null);
	const [currentPosition, setCurrentPosition] = useState<{
		x: number;
		y: number;
		data: { timestamp: number; value: number } | null;
	} | null>(null);
	const [dragStartX, setDragStartX] = useState(0);
	const [timeWindow, setTimeWindow] = useState({ startTime: 0, endTime: 0 });
	const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
	const [longPressTimer, setLongPressTimer] = useState<NodeJS.Timeout | null>(null);
	const animationRef = useRef<number | null>(null);
	const [lastPointerPositions, setLastPointerPositions] = useState<{ x: number; y: number; time: number }[]>([]);
	const [pointerStartPosition, setPointerStartPosition] = useState<{ x: number; y: number; time: number } | null>(
		null,
	);

	const visibleData = useMemo(() => {
		if (data.length === 0 || timeWindow.startTime === 0) return [];

		let firstIndex = -1;
		let lastIndex = -1;

		for (let i = 0; i < sortedData.length; i++) {
			const t = sortedData[i].timestamp;
			if (firstIndex === -1 && t >= timeWindow.startTime) firstIndex = i;
			if (t <= timeWindow.endTime) lastIndex = i;
			if (t > timeWindow.endTime && lastIndex !== -1) break;
		}

		if (firstIndex === -1 && lastIndex === -1) return [];

		const from = Math.max(0, (firstIndex === -1 ? 0 : firstIndex) - leftPad);
		const to = Math.min(sortedData.length - 1, (lastIndex === -1 ? sortedData.length - 1 : lastIndex) + rightPad);

		if (from > to) return [];

		return sortedData.slice(from, to + 1);
	}, [data, timeWindow, sortedData]);

	useEffect(() => {
		if (!setCurrentData || !setCurrentDate) return;

		const haveCurrent =
			currentPosition && currentPosition.data && currentPosition.data.timestamp && currentPosition.data.value;

		if (haveCurrent) {
			const { timestamp, value } = currentPosition.data!;
			const month = new Date(timestamp).getMonth() + 1;
			const day = new Date(timestamp).getDate();
			const hour = new Date(timestamp).getHours().toString().padStart(2, "0");
			const minute = new Date(timestamp).getMinutes().toString().padStart(2, "0");
			setCurrentData(value.toLocaleString());
			setCurrentDate(`${month}月${day}日 ${hour}:${minute}`);
			return;
		}

		if (Array.isArray(sortedData) && sortedData.length > 0) {
			const oldest = visibleData.length ? visibleData[leftPad - 1] : sortedData[0];
			const newest = visibleData.length
				? visibleData[visibleData.length - rightPad]
				: sortedData[sortedData.length - 1];
			const increment = newest.value - oldest.value;
			const year = new Date(oldest.timestamp).getFullYear();
			const month = new Date(oldest.timestamp).getMonth() + 1;
			const day = new Date(oldest.timestamp).getDate();
			const hour = new Date(oldest.timestamp).getHours().toString().padStart(2, "0");
			const minute = new Date(oldest.timestamp).getMinutes().toString().padStart(2, "0");
			const newestYear = new Date(newest.timestamp).getFullYear();
			const newestMonth = new Date(newest.timestamp).getMonth() + 1;
			const newestDay = new Date(newest.timestamp).getDate();
			const newestHour = new Date(newest.timestamp).getHours().toString().padStart(2, "0");
			const newestMinute = new Date(newest.timestamp).getMinutes().toString().padStart(2, "0");
			const timeRange = newest.timestamp - oldest.timestamp;
			if (year !== newestYear) {
				setCurrentDate(`${year}年${month}月${day}日–${newestYear}年${newestMonth}月${newestDay}日`);
			} else if (month !== newestMonth || timeRange > DAY) {
				setCurrentDate(`${year}年 ${month}月${day}日–${newestMonth}月${newestDay}日`);
			} else if (day !== newestDay) {
				setCurrentDate(
					`${month}月${day}日 ${hour}:${minute}–${newestMonth}月${newestDay}日 ${newestHour}:${newestMinute}`,
				);
			} else {
				setCurrentDate(`${month}月${day}日 ${hour}:${minute}–${newestHour}:${newestMinute}`);
			}
			setCurrentData("+" + increment.toLocaleString());
		}
	}, [
		currentPosition,
		currentPosition?.data?.timestamp,
		currentPosition?.data?.value,
		sortedData,
		setCurrentData,
		setCurrentDate,
		visibleData,
	]);

	useEffect(() => {
		if (!widthProbeRef.current) return;
		const { width } = widthProbeRef.current.getBoundingClientRect();
		setYLabelWidth(width);
	}, [widthProbeRef.current]);

	useEffect(() => {
		const updateDimensions = () => {
			if (containerRef.current) {
				const { width: containerWidth, height: containerHeight } = containerRef.current.getBoundingClientRect();
				setDimensions({ width: containerWidth, height: containerHeight });
			}
		};

		updateDimensions();
		window.addEventListener("resize", updateDimensions);
		return () => window.removeEventListener("resize", updateDimensions);
	}, []);

	useEffect(() => {
		if (svgRef.current) {
			const { width: svgWidth, height: svgHeight } = svgRef.current.getBoundingClientRect();
			setDimensions({ width: svgWidth, height: svgHeight });
		}
	}, [svgRef.current?.getBoundingClientRect().height]);

	useEffect(() => {
		if (!outside) return;
		setCurrentPosition(null);
	}, [outside]);

	useEffect(() => {
		if (data.length === 0) return;

		const minTime = sortedData[0].timestamp;
		const maxTime = sortedData[sortedData.length - 1].timestamp;

		const windowSize = getWindowSize(timeRange, { maxTime, minTime });

		const initialEndTime = maxTime;
		const initialStartTime = Math.max(minTime, initialEndTime - windowSize);

		setTimeWindow({
			startTime: initialStartTime,
			endTime: initialEndTime,
		});

		if (visibleValues.length > 0) {
			const targetMin = Math.max(0, visibleMin);
			const targetMax = visibleMax;

			const buffer = (targetMax - targetMin) * 0.3;

			// Align to nice numbers
			const alignedRange = alignRangeToNiceNumbers(targetMin - buffer, targetMax + buffer);

			setYAxisRange({
				min: alignedRange.min,
				max: alignedRange.max,
			});
		}
	}, [data, timeRange]);

	// Cleanup animations on unmount
	useEffect(() => {
		return () => {
			if (animationRef.current) {
				cancelAnimationFrame(animationRef.current);
			}
			if (yAxisAnimationRef.current) {
				cancelAnimationFrame(yAxisAnimationRef.current);
			}
		};
	}, []);

	const formatTimeLabel = useCallback((timestamp: number, timeRangeMs: number) => {
		const date = new Date(timestamp);

		if (timeRangeMs <= 6 * HOUR) {
			return date.toLocaleTimeString([], { hourCycle: "h23", hour: "2-digit", minute: "2-digit" });
		} else if (timeRangeMs <= DAY) {
			return date.toLocaleTimeString([], { hourCycle: "h23", hour: "2-digit", minute: "2-digit" });
		} else if (timeRangeMs <= 7 * DAY) {
			return date.toLocaleDateString([], { month: "numeric", day: "numeric" });
		} else {
			return date.toLocaleDateString([], { month: "numeric", day: "numeric" });
		}
	}, []);

	const visibleValues = useMemo(() => visibleData.map((d) => d.value), [visibleData]);

	const visibleMax = visibleValues[visibleValues.length - 1];

	const visibleMin = visibleValues[0];

	const generateNiceTicks = (min: number, max: number, targetTickCount: number = 4) => {
		const range = max - min;
		const roughStep = range / (targetTickCount - 1);

		const magnitude = Math.pow(10, Math.floor(Math.log10(roughStep)));
		const normalizedStep = roughStep / magnitude;

		let niceStep;
		if (normalizedStep <= 1) {
			niceStep = 1;
		} else if (normalizedStep <= 2) {
			niceStep = 2;
		} else if (normalizedStep <= 5) {
			niceStep = 5;
		} else {
			niceStep = 10;
		}

		niceStep *= magnitude;

		// Generate ticks
		const ticks = [];
		const firstTick = Math.floor(min / niceStep) * niceStep;
		const lastTick = Math.ceil(max / niceStep) * niceStep;

		for (let value = firstTick; value <= lastTick; value += niceStep) {
			if (value >= min && value <= max) {
				ticks.push(value);
			}
		}

		return ticks;
	};

	const { xScale, yScale, timeTicks, yTicks } = useMemo(() => {
		if (!visibleData.length || !dimensions.width || yAxisRange.min === yAxisRange.max) return {};

		const yScale = (value: number) => {
			const chartHeight = dimensions.height - 60;
			const normalizedValue = (value - yAxisRange.min) / (yAxisRange.max - yAxisRange.min);
			return chartHeight - normalizedValue * chartHeight + 20;
		};

		const xScale = (timestamp: number) => {
			const timeRange = timeWindow.endTime - timeWindow.startTime;
			const timePosition = timestamp - timeWindow.startTime;
			const xPosition = (timePosition / timeRange) * (dimensions.width - yLabelWidth - 10) + yLabelWidth + 5;
			return xPosition;
		};

		const generateTimeTicks = () => {
			const timeRange = timeWindow.endTime - timeWindow.startTime;
			let tickInterval: number;

			if (timeRange <= 6 * HOUR) {
				tickInterval = HOUR;
			} else if (timeRange <= DAY) {
				tickInterval = 4 * HOUR;
			} else if (timeRange <= 7 * DAY) {
				tickInterval = DAY;
			} else if (timeRange <= 30 * DAY) {
				tickInterval = 7 * DAY;
			} else if (timeRange <= 90 * DAY) {
				tickInterval = 3 * WEEK;
			} else {
				tickInterval = 30 * DAY;
			}

			const ticks = [];
			let currentTick = Math.ceil(timeWindow.startTime / tickInterval) * tickInterval - tickInterval;

			while (currentTick <= timeWindow.endTime + tickInterval) {
				const x = xScale(currentTick);
				if (x >= -30 && x <= dimensions.width + 30) {
					ticks.push({
						x,
						timestamp: currentTick,
						label: formatTimeLabel(currentTick, timeRange),
					});
				}
				currentTick += tickInterval;
			}

			return ticks;
		};

		const generateYTicks = () => {
			const ticks = generateNiceTicks(yAxisRange.min, yAxisRange.max, 6);
			return ticks.map((value) => ({
				y: yScale(value),
				value: value,
			}));
		};

		return {
			xScale,
			yScale,
			timeTicks: generateTimeTicks(),
			yTicks: generateYTicks(),
		};
	}, [visibleData, dimensions, timeWindow, timeRange, yLabelWidth, yAxisRange]);

	const generatePath = useCallback(() => {
		if (!visibleData || !xScale || !yScale) return "";

		if (!smoothInterpolation || visibleData.length < 3) {
			return visibleData
				.map(
					(point) =>
						`${point === visibleData[0] ? "M" : "L"} ${xScale(point.timestamp)} ${yScale(point.value)}`,
				)
				.join(" ");
		}

		const points = visibleData.map((point) => ({
			x: xScale(point.timestamp),
			y: yScale(point.value),
		}));

		let path = `M ${points[0].x} ${points[0].y}`;

		for (let i = 0; i < points.length - 1; i++) {
			const p0 = points[Math.max(0, i - 1)];
			const p1 = points[i];
			const p2 = points[i + 1];
			const p3 = points[Math.min(points.length - 1, i + 2)];

			const tension = Math.min(0.0005 * (p2.x - p1.x) ** 1.8, 1);
			const x1 = p1.x + ((p2.x - p0.x) / 6) * tension;
			const y1 = p1.y + ((p2.y - p0.y) / 6) * tension;
			const x2 = p2.x - ((p3.x - p1.x) / 6) * tension;
			const y2 = p2.y - ((p3.y - p1.y) / 6) * tension;

			path += ` C ${x1} ${y1} ${x2} ${y2} ${p2.x} ${p2.y}`;
		}

		return path;
	}, [visibleData, xScale, yScale, smoothInterpolation]);

	const updateCursorPosition = useCallback(
		(x: number) => {
			if (!visibleData || !xScale) return;

			let closestPoint = visibleData[0];
			let minDistance = Infinity;

			visibleData.forEach((point, idx) => {
				if (
					idx < leftPad||
					(idx > visibleData.length - rightPad + 1 &&
						visibleData[visibleData.length - 1].timestamp > timeWindow.endTime)
				)
					return;
				const pointX = xScale(point.timestamp);
				const distance = Math.abs(pointX - x);
				if (distance < minDistance) {
					minDistance = distance;
					closestPoint = point;
				}
			});

			setCurrentPosition({
				x: xScale(closestPoint.timestamp),
				y: yScale(closestPoint.value),
				data: closestPoint,
			});
		},
		[visibleData, xScale, yScale],
	);

	const handlePointerDown = useCallback(
		(e: React.PointerEvent) => {
			if (!svgRef.current) return;

			const rect = svgRef.current.getBoundingClientRect();
			const x = e.clientX - rect.left;

			setDragStartX(x);
			setPointerStartPosition({ x, y: e.clientY - rect.top, time: Date.now() });

			if (e.pointerType === "touch") {
				const timer = setTimeout(() => {
					updateCursorPosition(x);
				}, 500);
				setLongPressTimer(timer);
			}
		},
		[updateCursorPosition, setLongPressTimer, setDragStartX, setPointerStartPosition],
	);

	// Function to start Y-axis animation
	const startYAxisAnimation = useCallback(
		(targetMin: number, targetMax: number) => {
			if (yAxisAnimationRef.current) {
				cancelAnimationFrame(yAxisAnimationRef.current);
			}

			const animationDuration = 300; // 300ms animation
			const startTimestamp = performance.now();

			const currentMin = yAxisRange.min;
			const currentMax = yAxisRange.max;

			setYAxisAnimation({
				isAnimating: true,
				startMin: currentMin,
				startMax: currentMax,
				targetMin,
				targetMax,
				startTimestamp,
			});

			const animate = (currentTime: number) => {
				const elapsed = currentTime - startTimestamp;
				const progress = Math.min(elapsed / animationDuration, 1);

				// Easing function for smooth animation (ease-out)
				const easeProgress = 1 - Math.pow(1 - progress, 3);

				const newMin = currentMin + (targetMin - currentMin) * easeProgress;
				const newMax = currentMax + (targetMax - currentMax) * easeProgress;

				setYAxisRange({
					min: newMin,
					max: newMax,
				});

				if (progress < 1) {
					yAxisAnimationRef.current = requestAnimationFrame(animate);
				} else {
					// Animation complete
					setYAxisRange({
						min: targetMin,
						max: targetMax,
					});
					setYAxisAnimation(null);
					yAxisAnimationRef.current = null;
				}
			};

			yAxisAnimationRef.current = requestAnimationFrame(animate);
		},
		[yAxisRange.max, yAxisRange.min],
	);

	const alignRangeToNiceNumbers = (min: number, max: number) => {
		const range = max - min;
		const magnitude = Math.pow(10, Math.floor(Math.log10(range)));
		let alignedMax = Math.ceil(max / magnitude) * magnitude;
		let alignedMin = Math.floor(min / magnitude) * magnitude;

		return { min: alignedMin, max: alignedMax };
	};

	// Lazy adjustment logic for Y-axis range
	useEffect(() => {
		if (!visibleData.length || yAxisAnimation?.isAnimating) return;

		// Check if we need to adjust Y-axis range (when significant data is outside current range)
		const dataOutsideRange = visibleData.filter((d) => d.value < yAxisRange.min || d.value > yAxisRange.max).length;

		const threshold = visibleData.length * 0.1; // Lower threshold: 30% of data outside range
		const vmin = visibleData[leftPad - 1].value;
		const vmax = visibleData[visibleData.length - rightPad].value;

		if (dataOutsideRange > threshold) {
			const targetMin = Math.max(0, vmin);
			const targetMax = vmax;

			const buffer = (targetMax - targetMin) * 0.3;

			// Align to nice numbers
			const alignedRange = alignRangeToNiceNumbers(targetMin - buffer, targetMax + buffer);

			// Start animation to new range
			startYAxisAnimation(alignedRange.min, alignedRange.max);
		}
	}, [visibleData, yAxisRange, yAxisAnimation]);

	const handlePointerMove = useCallback(
		(e: React.PointerEvent) => {
			if (!svgRef.current) return;

			if (longPressTimer) {
				clearTimeout(longPressTimer);
			}

			setLastPointerPositions(
				[...lastPointerPositions, { x: e.clientX, y: e.clientY, time: Date.now() }].slice(-8),
			);

			const rect = svgRef.current.getBoundingClientRect();
			const x = e.clientX - rect.left;

			if (currentPosition && e.pointerType === "touch") {
				updateCursorPosition(x);
				return;
			}

			if (e.pointerType === "mouse") {
				updateCursorPosition(x);
			}

			if (!dragStartX) return;

			const deltaX = x - dragStartX;

			const windowWidth = dimensions.width - yLabelWidth - 5;
			const timeRange = timeWindow.endTime - timeWindow.startTime;

			const timeDelta = (deltaX / windowWidth) * timeRange;
			const newStartTime = timeWindow.startTime - timeDelta;
			const newEndTime = timeWindow.endTime - timeDelta;

			const minTime = sortedData[0].timestamp;
			const maxTime = sortedData[sortedData.length - 1].timestamp;

			const adjustedStartTime = Math.max(minTime - timeRange * 0.2, newStartTime);
			const adjustedEndTime = Math.min(maxTime + timeRange * 1, newEndTime);

			if (adjustedEndTime - adjustedStartTime < timeRange) {
				return;
			}

			setTimeWindow({
				startTime: adjustedStartTime,
				endTime: adjustedEndTime,
			});
			setDragStartX(x);

			if (longPressTimer) {
				clearTimeout(longPressTimer);
				setLongPressTimer(null);
			}
		},
		[dragStartX, timeWindow, dimensions.width, updateCursorPosition],
	);

	const startAnimation = useCallback(
		(targetStartTime: number, targetEndTime: number) => {
			if (animationRef.current) {
				cancelAnimationFrame(animationRef.current);
			}

			const animationDuration = 300; // 300ms animation
			const startTimestamp = performance.now();

			const currentStartTime = timeWindow.startTime;
			const currentEndTime = timeWindow.endTime;

			const animate = (currentTime: number) => {
				const elapsed = currentTime - startTimestamp;
				const progress = Math.min(elapsed / animationDuration, 1);

				// Easing function for smooth animation (ease-out)
				const easeProgress = 1 - Math.pow(1 - progress, 3);

				const newStartTime = currentStartTime + (targetStartTime - currentStartTime) * easeProgress;
				const newEndTime = currentEndTime + (targetEndTime - currentEndTime) * easeProgress;

				setTimeWindow({
					startTime: newStartTime,
					endTime: newEndTime,
				});

				if (progress < 1) {
					animationRef.current = requestAnimationFrame(animate);
				} else {
					// Animation complete
					setTimeWindow({
						startTime: targetStartTime,
						endTime: targetEndTime,
					});
					animationRef.current = null;
				}
			};

			animationRef.current = requestAnimationFrame(animate);
		},
		[timeWindow],
	);

	const getWindowShiftForAnimation = (timeRange: number, delta: number, type: "mouse" | "pen" | "touch") => {
		if (type === "mouse") {
			return delta > 0 ? -timeRange / 3 : timeRange / 3;
		} else return delta > 0 ? -timeRange : timeRange;
	};

	const handlePointerUp = useCallback(
		(e: React.PointerEvent) => {
			if (!svgRef.current) return;
			const rect = svgRef.current.getBoundingClientRect();
			const x = e.clientX - rect.left;

			setDragStartX(0);

			if (longPressTimer) {
				updateCursorPosition(x);
			}

			if (currentPosition) return;

			// Check for swipe gesture on all devices (not just touch)
			if (!lastPointerPositions || !pointerStartPosition) return;

			const totalDeltaX = x - pointerStartPosition.x;
			const totalTimeDelta = Date.now() - pointerStartPosition.time;

			const avgAcceleration = calculateXAxisAverageAcceleration(lastPointerPositions);
			const signAcc = avgAcceleration / Math.abs(avgAcceleration);

			// Only animate if the average acceleration is in the same direction as the swipe
			if ((avgAcceleration - 500 * signAcc) * totalDeltaX > 0) {
				const timeRange = timeWindow.endTime - timeWindow.startTime;
				const windowShift = getWindowShiftForAnimation(timeRange, totalDeltaX, e.pointerType);

				const newStartTime = Math.max(data[0]?.timestamp || 0, timeWindow.startTime + windowShift);
				const newEndTime = Math.min(
					data[data.length - 1]?.timestamp || Infinity,
					timeWindow.endTime + windowShift,
				);

				if (newEndTime - newStartTime === timeRange) {
					// Use smooth animation instead of direct set
					startAnimation(newStartTime, newEndTime);
				}
			}
			setPointerStartPosition(null);
			setLastPointerPositions([]);
		},
		[longPressTimer, timeWindow, data, startAnimation],
	);

	const handlePointerLeave = useCallback((e: React.PointerEvent) => {
		if (e.pointerType === "touch") {
			return;
		}
		setCurrentPosition(null);
		setDragStartX(0);
		setPointerStartPosition(null);
		setLastPointerPositions([]);
	}, []);

	if (!data.length) {
		return (
			<div ref={containerRef} className="relative" style={{ width, height }}>
				<div className="flex items-center justify-center h-full text-sm">暂无数据</div>
			</div>
		);
	}

	return (
		<div {...rest} ref={ref}>
			<div
				ref={containerRef}
				className="relative touch-none select-none h-full"
				style={{
					width,
				}}
			>
				<svg
					ref={svgRef}
					width={dimensions.width}
					className="block h-full"
					onPointerDown={handlePointerDown}
					onPointerMove={handlePointerMove}
					onPointerUp={handlePointerUp}
					onPointerLeave={handlePointerLeave}
				>
					<defs>
						<clipPath id="pathClip">
							<rect
								x={yLabelWidth + 5}
								y={20}
								width={dimensions.width - yLabelWidth - 10}
								height={dimensions.height - 60}
							/>
						</clipPath>
						<clipPath id="timeTicksClip">
							<rect
								x={yLabelWidth + 5}
								y={0}
								width={dimensions.width - yLabelWidth - 10}
								height={dimensions.height}
							/>
						</clipPath>
					</defs>
					{/* Y轴刻度 */}
					{yTicks &&
						yTicks.map((tick, index) => (
							<g key={`y-${index}`}>
								<line
									x1={yLabelWidth + 5}
									y1={tick.y}
									x2={dimensions.width - 5}
									y2={tick.y}
									stroke="rgba(200, 200, 200, 0.3)"
									strokeWidth="1"
									strokeDasharray="2,2"
								/>
								<text
									x={yLabelWidth - 2}
									y={tick.y + 4}
									textAnchor="end"
									className="text-xs fill-neutral-600 dark:fill-neutral-300"
								>
									{Math.round(tick.value / 1000) / 10} 万
								</text>
							</g>
						))}

					<text
						ref={widthProbeRef}
						className="text-xs fill-neutral-600 dark:fill-neutral-300 [visibility:hidden]"
					>
						{Math.round(globalMaxValue / 1000) / 10} 万
					</text>

					{/* X轴时间刻度 */}
					{timeTicks &&
						timeTicks.map((tick, index) => (
							<g key={`x-${index}`} clipPath="url(#timeTicksClip)">
								<line
									x1={tick.x}
									y1={20}
									x2={tick.x}
									y2={dimensions.height - 40}
									stroke="rgba(200, 200, 200, 0.3)"
									strokeWidth="1"
									strokeDasharray="2,2"
								/>
								<text
									x={tick.x}
									y={dimensions.height - 8}
									textAnchor="middle"
									className="text-xs fill-neutral-600 dark:fill-neutral-300"
								>
									{tick.label}
								</text>
							</g>
						))}

					{/* The curve line */}
					<path
						d={generatePath()}
						fill="none"
						stroke={accentColor}
						strokeWidth="2"
						strokeLinecap="round"
						clipPath="url(#pathClip)"
					/>

					{/* 当前光标指示线 */}
					{currentPosition && (
						<g>
							{/* 垂直指示线 */}
							<line
								x1={currentPosition.x}
								y1={0}
								x2={currentPosition.x}
								y2={dimensions.height - 25}
								strokeWidth="1"
								className="stroke-neutral-500 dark:stroke-neutral-700"
							/>
							{/* 数据点 */}
							<circle
								cx={currentPosition.x}
								cy={currentPosition.y}
								r="4"
								fill={accentColor}
								stroke="white"
								strokeWidth="2"
							/>
						</g>
					)}
				</svg>
			</div>
		</div>
	);
};
