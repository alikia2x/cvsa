import { useState, useRef, useMemo, useCallback, useEffect } from "react";

export const TimeSeriesChart = ({
	data = [],
	width = "100%",
	height = 300,
	accentColor = "#007AFF",
	showGrid = true,
	smoothInterpolation = true,
	timeRange = "auto", // '6h', '1d', '7d', '30d', 'auto'
}: {
	data: { timestamp: number; value: number }[];
	width?: string;
	height?: number;
	accentColor?: string;
	showGrid?: boolean;
	smoothInterpolation?: boolean;
	timeRange?: "6h" | "1d" | "7d" | "30d" | "auto";
}) => {
	const svgRef = useRef<SVGSVGElement>(null);
	const containerRef = useRef<HTMLDivElement>(null);
	const [currentPosition, setCurrentPosition] = useState<{
		x: number;
		y: number;
		data: { timestamp: number; value: number } | null;
	} | null>(null);
	const [isDragging, setIsDragging] = useState(false);
	const [dragStartX, setDragStartX] = useState(0);
	const [viewBox, setViewBox] = useState({ start: 0, end: 1 });
	const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
	const [isTouchActive, setIsTouchActive] = useState(false);
	const [longPressTimer, setLongPressTimer] = useState<NodeJS.Timeout | null>(null);
	const [touchStartPosition, setTouchStartPosition] = useState<{ x: number; y: number } | null>(null);

	// 响应式尺寸处理
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

	// 格式化时间标签
	const formatTimeLabel = useCallback((timestamp: number, range: "6h" | "1d" | "7d" | "30d" | "auto") => {
		const date = new Date(timestamp);
		const now = new Date();
		const isToday = date.toDateString() === now.toDateString();

		switch (range) {
			case "6h":
				return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
			case "1d":
				return isToday ? "今天" : date.toLocaleDateString([], { month: "short", day: "numeric" });
			case "7d":
				return date.toLocaleDateString([], { weekday: "short" });
			case "30d":
				return date.toLocaleDateString([], { month: "short", day: "numeric" });
			default:
				return date.toLocaleDateString([], { month: "short", day: "numeric" });
		}
	}, []);

	// 处理数据范围和时间间隔
	const { xScale, yScale, visibleData, timeTicks, yTicks } = useMemo(() => {
		if (!data.length || !dimensions.width) return {};

		const visibleDataPoints = data.filter((point, index) => {
			const progress = index / (data.length - 1);
			return progress >= viewBox.start && progress <= viewBox.end;
		});

		if (!visibleDataPoints.length) return {};

		// 计算Y轴范围（带一些边距）
		const values = visibleDataPoints.map((d) => d.value);
		const minValue = Math.min(...values);
		const maxValue = Math.max(...values);
		const valueRange = maxValue - minValue;
		const padding = valueRange * 0.1;

		const yScale = (value: number) => {
			const chartHeight = dimensions.height - 60; // 为标签留出空间
			return chartHeight - ((value - (minValue - padding)) / (maxValue - minValue + 2 * padding)) * chartHeight + 20;
		};

		// X轴比例尺
		const xScale = (index: number) => {
			const totalPoints = visibleDataPoints.length;
			return (index / (totalPoints - 1)) * (dimensions.width - 60) + 40;
		};

		// 生成时间刻度
		const generateTimeTicks = () => {
			const tickCount = Math.min(6, Math.floor(dimensions.width / 80));
			const ticks = [];

			for (let i = 0; i < tickCount; i++) {
				const dataIndex = Math.floor((i / (tickCount - 1)) * (visibleDataPoints.length - 1));
				if (visibleDataPoints[dataIndex]) {
					ticks.push({
						x: xScale(dataIndex),
						timestamp: visibleDataPoints[dataIndex].timestamp,
						label: formatTimeLabel(visibleDataPoints[dataIndex].timestamp, timeRange),
					});
				}
			}
			return ticks;
		};

		// 生成Y轴刻度
		const generateYTicks = () => {
			const tickCount = 4;
			const ticks = [];

			for (let i = 0; i <= tickCount; i++) {
				const value = minValue - padding + (maxValue + padding - (minValue - padding)) * (i / tickCount);
				const y = yScale(value);
				ticks.push({
					y,
					value: Math.round(value * 100) / 100, // 保留两位小数
				});
			}
			return ticks;
		};

		return {
			xScale,
			yScale,
			visibleData: visibleDataPoints,
			timeTicks: generateTimeTicks(),
			yTicks: generateYTicks(),
		};
	}, [data, dimensions, viewBox, timeRange]);

	// 生成平滑路径
	const generatePath = useCallback(() => {
		if (!visibleData || !xScale || !yScale) return "";

		if (!smoothInterpolation || visibleData.length < 3) {
			// 直线连接
			return visibleData
				.map((point, index) => `${index === 0 ? "M" : "L"} ${xScale(index)} ${yScale(point.value)}`)
				.join(" ");
		}

		// Catmull-Rom 平滑曲线
		const points = visibleData.map((point, index) => ({
			x: xScale(index),
			y: yScale(point.value),
		}));

		let path = `M ${points[0].x} ${points[0].y}`;

		for (let i = 0; i < points.length - 1; i++) {
			const p0 = points[Math.max(0, i - 1)];
			const p1 = points[i];
			const p2 = points[i + 1];
			const p3 = points[Math.min(points.length - 1, i + 2)];

			const tension = 0.5;
			const x1 = p1.x + ((p2.x - p0.x) / 6) * tension;
			const y1 = p1.y + ((p2.y - p0.y) / 6) * tension;
			const x2 = p2.x - ((p3.x - p1.x) / 6) * tension;
			const y2 = p2.y - ((p3.y - p1.y) / 6) * tension;

			path += ` C ${x1} ${y1} ${x2} ${y2} ${p2.x} ${p2.y}`;
		}

		return path;
	}, [visibleData, xScale, yScale, smoothInterpolation]);

	// 更新光标位置和对应数据点
	const updateCursorPosition = useCallback(
		(x: number) => {
			if (!visibleData || !xScale) return;

			// 找到最近的数据点
			let closestIndex = 0;
			let minDistance = Infinity;

			visibleData.forEach((point, index) => {
				const pointX = xScale(index);
				const distance = Math.abs(pointX - x);
				if (distance < minDistance) {
					minDistance = distance;
					closestIndex = index;
				}
			});

			if (minDistance < 50) {
				// 灵敏度阈值
				const point = visibleData[closestIndex];
				setCurrentPosition({
					x: xScale(closestIndex),
					y: yScale(point.value),
					data: point,
				});
			} else {
				setCurrentPosition(null);
			}
		},
		[visibleData, xScale, yScale],
	);

	// 鼠标事件处理
	const handleMouseDown = useCallback(
		(e: React.MouseEvent) => {
			console.log("mouse down");
			if (!svgRef.current) return;

			const rect = svgRef.current.getBoundingClientRect();
			const x = e.clientX - rect.left;

			setIsDragging(true);
			setDragStartX(x);
			updateCursorPosition(x);
		},
		[updateCursorPosition],
	);

	const handleMouseMove = useCallback(
		(e: React.MouseEvent) => {
			console.log("mouse move");
			if (!svgRef.current) return;

			const rect = svgRef.current.getBoundingClientRect();
			const x = e.clientX - rect.left;

			if (isDragging) {
				// 滚动逻辑
				const deltaX = x - dragStartX;
				if (Math.abs(deltaX) > 10) {
					const dragSpeed = 0.02;
					const newStart = Math.max(0, viewBox.start - deltaX * dragSpeed);
					const newEnd = Math.min(1, viewBox.end - deltaX * dragSpeed);

					if (newEnd - newStart === viewBox.end - viewBox.start) {
						setViewBox({ start: newStart, end: newEnd });
					}
					setDragStartX(x);
				} else {
					// 光标位置更新
					updateCursorPosition(x);
				}
			} else {
				// 悬停时更新光标位置
				updateCursorPosition(x);
			}
		},
		[isDragging, dragStartX, viewBox, updateCursorPosition],
	);

	const handleMouseUp = useCallback(() => {
		console.log("mouse up");
		setIsDragging(false);
	}, []);

	const handleMouseLeave = useCallback(() => {
		console.log("mouse leave");
		setIsDragging(false);
		setCurrentPosition(null);
	}, []);

	// 触摸事件处理
	const handleTouchStart = useCallback(
		(e: React.TouchEvent) => {
			console.log("touch start");
			if (!svgRef.current) return;

			const touch = e.touches[0];
			const rect = svgRef.current.getBoundingClientRect();
			const x = touch.clientX - rect.left;

			setIsDragging(true);
			setDragStartX(x);
			updateCursorPosition(x);
		},
		[updateCursorPosition],
	);

	const handleTouchMove = useCallback(
		(e: React.TouchEvent) => {
			console.log("touch move");
			if (!isDragging || !svgRef.current) return;

			const touch = e.touches[0];
			const rect = svgRef.current.getBoundingClientRect();
			const x = touch.clientX - rect.left;

			// 滚动逻辑
			const deltaX = x - dragStartX;
			if (Math.abs(deltaX) > 10) {
				const dragSpeed = 0.02;
				const newStart = Math.max(0, viewBox.start - deltaX * dragSpeed);
				const newEnd = Math.min(1, viewBox.end - deltaX * dragSpeed);

				if (newEnd - newStart === viewBox.end - viewBox.start) {
					setViewBox({ start: newStart, end: newEnd });
				}
				setDragStartX(x);
			} else {
				// 光标位置更新
				updateCursorPosition(x);
			}
		},
		[isDragging, dragStartX, viewBox, updateCursorPosition],
	);

	const handleTouchEnd = useCallback(() => {
		console.log("touch end");
		setIsDragging(false);
		setCurrentPosition(null);
	}, []);


	if (!data.length) {
		return (
			<div ref={containerRef} className="time-series-chart" style={{ width, height, position: "relative" }}>
				<div
					style={{
						display: "flex",
						alignItems: "center",
						justifyContent: "center",
						height: "100%",
						color: "#999",
						fontSize: "14px",
					}}
				>
					暂无数据
				</div>
			</div>
		);
	}

	return (
		<div
			ref={containerRef}
			className="time-series-chart"
			style={{
				width,
				height,
				position: "relative",
				touchAction: "none",
				userSelect: "none",
				WebkitUserSelect: "none",
			}}
		>
			<svg
				ref={svgRef}
				width={dimensions.width}
				height={dimensions.height}
				style={{ display: "block" }}
				onMouseDown={handleMouseDown}
				onMouseMove={handleMouseMove}
				onMouseUp={handleMouseUp}
				onMouseLeave={handleMouseLeave}
				onTouchStart={handleTouchStart}
				onTouchMove={handleTouchMove}
				onTouchEnd={handleTouchEnd}
			>

				{/* Y轴刻度 */}
				{yTicks &&
					yTicks.map((tick, index) => (
						<g key={`y-${index}`}>
							<line
								x1={40}
								y1={tick.y}
								x2={dimensions.width - 20}
								y2={tick.y}
								stroke="rgba(200, 200, 200, 0.3)"
								strokeWidth="1"
								strokeDasharray="2,2"
							/>
							<text
								x={35}
								y={tick.y + 4}
								textAnchor="end"
								className="text-xs fill-gray-400 dark:fill-neutral-300"
							>
								{tick.value.toLocaleString()}
							</text>
						</g>
					))}

				{/* X轴时间刻度 */}
				{timeTicks &&
					timeTicks.map((tick, index) => (
						<g key={`x-${index}`}>
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
								className="text-xs fill-gray-400 dark:fill-neutral-300"
							>
								{tick.label}
							</text>
						</g>
					))}

				{/* 折线路径 */}
				<path d={generatePath()} fill="none" stroke={accentColor} strokeWidth="2" strokeLinecap="round" />

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
							className="stroke-gray-500 dark:stroke-neutral-700"
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

			{/* 浮动数据标签 */}
			{currentPosition && (
				<div
					style={{
						position: "absolute",
						top: 10,
						left: 20,
						right: 20,
						pointerEvents: "none",
					}}
				>
					<div
						style={{
							display: "flex",
							justifyContent: "space-between",
							alignItems: "center",
						}}
					>
						<div className="bg-white/90 dark:bg-neutral-800 text-sm rounded-md px-2 py-1">
							{currentPosition.data &&
								new Date(currentPosition.data.timestamp).toLocaleTimeString([], {
									month: "2-digit",
									day: "2-digit",
									hour: "2-digit",
									minute: "2-digit",
								})}
						</div>

						<div className="bg-white/90 dark:bg-neutral-800 text-sm rounded-md px-2 py-1">
							{currentPosition.data && currentPosition.data.value.toLocaleString()}
						</div>
					</div>
				</div>
			)}
		</div>
	);
};
