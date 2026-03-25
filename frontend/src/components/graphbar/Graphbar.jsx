import React, { useState, useEffect, useMemo, useCallback, useContext, useRef } from "react";
// import Plot from "react-plotly.js";
// import Papa from "papaparse";
import "./graphbar.css";
import ReactFlow, {
    Controls,
    Background,
    useNodesState,
    useEdgesState,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { Context } from "../../context/Context";
// import companies from '../../../companies.json';


class GraphNode {
    constructor(value, metadata = {}) {
        this.value = value;
        this.children = []; // Can have multiple or no children
        this.metadata = metadata;
    }

    addChild(child) {
        this.children.push(child);
        return this;
    }
}

const createSampleGraph = (graphData) => {
    const nodesList = {};
    const levelCounts = {}; // Object to track the number of nodes at each level
    const totalWidth = 400; // Total width for positioning nodes
    const verticalSpacing = 150; // Vertical spacing between levels (already given)


    for (const nodeData of graphData.nodes) {
        nodesList[nodeData.value] = new GraphNode(nodeData.value, nodeData.metadata);
    }


    for (const edge of graphData.edges) {
        const sourceNode = nodesList[edge.source];
        const targetNode = nodesList[edge.target];
        sourceNode.addChild(targetNode); // Add actual node references
    }


    const setNodeLevel = (node, level) => {
        if (node.level !== undefined) return; // Avoid resetting level if already set
        node.level = level;

        // Track the number of nodes at each level
        levelCounts[level] = (levelCounts[level] || 0) + 1;

        // Propagate the level to child nodes
        for (const child of node.children) {
            setNodeLevel(child, level + 1); // Set child's level to parent level + 1
        }
    };

    // Start from Node 0 (the root) and set levels
    setNodeLevel(nodesList[0], 0);

    // Position the nodes along the x-axis for each level
    const positionNodes = () => {
        // Iterate through each level and position nodes horizontally
        for (const level in levelCounts) {
            const numberOfNodesAtLevel = levelCounts[level];

            // Special case when there's only one node
            if (numberOfNodesAtLevel === 1) {
                const node = Object.values(nodesList).find(node => node.level == level);
                node.x = totalWidth / 2; // Center the single node
                node.y = level * verticalSpacing; // Assign y position based on the level
                continue;
            }

            // Calculate the horizontal spacing
            const spacing = totalWidth / (numberOfNodesAtLevel - 1);

            // Get all nodes at the current level
            const nodesAtLevel = Object.values(nodesList).filter(node => node.level == level);

            // Position each node symmetrically
            nodesAtLevel.forEach((node, index) => {
                node.x = spacing * index; // Assign x position (symmetric, starting from 0)
                node.y = level * verticalSpacing; // Assign y position based on the level
            });
        }
    };
    // Position nodes after setting their levels
    positionNodes();

    const nodes = [];
    const edges = [];

    Object.values(nodesList).forEach(node => {
        nodes.push({
            id: String(node.value),
            position: { x: node.x, y: node.y },
            data: {
                label: String(node.value),
                metadata: node.metadata
            },
            style: {
                backgroundColor: '#e6f3ff',
                color: 'black',
                border: '1px solid #3a9bdc',
                borderRadius: '50%',
                width: '40px',
                height: '40px',
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
            }
        });

        const childCount = node.children.length;
        node.children.forEach((child, index) => {
            // Add edge
            edges.push({
                id: `edge-${node.value}-${child.value}`,
                source: String(node.value),
                target: String(child.value),
                animated: true,
                style: { stroke: '#1C1CF0' },
            });

        });

    });
    return { nodes, edges }; // Return the root node
}



const GraphBar = ({ isOpen, onToggle, offset = 0, drawerWidth = 420, onResize }) => {
    const [popupData, setPopupData] = useState(null);
    const graphContainerRef = useRef(null);
    // const [chartType, setChartType] = useState("candlestick");
    // const [companyData, setCompanyData] = useState([]); // Company metadata
    // const [stockData, setStockData] = useState({}); // Stock data for all companies
    // const [appleData, setAppleData] = useState([]);
    // const [msftData, setMsftData] = useState([]);
    const {
        graphData,
        setGraphData,
    } = useContext(Context);


    // Convert graph to React Flow elements
    const { nodes: flowNodes, edges: flowEdges } = useMemo(() => {
        if (graphData) {
            console.log('Creating graph from data:', graphData);
            return createSampleGraph(graphData);  // Only create the graph if graphData is valid
        }
        return { nodes: [], edges: [] };  // Fallback to empty nodes and edges if no data
    }, [graphData]);

    // Use React Flow hooks
    const [nodes, setNodes, onNodesChange] = useNodesState(flowNodes);
    const [edges, setEdges, onEdgesChange] = useEdgesState(flowEdges);

    useEffect(() => {
        if (graphData) {
            const { nodes: updatedNodes, edges: updatedEdges } = createSampleGraph(graphData);
            setNodes(updatedNodes);
            setEdges(updatedEdges);
        } else {
            // If no data, reset to empty nodes/edges
            setNodes([]);
            setEdges([]);
        }
    }, [graphData, setNodes, setEdges]);

    const handleNodeMouseEnter = useCallback((event, node) => {
        const container = graphContainerRef.current;
        if (!container) {
            return;
        }
        const bounds = container.getBoundingClientRect();
        setPopupData({
            data: node.data,
            position: { x: event.clientX - bounds.left, y: event.clientY - bounds.top },
        });
    }, []);

    const handleNodeMouseMove = useCallback((event, node) => {
        const container = graphContainerRef.current;
        if (!container) {
            return;
        }
        const bounds = container.getBoundingClientRect();
        setPopupData((prev) => {
            if (!prev || prev.data?.label !== node.data?.label) {
                return prev;
            }
            return {
                ...prev,
                position: { x: event.clientX - bounds.left, y: event.clientY - bounds.top },
            };
        });
    }, []);

    const handleNodeMouseLeave = useCallback(() => {
        setPopupData(null);
    }, []);

    const handleResizeStart = useCallback(
        (event) => {
            if (!isOpen || !onResize) {
                return;
            }
            event.preventDefault();
            const initialCursor = document.body.style.cursor;
            const initialUserSelect = document.body.style.userSelect;
            document.body.style.cursor = "ew-resize";
            document.body.style.userSelect = "none";

            const handleMove = (moveEvent) => {
                const nextWidth = window.innerWidth - moveEvent.clientX;
                onResize(nextWidth);
            };

            const handleUp = () => {
                document.body.style.cursor = initialCursor;
                document.body.style.userSelect = initialUserSelect;
                window.removeEventListener("pointermove", handleMove);
                window.removeEventListener("pointerup", handleUp);
            };

            window.addEventListener("pointermove", handleMove);
            window.addEventListener("pointerup", handleUp);
        },
        [isOpen, onResize]
    );


    const [tableData, setTableData] = useState([]);
    const [headers, setHeaders] = useState([]);

    // const handleFileUpload = (event) => {
    //     const file = event.target.files[0];
    //     if (!file) return;

    //     Papa.parse(file, {
    //         header: true,
    //         complete: (result) => {
    //             setHeaders(result.meta.fields || []);
    //             setTableData(result.data);
    //         },
    //     });
    // };
    // useEffect(() => {
    //     companies.forEach((company) => {
    //         Papa.parse(company.file_path, {
    //             download: true,
    //             header: true,
    //             complete: (result) => {
    //                 setStockData((prevData) => ({
    //                     ...prevData,
    //                     [company.ticker_symbol]: result.data,
    //                 }));
    //             },
    //         });
    //     });
    // }, []); // Empty dependency array to run once on component mount

    // const renderChart = () => {
    //     const chartData = Object.keys(stockData).map((ticker, index) => {
    //         const data = stockData[ticker];
            
    //         // Define different colors for each company
    //         const lineColor = index % 2 === 0 ? "blue" : "orange"; // Alternate between blue and orange for line charts
    //         const candlestickColor = index % 2 === 0 ? "green" : "purple"; // Alternate between green and purple for candlestick charts
    
    //         if (chartType === "candlestick") {
    //             return {
    //                 x: data.map((row) => row.Date),
    //                 open: data.map((row) => parseFloat(row.Open)),
    //                 high: data.map((row) => parseFloat(row.High)),
    //                 low: data.map((row) => parseFloat(row.Low)),
    //                 close: data.map((row) => parseFloat(row.Close)),
    //                 type: "candlestick",
    //                 name: ticker,
    //                 increasing: { line: { color: candlestickColor } },
    //                 decreasing: { line: { color: "red" } }, // Keep red for decreasing
    //             };
    //         } else if (chartType === "line") {
    //             return {
    //                 x: data.map((row) => row.Date),
    //                 y: data.map((row) => parseFloat(row.Close)),
    //                 type: "line",
    //                 name: ticker,
    //                 line: { color: lineColor }, // Alternate color for line charts
    //             };
    //         }
    //         return null;
    //     }).filter((data) => data !== null);
    
    //     return (
    //         <Plot
    //             data={chartData}
    //             layout={{
    //                 title: chartType === "candlestick" ? "Stock Candlestick Chart" : "Stock Time Series",
    //                 xaxis: { title: "Date" },
    //                 yaxis: { title: "Price (USD)" },
    //                 responsive: true,
    //             }}
    //             useResizeHandler
    //             style={{ width: "100%", height: "100%" }}
    //         />
    //     );
    // };
    


    return (
        <div
            className="graph-main"
            style={{ right: `${offset}px`, "--drawer-width": `${drawerWidth}px` }}
        >
            <button
                className={`toggle-button ${isOpen ? "open" : "closed"}`}
                onClick={() => onToggle(!isOpen)}
            >
                {isOpen ? "Graph →" : "Graph ←"}
            </button>

            <div className={`graph-bar ${isOpen ? "open" : "closed"}`}>
                <div
                    className="drawer-resize-handle"
                    role="separator"
                    aria-label="Resize graph panel"
                    onPointerDown={handleResizeStart}
                />
                {isOpen && (
                    <div className="panel-header">
                        <div>
                            <p className="panel-title">Graph View</p>
                            <p className="panel-subtitle">Relationship map</p>
                        </div>
                        <span className="panel-pill">Live</span>
                    </div>
                )}
                <div className="graph-render" ref={graphContainerRef}>
                    {nodes.length === 0 ? (
                        <div className="graph-empty">
                            <p>No graph yet.</p>
                            <span>Run a query to generate the relationship map.</span>
                        </div>
                    ) : (
                        <ReactFlow
                            nodes={nodes}
                            edges={edges}
                            onNodesChange={onNodesChange}
                            onEdgesChange={onEdgesChange}
                            fitView
                            attributionPosition="top-right"
                            onNodeMouseEnter={handleNodeMouseEnter}
                            onNodeMouseMove={handleNodeMouseMove}
                            onNodeMouseLeave={handleNodeMouseLeave}
                        >
                            <Controls />
                            <Background color="#1C1CF0" gap={6} variant="dots" />
                        </ReactFlow>
                    )}
                    {popupData && (
                        <div
                            className="graph-tooltip"
                            style={{
                                top: popupData.position.y + 10, // Offset from click
                                left: popupData.position.x + 10,
                            }}
                        >
                            <div className="details">
                                <strong>Node Details:</strong>
                                {popupData.data.metadata && (
                                    <div>
                                        <ul style={{ listStyleType: "none", padding: 0 }}>
                                            {Object.entries(popupData.data.metadata).map(([key, value]) => (
                                                <li key={key} style={{ marginBottom: "5px" }}>
                                                    <strong>{key}:</strong> {value}
                                                </li>
                                            ))}
                                        </ul>
                                    </div>
                                )}
                            </div>
                        </div>
                    )}
                </div>
                {/* { /activeButton === 2 && (
                    <div className="chart-container">
                        <div style={{ marginBottom: "20px" }}>
                            <label htmlFor="chartType" style={{ marginLeft: "20px" }}>
                                Select Chart Type: </label>
                            <select
                                id="chartType"
                                value={chartType}
                                style={{ marginLeft: '10px' }}
                                onChange={(e) => setChartType(e.target.value)}
                            >
                                <option value="candlestick">Candlestick</option>
                                <option value="line">Time Series</option>
                                <option value="bullet">Bullet Chart</option> */}
                            {/* </select> */}
                        {/* </div> */}
                        {/* {renderChart()} */}
                    {/* </div> */}
                {/* {activeButton === 3 && (
                    <div className="csv-render" style={{ padding: "20px", overflowX: "scroll" }}>
                        <input type="file" accept=".csv" onChange={handleFileUpload} />
                        {tableData.length > 0 && (
                            <table border="1" style={{ marginTop: "20px", width: "100%", textAlign: "left" }}>
                                <thead>
                                    <tr>
                                        {headers.map((header, index) => (
                                            <th key={index}>{header}</th>
                                        ))}
                                    </tr>
                                </thead>
                                <tbody>
                                    {tableData.map((row, index) => (
                                        <tr key={index}>
                                            {headers.map((header, colIndex) => (
                                                <td key={colIndex}>{row[header]}</td>
                                            ))}
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        )}
                    </div>
                )} */}
            </div>
        </div>
    );
};

export default GraphBar;
