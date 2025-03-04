import React, { useState, useEffect, useRef } from 'react';
import * as d3 from 'd3';

// Function to load available data sources from server
const fetchDataSources = async () => {
  try {
    const response = await fetch('http://localhost:3001/api/datasources');
    if (!response.ok) {
      throw new Error('Failed to fetch data sources');
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching data sources:', error);
    // Return fallback data sources if server is not available
    return [
      { id: 'employment_level', name: 'Employment Level' },
      { id: 'employment_dynamics', name: 'Employment Dynamics' },
      { id: 'inflation_level', name: 'Inflation Level' },
      { id: 'inflation_dynamics', name: 'Inflation Dynamics' }
    ];
  }
};

// Function to load data from JSON file via server
const loadDataFromServer = async (sourceId) => {
  try {
    const response = await fetch(`http://localhost:3001/api/data/${sourceId}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch data for ${sourceId}`);
    }
    const jsonData = await response.json();
    return transformData(jsonData);
  } catch (error) {
    console.error("Error loading data from server:", error);
    // Fallback to sample data if server request fails
    return generateSampleData();
  }
};

// Transform JSON data to the format needed by visualization
const transformData = (jsonData) => {
  const docSet = new Set();
  const dateMap = {};
  const pairs = [];
  const rankings = [];
  
  // Process each pair in the JSON data
  jsonData.forEach(item => {
    const doc1 = item.doc_1;
    const doc2 = item.doc_2;
    const ranking = item.ranking;
    const date1 = new Date(item.date_1);
    const date2 = new Date(item.date_2);
    
    // Add documents to set and date map
    docSet.add(doc1);
    docSet.add(doc2);
    dateMap[doc1] = date1;
    dateMap[doc2] = date2;
    
    // Add pair and ranking
    pairs.push([doc1, doc2]);
    rankings.push(ranking);
  });
  
  // Convert set to array
  const docIds = Array.from(docSet);
  
  return {
    docIds,
    dates: dateMap,
    pairs,
    rankings
  };
};

// Sample data generation for demo purposes
const generateSampleData = () => {
  const startDate = new Date(2014, 0, 1);
  const endDate = new Date(2022, 0, 1);
  const timeRange = endDate - startDate;
  
  // Generate document IDs and dates
  const docIds = Array.from({ length: 30 }, (_, i) => `doc_${i+1}`);
  const docDates = docIds.map(id => {
    const randomOffset = Math.random() * timeRange;
    return new Date(startDate.getTime() + randomOffset);
  });
  
  // Create mapping from document ID to date
  const dates = {};
  docIds.forEach((id, index) => {
    dates[id] = docDates[index];
  });
  
  // Generate pairwise rankings
  const pairs = [];
  const rankings = [];
  
  // Generate some pairwise rankings
  for (let i = 0; i < docIds.length; i++) {
    for (let j = i + 1; j < docIds.length; j++) {
      // Only create a comparison for some pairs (randomly)
      if (Math.random() < 0.3) {
        pairs.push([docIds[i], docIds[j]]);
        
        // Prefer to rank newer documents higher with some randomness
        const dateI = dates[docIds[i]];
        const dateJ = dates[docIds[j]];
        const timeDiff = dateJ - dateI;
        
        // Probabilistic ranking based on time difference
        let probNewerIsHigher = 0.7 + (timeDiff / timeRange) * 0.2;
        probNewerIsHigher = Math.min(Math.max(probNewerIsHigher, 0.1), 0.9);
        
        if (Math.random() < 0.15) {
          // Small chance of equal ranking
          rankings.push(0);
        } else if (Math.random() < probNewerIsHigher) {
          // Assign ranking based on time (newer doc tends to be higher)
          rankings.push(dateI < dateJ ? -1 : 1);
        } else {
          // Sometimes go against the time trend
          rankings.push(dateI < dateJ ? 1 : -1);
        }
      }
    }
  }
  
  return { docIds, dates, pairs, rankings };
};

const DocumentRankingGraph = () => {
  // State for data sources, selected data source, and loading state
  const [dataSources, setDataSources] = useState([]);
  const [selectedDataSourceId, setSelectedDataSourceId] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  
  // State for the selected document, date range, and graph data
  const [selectedDoc, setSelectedDoc] = useState(null);
  const [dateRange, setDateRange] = useState([new Date(2014, 0, 1), new Date(2022, 0, 1)]);
  const [sliderValues, setSliderValues] = useState([0, 100]);
  const [data, setData] = useState(null);
  const [filteredData, setFilteredData] = useState(null);
  const [hoveredNode, setHoveredNode] = useState(null);
  
  // Refs for the SVG and tooltip elements
  const svgRef = useRef(null);
  const tooltipRef = useRef(null);
  
  // Load available data sources on component mount
  useEffect(() => {
    const loadDataSources = async () => {
      try {
        const sources = await fetchDataSources();
        setDataSources(sources);
        
        if (sources.length > 0) {
          setSelectedDataSourceId(sources[0].id);
        }
      } catch (error) {
        console.error("Error loading data sources:", error);
      }
    };
    
    loadDataSources();
  }, []);
  
  // Load data when selected data source changes
  useEffect(() => {
    const loadData = async () => {
      if (!selectedDataSourceId) return;
      
      setIsLoading(true);
      
      try {
        const rawData = await loadDataFromServer(selectedDataSourceId);
        setData(rawData);
        
        // Select the first document by default
        if (rawData.docIds.length > 0) {
          setSelectedDoc(rawData.docIds[0]);
        }
        
        // Set initial date range from the data
        const dates = Object.values(rawData.dates);
        const minDate = new Date(Math.min(...dates.map(d => d.getTime())));
        const maxDate = new Date(Math.max(...dates.map(d => d.getTime())));
        setDateRange([minDate, maxDate]);
      } catch (error) {
        console.error("Error loading data:", error);
      } finally {
        setIsLoading(false);
      }
    };
    
    loadData();
  }, [selectedDataSourceId]);

  // Update filtered data when selection changes
  useEffect(() => {
    if (!data || !selectedDoc) return;
    
    // Filter documents within the selected date range
    const { docIds, dates, pairs, rankings } = data;
    const minDate = dateRange[0];
    const maxDate = dateRange[1];
    
    // Filter documents by date
    const filteredDocIds = docIds.filter(id => {
      const date = dates[id];
      return date >= minDate && date <= maxDate;
    });
    
    // Calculate document scores (wins minus losses)
    const scores = {};
    docIds.forEach(id => { scores[id] = 0; });
    
    pairs.forEach((pair, index) => {
      const [doc1, doc2] = pair;
      const rank = rankings[index];
      
      if (rank === 1) {
        scores[doc1] += 1;
        scores[doc2] -= 1;
      } else if (rank === -1) {
        scores[doc1] -= 1;
        scores[doc2] += 1;
      }
    });
    
    // Find connections to the selected document
    const connections = [];
    pairs.forEach((pair, index) => {
      const [doc1, doc2] = pair;
      const rank = rankings[index];
      
      if (doc1 === selectedDoc && filteredDocIds.includes(doc2)) {
        connections.push({ 
          source: selectedDoc, 
          target: doc2, 
          rank, 
          sourceDate: dates[doc1],
          targetDate: dates[doc2]
        });
      } else if (doc2 === selectedDoc && filteredDocIds.includes(doc1)) {
        connections.push({ 
          source: selectedDoc, 
          target: doc1, 
          rank: -rank, // Flip the rank since we're flipping the direction
          sourceDate: dates[doc2],
          targetDate: dates[doc1]
        });
      }
    });
    
    // Set the filtered data
    setFilteredData({
      selected: selectedDoc,
      connections,
      allDocs: filteredDocIds.map(id => ({
        id,
        date: dates[id],
        score: scores[id]
      }))
    });
  }, [data, selectedDoc, dateRange]);

  // Render the graph when filtered data changes
  useEffect(() => {
    if (!filteredData || !svgRef.current) return;

    try {
      // Clear previous graph
      d3.select(svgRef.current).selectAll('*').remove();
      
      const svg = d3.select(svgRef.current);
      const width = svgRef.current.clientWidth || 800;
      const height = svgRef.current.clientHeight || 400;
      const centerX = width / 2;
      const centerY = height / 2;
      const radius = Math.min(width, height) * 0.4;
      
      // Extract dates for time scale
      const timeExtent = d3.extent(filteredData.allDocs, d => d.date);
      if (!timeExtent[0] || !timeExtent[1]) return;
      
      const timeScale = d3.scaleTime()
        .domain(timeExtent)
        .range([0, 2 * Math.PI]);
      
      // Score scale for node size
      const maxAbsScore = Math.max(1, ...filteredData.allDocs.map(d => Math.abs(d.score)));
      const scoreScale = d3.scaleLinear()
        .domain([-maxAbsScore, maxAbsScore])
        .range([5, 15]);
      
      // Create a time-based circular layout
      const nodes = filteredData.allDocs.map(doc => {
        const angle = timeScale(doc.date);
        return {
          id: doc.id,
          x: centerX + radius * Math.cos(angle),
          y: centerY + radius * Math.sin(angle),
          date: doc.date,
          score: doc.score,
          radius: scoreScale(doc.score),
          color: doc.score > 0 ? "#4CAF50" : doc.score < 0 ? "#F44336" : "#9E9E9E",
          isSelected: doc.id === filteredData.selected
        };
      });
      
      // Create a map for quick node lookup
      const nodeMap = {};
      nodes.forEach(node => { nodeMap[node.id] = node; });
      
      // Define arrow markers
      svg.append("defs").selectAll("marker")
        .data(["higher", "lower", "equal"])
        .enter().append("marker")
        .attr("id", d => d)
        .attr("viewBox", "0 -5 10 10")
        .attr("refX", 15)
        .attr("refY", 0)
        .attr("markerWidth", 6)
        .attr("markerHeight", 6)
        .attr("orient", "auto")
        .append("path")
        .attr("d", d => {
          if (d === "higher") return "M0,-5L10,0L0,5";
          if (d === "lower") return "M0,-5L10,0L0,5";
          return "M0,-5L10,-5L10,5L0,5"; // equal
        })
        .attr("fill", d => {
          if (d === "higher") return "#4CAF50";
          if (d === "lower") return "#F44336";
          return "#9E9E9E"; // equal
        });
      
      // Draw links
      const links = svg.append("g")
        .attr("class", "links")
        .selectAll("path")
        .data(filteredData.connections)
        .enter().append("path")
        .attr("d", d => {
          const source = nodeMap[d.source];
          const target = nodeMap[d.target];
          if (!source || !target) return "";
          
          // Calculate control points for a curved line
          const dx = target.x - source.x;
          const dy = target.y - source.y;
          const dr = Math.sqrt(dx * dx + dy * dy);
          
          // Determine control point offset (perpendicular to the line)
          const offset = dr * 0.2;
          const midX = (source.x + target.x) / 2;
          const midY = (source.y + target.y) / 2;
          const nx = -dy / dr; // normalized perpendicular vector
          const ny = dx / dr;
          
          const controlX = midX + offset * nx;
          const controlY = midY + offset * ny;
          
          return `M${source.x},${source.y} Q${controlX},${controlY} ${target.x},${target.y}`;
        })
        .attr("fill", "none")
        .attr("stroke", d => {
          if (d.rank === 1) return "#4CAF50"; // Higher
          if (d.rank === -1) return "#F44336"; // Lower
          return "#9E9E9E"; // Equal
        })
        .attr("stroke-width", 2)
        .attr("marker-end", d => {
          if (d.rank === 1) return "url(#higher)";
          if (d.rank === -1) return "url(#lower)";
          return "url(#equal)";
        })
        .attr("opacity", 0.7);
      
      // Draw timeline circle
      svg.append("circle")
        .attr("cx", centerX)
        .attr("cy", centerY)
        .attr("r", radius)
        .attr("fill", "none")
        .attr("stroke", "#ddd")
        .attr("stroke-dasharray", "3,3")
        .attr("stroke-width", 1);
      
      // Add time markers
      const years = d3.timeYear.range(
        new Date(timeExtent[0].getFullYear(), 0, 1),
        new Date(timeExtent[1].getFullYear() + 1, 0, 1),
        1
      );
      
      svg.append("g")
        .selectAll("text")
        .data(years)
        .enter().append("text")
        .attr("x", d => {
          const angle = timeScale(d);
          return centerX + (radius + 20) * Math.cos(angle);
        })
        .attr("y", d => {
          const angle = timeScale(d);
          return centerY + (radius + 20) * Math.sin(angle);
        })
        .attr("text-anchor", "middle")
        .attr("alignment-baseline", "middle")
        .attr("font-size", "10px")
        .attr("fill", "#666")
        .text(d => d.getFullYear());
      
      // Draw nodes
      const nodeElements = svg.append("g")
        .attr("class", "nodes")
        .selectAll("circle")
        .data(nodes)
        .enter().append("circle")
        .attr("cx", d => d.x)
        .attr("cy", d => d.y)
        .attr("r", d => d.isSelected ? d.radius * 1.5 : d.radius)
        .attr("fill", d => d.isSelected ? "#FFC107" : d.color)
        .attr("stroke", d => d.isSelected ? "#FF6F00" : "white")
        .attr("stroke-width", d => d.isSelected ? 3 : 1)
        .style("cursor", "pointer")
        .on("click", (event, d) => {
          setSelectedDoc(d.id);
        })
        .on("mouseover", (event, d) => {
          setHoveredNode(d);
          
          // Show tooltip
          const tooltip = d3.select(tooltipRef.current);
          tooltip.style("opacity", 1)
            .style("left", (event.pageX + 10) + "px")
            .style("top", (event.pageY - 10) + "px");
        })
        .on("mouseout", () => {
          setHoveredNode(null);
          
          // Hide tooltip
          d3.select(tooltipRef.current).style("opacity", 0);
        });
      
      // Draw date labels
      svg.append("g")
        .attr("class", "labels")
        .selectAll("text")
        .data(nodes)
        .enter().append("text")
        .attr("x", d => d.x)
        .attr("y", d => d.y + d.radius + 10)
        .attr("text-anchor", "middle")
        .attr("font-size", "9px")
        .attr("fill", d => d.isSelected ? "#FF6F00" : "#333")
        .attr("font-weight", d => d.isSelected ? "bold" : "normal")
        .text(d => formatDate(d.date));
    } catch (error) {
      console.error("Error rendering graph:", error);
    }
  }, [filteredData]);
  
  // Format date for display
  const formatDate = (date) => {
    if (!date) return "";
    return date.toLocaleDateString('en-US', { 
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  };
  
  // Handle document selection
  const handleDocumentSelect = (e) => {
    setSelectedDoc(e.target.value);
  };
  
  // Handle data source selection
  const handleDataSourceSelect = (e) => {
    const sourceId = e.target.value;
    setSelectedDataSourceId(sourceId);
  };
  
  // Handle slider change for date range
  const handleSliderChange = (values) => {
    if (!data) return;
    
    setSliderValues(values);
    
    // Extract all dates
    const allDates = Object.values(data.dates).map(d => d.getTime());
    const minTime = Math.min(...allDates);
    const maxTime = Math.max(...allDates);
    const timeRange = maxTime - minTime;
    
    // Calculate new date range based on slider values
    const newMinDate = new Date(minTime + (timeRange * values[0]) / 100);
    const newMaxDate = new Date(minTime + (timeRange * values[1]) / 100);
    
    setDateRange([newMinDate, newMaxDate]);
  };
  
  if (isLoading) {
    return (
      <div className="card w-full h-full border rounded shadow-lg">
        <div className="card-content flex items-center justify-center h-64 p-6">
          <div className="text-center">
            <div className="loader mx-auto mb-4"></div>
            <p>Loading data...</p>
          </div>
        </div>
      </div>
    );
  }
  
  if (!data || dataSources.length === 0) {
    return (
      <div className="card w-full h-full border rounded shadow-lg">
        <div className="card-content flex items-center justify-center h-64 p-6">
          <div className="text-center text-red-500">
            <p>Error loading data. Please check if the server is running at http://localhost:3001</p>
          </div>
        </div>
      </div>
    );
  }
  
  return (
    <div className="card w-full h-full border rounded shadow-lg">
      <div className="card-header p-4 border-b">
        <h2 className="card-title text-xl font-bold">Document Ranking Visualization</h2>
      </div>
      <div className="card-content p-4">
        <div className="mb-6 grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium mb-1">Data Source:</label>
            <select 
              className="w-full p-2 border rounded" 
              value={selectedDataSourceId || ""}
              onChange={handleDataSourceSelect}
            >
              {dataSources.map(source => (
                <option key={source.id} value={source.id}>
                  {source.name}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Selected Document:</label>
            <select 
              className="w-full p-2 border rounded" 
              value={selectedDoc || ""}
              onChange={handleDocumentSelect}
            >
              {data.docIds.map(id => (
                <option key={id} value={id}>
                  {formatDate(data.dates[id])}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">
              Date Range: {formatDate(dateRange[0])} - {formatDate(dateRange[1])}
            </label>
            <div className="mt-4 flex items-center">
              <input
                type="range"
                min="0"
                max="100"
                value={sliderValues[0]}
                onChange={(e) => handleSliderChange([parseInt(e.target.value), sliderValues[1]])}
                className="w-full mr-2"
              />
              <input
                type="range"
                min="0"
                max="100"
                value={sliderValues[1]}
                onChange={(e) => handleSliderChange([sliderValues[0], parseInt(e.target.value)])}
                className="w-full"
              />
            </div>
          </div>
        </div>
        
        <div className="relative border rounded-lg" style={{ width: '100%', height: '400px' }}>
          <svg ref={svgRef} width="100%" height="100%" style={{border: '1px solid #eee'}}></svg>
          
          <div 
            ref={tooltipRef} 
            className="absolute bg-white p-2 rounded shadow-lg border opacity-0 pointer-events-none z-10 transition-opacity"
            style={{ maxWidth: '200px' }}
          >
            {hoveredNode && (
              <div>
                <p className="text-sm">Date: {formatDate(hoveredNode.date)}</p>
                <p className="text-sm">Score: {hoveredNode.score}</p>
              </div>
            )}
          </div>
        </div>
        
        <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="flex items-center">
            <div className="w-4 h-4 bg-green-500 rounded-full mr-2"></div>
            <span className="text-sm">Higher Ranked</span>
          </div>
          <div className="flex items-center">
            <div className="w-4 h-4 bg-red-500 rounded-full mr-2"></div>
            <span className="text-sm">Lower Ranked</span>
          </div>
          <div className="flex items-center">
            <div className="w-4 h-4 bg-gray-500 rounded-full mr-2"></div>
            <span className="text-sm">Equal Ranking</span>
          </div>
        </div>
        
        {/* Selected document info section - simplified to just show date */}
        <div className="mt-6 border rounded-lg p-4">
          <h3 className="text-lg font-semibold mb-2">Selected Document</h3>
          <div className="bg-gray-50 p-4 rounded">
            <p className="text-lg">{formatDate(data.dates[selectedDoc])}</p>
          </div>
        </div>
        
        <div className="mt-4 text-sm text-gray-600 bg-blue-50 p-4 rounded-lg border border-blue-100">
          <p className="font-semibold mb-2">How to interpret this visualization:</p>
          <ul className="list-disc pl-5 space-y-2">
            <li>
              Each <strong>node</strong> on the circular timeline represents a document from a specific date.
            </li>
            <li>
              <strong>Node size</strong> indicates the overall ranking score - larger nodes have higher scores.
            </li>
            <li>
              <strong>Node color</strong> shows the score direction - green (positive), red (negative), or gray (neutral).
            </li>
            <li>
              <strong>Green arrows</strong> point to documents that are ranked higher than the selected document.
            </li>
            <li>
              <strong>Red arrows</strong> point to documents that are ranked lower than the selected document.
            </li>
            <li>
              <strong>Gray connections</strong> represent equal ranking between documents.
            </li>
            <li>
              Documents are arranged <strong>chronologically</strong> clockwise around the circle.
            </li>
            <li>
              You can <strong>click on any node</strong> to select that document and see its relationships.
            </li>
          </ul>
        </div>
      </div>
      <style jsx>{`
        .loader {
          border: 4px solid #f3f3f3;
          border-top: 4px solid #3498db;
          border-radius: 50%;
          width: 40px;
          height: 40px;
          animation: spin 2s linear infinite;
        }
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

export default DocumentRankingGraph;