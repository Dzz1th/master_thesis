<template>
  <div class="time-series-graph">
    <h1>{{ selectedSeries }}</h1>
    <p class="description">{{ description }}</p>
    
    <div class="chart-container" ref="chartContainer">
      <svg ref="chart" class="chart"></svg>
    </div>
  </div>
</template>

<script>
import * as d3 from 'd3';

export default {
  props: {
    selectedSeries: {
      type: String,
      required: true
    },
    description: {
      type: String,
      default: ''
    },
    data: {
      type: Array,
      required: true
    }
  },
  
  data() {
    return {
      chart: null,
      svg: null,
      width: 0,
      height: 0,
      margin: { top: 20, right: 30, bottom: 30, left: 50 }
    };
  },
  
  watch: {
    data: {
      handler() {
        this.updateChart();
      },
      deep: true
    }
  },
  
  mounted() {
    window.addEventListener('resize', this.handleResize);
    this.initChart();
    this.updateChart();
  },
  
  beforeUnmount() {
    window.removeEventListener('resize', this.handleResize);
  },
  
  methods: {
    initChart() {
      const container = this.$refs.chartContainer;
      this.width = container.clientWidth - this.margin.left - this.margin.right;
      this.height = container.clientHeight - this.margin.top - this.margin.bottom;
      
      this.svg = d3.select(this.$refs.chart)
        .attr('width', '100%')
        .attr('height', '100%')
        .attr('viewBox', `0 0 ${this.width + this.margin.left + this.margin.right} ${this.height + this.margin.top + this.margin.bottom}`)
        .append('g')
        .attr('transform', `translate(${this.margin.left},${this.margin.top})`);
    },
    
    updateChart() {
      if (!this.data || this.data.length === 0) return;
      
      // Clear previous chart
      this.svg.selectAll('*').remove();
      
      // Parse dates and create scales
      const parseDate = d3.timeParse('%Y-%m-%d');
      const data = this.data.map(d => ({
        ...d,
        date: parseDate(d.date)
      })).sort((a, b) => a.date - b.date);
      
      const xScale = d3.scaleTime()
        .domain(d3.extent(data, d => d.date))
        .range([0, this.width]);
      
      const yScale = d3.scaleLinear()
        .domain([0, d3.max(data, d => d.value) * 1.1])
        .range([this.height, 0]);
      
      // Add axes
      this.svg.append('g')
        .attr('class', 'x-axis')
        .attr('transform', `translate(0,${this.height})`)
        .call(d3.axisBottom(xScale).ticks(6).tickFormat(d3.timeFormat('%b %Y')));
      
      this.svg.append('g')
        .attr('class', 'y-axis')
        .call(d3.axisLeft(yScale));
      
      // Add line
      const line = d3.line()
        .x(d => xScale(d.date))
        .y(d => yScale(d.value))
        .curve(d3.curveMonotoneX);
      
      this.svg.append('path')
        .datum(data)
        .attr('class', 'line')
        .attr('fill', 'none')
        .attr('stroke', '#1890ff')
        .attr('stroke-width', 2)
        .attr('d', line);
      
      // Add dots
      const dots = this.svg.selectAll('.dot')
        .data(data)
        .enter()
        .append('circle')
        .attr('class', 'dot')
        .attr('cx', d => xScale(d.date))
        .attr('cy', d => yScale(d.value))
        .attr('r', 5)
        .attr('fill', '#1890ff')
        .style('cursor', 'pointer');
      
      // Create tooltip
      const tooltip = d3.select('body').append('div')
        .attr('class', 'tooltip')
        .style('opacity', 0)
        .style('position', 'absolute')
        .style('background-color', 'white')
        .style('border', '1px solid #ddd')
        .style('border-radius', '4px')
        .style('padding', '10px')
        .style('box-shadow', '0 2px 8px rgba(0,0,0,0.15)')
        .style('pointer-events', 'none');
      
      // Add interactions
      dots.on('mouseover', (event, d) => {
        d3.select(event.currentTarget)
          .transition()
          .duration(200)
          .attr('r', 8);
        
        tooltip.transition()
          .duration(200)
          .style('opacity', 0.9);
        
        tooltip.html(`
          <div>
            <strong>${d3.timeFormat('%B %d, %Y')(d.date)}</strong><br/>
            ${this.selectedSeries}: ${d.value}%<br/>
            <small>${d.short_desc}</small><br/>
            <span style="color: #1890ff; font-size: 12px;">Click for details</span>
          </div>
        `)
          .style('left', (event.pageX + 10) + 'px')
          .style('top', (event.pageY - 28) + 'px');
      })
      .on('mouseout', (event) => {
        d3.select(event.currentTarget)
          .transition()
          .duration(200)
          .attr('r', 5);
        
        tooltip.transition()
          .duration(500)
          .style('opacity', 0);
      })
      .on('click', (event, d) => {
        this.$emit('point-click', d);
      });
    },
    
    handleResize() {
      this.initChart();
      this.updateChart();
    }
  }
}
</script>

<style scoped>
.time-series-graph {
  background-color: white;
  border-radius: 8px;
  box-shadow: 0 0 10px rgba(0,0,0,0.1);
  padding: 1.5rem;
  height: 100%;
  display: flex;
  flex-direction: column;
}

h1 {
  font-size: 1.5rem;
  font-weight: bold;
  margin-bottom: 0.5rem;
}

.description {
  color: #666;
  margin-bottom: 1.5rem;
}

.chart-container {
  flex-grow: 1;
  position: relative;
}

.chart {
  width: 100%;
  height: 100%;
}

:deep(.line) {
  fill: none;
  stroke: #1890ff;
  stroke-width: 2px;
}

:deep(.dot) {
  fill: #1890ff;
  transition: all 0.2s;
}

:deep(.dot:hover) {
  fill: #096dd9;
}

:deep(.x-axis path),
:deep(.y-axis path),
:deep(.x-axis line),
:deep(.y-axis line) {
  stroke: #e0e0e0;
}

:deep(.x-axis text),
:deep(.y-axis text) {
  font-size: 12px;
  fill: #666;
}
</style>