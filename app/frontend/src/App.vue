<template>
  <div class="app">
    <div class="layout">
      <!-- Left panel -->
      <div class="left-panel">
        <SeriesSelector 
          :seriesList="seriesList" 
          :selectedSeries="selectedSeries"
          @select-series="selectSeries"
        />
      </div>
      
      <!-- Main content -->
      <div class="main-content">
        <TimeSeriesGraph 
          :selectedSeries="selectedSeries"
          :description="timeSeriesData.description"
          :data="timeSeriesData.data"
          @point-click="handlePointClick"
        />
      </div>
    </div>
    
    <!-- Side panel -->
    <SidePanel 
      :isOpen="isDetailPanelOpen"
      :pointDetails="pointDetails"
      @close="closeDetailPanel"
    />
    
    <!-- Debug info if there are errors -->
    <div v-if="debug.error" class="debug-info">
      <h3>Debug Info:</h3>
      <pre>{{ debug.error }}</pre>
      <button @click="fetchTimeSeriesData">Retry Fetch</button>
    </div>
  </div>
</template>

<script>
import SeriesSelector from './components/SeriesSelector.vue';
import TimeSeriesGraph from './components/TimeSeriesGraph.vue';
import SidePanel from './components/SidePanel.vue';
import api from './services/api';

export default {
  components: {
    SeriesSelector,
    TimeSeriesGraph,
    SidePanel
  },
  
  data() {
    return {
      seriesList: [],
      selectedSeries: 'Inflation',
      timeSeriesData: {
        description: '',
        data: []
      },
      selectedPoint: null,
      pointDetails: null,
      isDetailPanelOpen: false,
      // Add debug info
      debug: {
        error: null
      }
    };
  },
  
  async created() {
    try {
      console.log('Initializing app...');
      // Fetch series list
      console.log('Fetching series list...');
      this.seriesList = await api.getSeriesList();
      console.log('Series list:', this.seriesList);
      
      // Fetch initial series data
      await this.fetchTimeSeriesData();
    } catch (error) {
      console.error('Error initializing app:', error);
      this.debug.error = `Init error: ${error.message}`;
      
      // Fallback to hardcoded series list if API fails
      if (this.seriesList.length === 0) {
        this.seriesList = ['Inflation', 'Labour Market', 'Interest Rate Guidance'];
      }
    }
  },
  
  methods: {
    async selectSeries(series) {
      this.selectedSeries = series;
      await this.fetchTimeSeriesData();
      this.closeDetailPanel();
    },
    
    async fetchTimeSeriesData() {
      try {
        console.log(`Fetching time series data for: ${this.selectedSeries}`);
        this.timeSeriesData = await api.getTimeSeries(this.selectedSeries);
        console.log('Time series data:', this.timeSeriesData);
        this.debug.error = null;
      } catch (error) {
        console.error('Error fetching time series data:', error);
        this.debug.error = `API error: ${error.message}`;
        this.timeSeriesData = { 
          description: 'Failed to load data',
          data: [] 
        };
      }
    },
    
    async handlePointClick(point) {
      try {
        console.log(`Fetching details for point: ${point.id}`);
        this.selectedPoint = point;
        this.pointDetails = await api.getPointDetails(point.id);
        console.log('Point details:', this.pointDetails);
        this.isDetailPanelOpen = true;
      } catch (error) {
        console.error('Error fetching point details:', error);
        this.debug.error = `API error: ${error.message}`;
      }
    },
    
    closeDetailPanel() {
      this.isDetailPanelOpen = false;
      this.selectedPoint = null;
      this.pointDetails = null;
    }
  }
};
</script>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: 'Avenir', Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  color: #2c3e50;
}

.app {
  width: 100%;
  height: 100vh;
  overflow: hidden;
  position: relative;
}

.layout {
  display: flex;
  height: 100%;
}

.left-panel {
  width: 250px;
  height: 100%;
  border-right: 1px solid #eaeaea;
}

.main-content {
  flex: 1;
  padding: 24px;
  background-color: #f5f7fa;
  height: 100%;
  overflow: auto;
}

.debug-info {
  position: fixed;
  bottom: 0;
  left: 0;
  right: 0;
  background-color: #ffebee;
  padding: 1rem;
  border-top: 1px solid #f44336;
  z-index: 1000;
}

.debug-info pre {
  font-family: monospace;
  margin: 0.5rem 0;
  background-color: rgba(0, 0, 0, 0.05);
  padding: 0.5rem;
  border-radius: 4px;
  overflow: auto;
  max-height: 150px;
}

.debug-info button {
  background-color: #f44336;
  color: white;
  border: none;
  padding: 0.5rem 1rem;
  border-radius: 4px;
  cursor: pointer;
}

.debug-info button:hover {
  background-color: #d32f2f;
}
</style>