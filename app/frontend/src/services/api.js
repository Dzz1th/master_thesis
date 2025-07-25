import axios from 'axios';

const API_URL = 'http://localhost:8000/api';

export default {
  async getSeriesList() {
    const response = await axios.get(`${API_URL}/series-list`);
    return response.data;
  },
  
  async getTimeSeries(name) {
    const response = await axios.get(`${API_URL}/time-series`, {
      params: { name }
    });
    return response.data;
  },
  
  async getPointDetails(pointId) {
    const response = await axios.get(`${API_URL}/point-details/${pointId}`);
    return response.data;
  },
  
  async loadTranscript() {
    const response = await axios.get(`${API_URL}/load-transcript`);
    return response.data;
  }
};