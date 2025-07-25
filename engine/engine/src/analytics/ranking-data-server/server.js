const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = 3001;

// Enable CORS for all routes
app.use(cors());

// Endpoint to get a list of available data sources
app.get('/api/datasources', (req, res) => {
  const dataDir = '/Users/dzz1th/Job/mgi/Soroka/data/pc_data/ranked_pairs';
  
  fs.readdir(dataDir, (err, files) => {
    if (err) {
      console.error('Error reading directory:', err);
      return res.status(500).json({ error: 'Failed to read data directory' });
    }
    
    // Filter for JSON files and extract data source names
    const dataSources = files
      .filter(file => file.endsWith('_ranked_pairs.json'))
      .map(file => {
        const name = file.replace('_ranked_pairs.json', '');
        return {
          id: name,
          name: name.split('_').map(word => 
            word.charAt(0).toUpperCase() + word.slice(1)
          ).join(' '),
          filePath: file
        };
      });
    
    res.json(dataSources);
  });
});

// Endpoint to get data for a specific source
app.get('/api/data/:sourceId', (req, res) => {
  const sourceId = req.params.sourceId;
  const filePath = path.join(
    '/Users/dzz1th/Job/mgi/Soroka/data/pc_data/ranked_pairs',
    `${sourceId}_ranked_pairs.json`
  );
  
  fs.readFile(filePath, 'utf8', (err, data) => {
    if (err) {
      console.error('Error reading file:', err);
      return res.status(500).json({ error: `Failed to read data for ${sourceId}` });
    }
    
    try {
      const jsonData = JSON.parse(data);
      res.json(jsonData);
    } catch (parseErr) {
      console.error('Error parsing JSON:', parseErr);
      res.status(500).json({ error: 'Failed to parse JSON data' });
    }
  });
});

// Start the server
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});