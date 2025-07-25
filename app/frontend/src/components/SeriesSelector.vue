<template>
  <div class="series-selector">
    <h2>Time Series</h2>
    <ul>
      <li 
        v-for="item in seriesList" 
        :key="item" 
        :class="{ 'active': selectedSeries === item }"
        @click="$emit('select-series', item)"
      >
        {{ item }}
        <span v-if="selectedSeries === item" class="chevron">›</span>
      </li>
    </ul>
  </div>
</template>

<script>
export default {
  props: {
    seriesList: {
      type: Array,
      required: true
    },
    selectedSeries: {
      type: String,
      required: true
    }
  }
}
</script>

<style scoped>
.series-selector {
  background-color: white;
  height: 100%;
  padding: 1rem;
  box-shadow: 0 0 10px rgba(0,0,0,0.1);
}

h2 {
  font-size: 1.5rem;
  font-weight: bold;
  margin-bottom: 1rem;
}

ul {
  list-style: none;
  padding: 0;
}

li {
  padding: 0.75rem 1rem;
  margin-bottom: 0.5rem;
  border-radius: 4px;
  cursor: pointer;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

li:hover {
  background-color: #f5f5f5;
}

li.active {
  background-color: #e6f7ff;
  color: #1890ff;
  font-weight: 500;
}

.chevron {
  font-size: 1.2rem;
}
</style>