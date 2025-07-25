<template>
  <div class="side-panel" :class="{ 'open': isOpen }">
    <div class="header">
      <h2>Details</h2>
      <button class="close-btn" @click="$emit('close')">×</button>
    </div>
    
    <div v-if="pointDetails" class="content">
      <h3>{{ pointDetails.title }}</h3>
      <p class="date">{{ formatDate(pointDetails.date) }}</p>
      
      <div class="value-badge">
        {{ pointDetails.value }}%
      </div>
      
      <!-- Structured Summary section -->
      <div class="summary-section">
        <h4>Summary</h4>
        <div class="structured-summary">
          <div v-if="pointDetails.structured_summary" class="summary-categories">
            <div v-for="(content, category) in pointDetails.structured_summary" 
                 :key="category" 
                 class="summary-category">
              <h5 class="category-title" :class="getCategoryClass(category)">
                {{ formatCategoryName(category) }}
              </h5>
              <p class="category-content">{{ content }}</p>
            </div>
          </div>
        </div>
      </div>
      
      <!-- Transcript section -->
      <div class="transcript-section">
        <h4>Transcript</h4>
        
        <!-- Legend -->
        <div class="highlight-legend">
          <div v-for="category in pointDetails.highlightCategories" :key="category.id" class="legend-item">
            <div class="color-box" :style="{ backgroundColor: category.color }"></div>
            <span>{{ category.label }}</span>
          </div>
        </div>
        
        <!-- Q&A Format Transcript -->
        <div class="transcript-content">
          <div v-for="(qa, qaIndex) in pointDetails.transcript" :key="qaIndex" class="qa-block">
            <div class="speaker">{{ qa.speaker }}:</div>
            <div class="statement">
              <div v-for="(paragraph, pIndex) in qa.content" :key="`p-${qaIndex}-${pIndex}`" 
                   :class="{ 'paragraph': true, 'mt-4': pIndex > 0 }">
                <template v-for="(segment, sIndex) in paragraph.content" :key="`${qaIndex}-${pIndex}-${sIndex}`">
                  <!-- No highlights -->
                  <span v-if="segment.highlights.length === 0">
                    {{ segment.text }}
                  </span>
                  
                  <!-- With highlights -->
                  <span v-else
                        class="highlighted-text"
                        :style="getHighlightStyle(segment.highlights)"
                        :title="getHighlightTitle(segment.highlights)">
                    {{ segment.text }}
                  </span>
                </template>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  props: {
    isOpen: {
      type: Boolean,
      default: false
    },
    pointDetails: {
      type: Object,
      default: null
    }
  },
  
  methods: {
    formatDate(dateStr) {
      const date = new Date(dateStr);
      return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
      });
    },
    
    formatCategoryName(category) {
      // Convert camelCase or snake_case to Title Case With Spaces
      return category
        .replace(/_/g, ' ')
        .replace(/([A-Z])/g, ' $1')
        .replace(/^./, str => str.toUpperCase())
        .trim();
    },
    
    getCategoryClass(category) {
      // Map category names to CSS classes for styling
      const categoryMap = {
        'employment': 'employment-category',
        'inflation': 'inflation-category',
        'interest_rate_projections': 'interest-rate-category',
        'balance_sheet_projections': 'balance-sheet-category',
        'forward_guidance': 'forward-guidance-category'
      };
      
      return categoryMap[category] || '';
    },
    
    getHighlightStyle(highlightIds) {
      // Get colors for all highlight IDs
      const colors = highlightIds.map(id => 
        this.pointDetails.highlightCategories.find(c => c.id === id)?.color
      ).filter(Boolean);
      
      if (colors.length === 0) return {};
      if (colors.length === 1) return { backgroundColor: colors[0] };
      
      // Create gradient for multiple highlights
      return {
        backgroundImage: `linear-gradient(135deg, ${colors.join(', ')})`
      };
    },
    
    getHighlightTitle(highlightIds) {
      return highlightIds.map(id => 
        this.pointDetails.highlightCategories.find(c => c.id === id)?.label
      ).filter(Boolean).join(', ');
    }
  }
}
</script>

<style scoped>
.side-panel {
  position: fixed;
  top: 0;
  right: 0;
  width: 50%; 
  height: 100%;
  background-color: white;
  box-shadow: -2px 0 10px rgba(0,0,0,0.1);
  padding: 1.5rem;
  transform: translateX(100%);
  transition: transform 0.3s ease-out;
  z-index: 100;
  overflow-y: auto;
}

.side-panel.open {
  transform: translateX(0);
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.5rem;
}

h2 {
  font-size: 1.5rem;
  font-weight: bold;
  margin: 0;
}

.close-btn {
  background: none;
  border: none;
  font-size: 1.5rem;
  cursor: pointer;
  color: #999;
}

.close-btn:hover {
  color: #333;
}

h3 {
  font-size: 1.25rem;
  font-weight: 600;
  margin-bottom: 0.5rem;
}

.date {
  color: #666;
  font-size: 0.875rem;
  margin-bottom: 1.5rem;
}

.value-badge {
  display: inline-block;
  background-color: #e6f7ff;
  color: #1890ff;
  font-weight: bold;
  font-size: 1.25rem;
  padding: 0.5rem 1rem;
  border-radius: 4px;
  margin-bottom: 1.5rem;
}

.summary-section, .transcript-section {
  margin-bottom: 1.5rem;
}

h4 {
  font-size: 1.125rem;
  font-weight: 600;
  margin-bottom: 1rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid #eaeaea;
}

/* Structured Summary Styles */
.structured-summary {
  background-color: #f9f9f9;
  border-radius: 8px;
  padding: 1rem;
}

.summary-category {
  margin-bottom: 1.25rem;
}

.summary-category:last-child {
  margin-bottom: 0;
}

.category-title {
  font-size: 1rem;
  font-weight: 600;
  margin-bottom: 0.5rem;
  padding: 0.25rem 0.75rem;
  border-radius: 4px;
  display: inline-block;
}

.category-content {
  line-height: 1.6;
  margin-top: 0.5rem;
  text-align: justify;
}

/* Category-specific styles */
.employment-category {
  background-color: #C8E6C9;
  color: #1B5E20;
}

.inflation-category {
  background-color: #FFCDD2;
  color: #B71C1C;
}

.interest-rate-category {
  background-color: #BBDEFB;
  color: #0D47A1;
}

.balance-sheet-category {
  background-color: #D1C4E9;
  color: #4527A0;
}

.forward-guidance-category {
  background-color: #FFE0B2;
  color: #E65100;
}

/* Transcript Styles */
.highlight-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 16px;
  padding: 12px;
  background-color: #f5f5f5;
  border-radius: 4px;
}

.legend-item {
  display: flex;
  align-items: center;
  margin-right: 12px;
  margin-bottom: 4px;
}

.color-box {
  width: 16px;
  height: 16px;
  margin-right: 4px;
  border-radius: 2px;
}

.transcript-content {
  background-color: #f5f5f5;
  padding: 16px;
  border-radius: 4px;
  line-height: 1.6;
  font-size: 1rem;
}

.qa-block {
  margin-bottom: 1.5rem;
}

.qa-block:last-child {
  margin-bottom: 0;
}

.speaker {
  font-weight: bold;
  margin-bottom: 0.5rem;
  color: #1890ff;
}

.statement {
  padding-left: 1rem;
}

.paragraph {
  margin-bottom: 1rem;
}

.paragraph:last-child {
  margin-bottom: 0;
}

.highlighted-text {
  border-radius: 2px;
  padding: 0 2px;
}
</style>