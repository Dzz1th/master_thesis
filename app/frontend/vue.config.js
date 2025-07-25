module.exports = {
    devServer: {
      proxy: {
        '/api': {
          target: 'http://localhost:8000',
          changeOrigin: true,
          logLevel: 'debug'
        }
      }
    },
    lintOnSave: false
  }