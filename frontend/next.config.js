/** @type {import('next').NextConfig} */
const path = require('path')

const nextConfig = {
  reactStrictMode: true,
  typescript: {
    ignoreBuildErrors: false
  },
  images: {
    domains: ['avatars.githubusercontent.com'],
    formats: ['image/webp', 'image/avif'],
  },

  // Bundle optimization
  compiler: {
    removeConsole: process.env.NODE_ENV === 'production',
  },

  // Webpack optimizations
  webpack: (config, { dev, isServer }) => {
    // Import path aliases
    config.resolve.alias['@'] = path.resolve(__dirname, '.')

    // Bundle optimization for production
    if (!dev && !isServer) {
      // Split chunks for better caching
      config.optimization.splitChunks = {
        chunks: 'all',
        cacheGroups: {
          // Vendor libraries
          vendor: {
            test: /[\\/]node_modules[\\/]/,
            name: 'vendors',
            chunks: 'all',
            priority: 10,
          },
          // React and React-DOM
          react: {
            test: /[\\/]node_modules[\\/](react|react-dom)[\\/]/,
            name: 'react',
            chunks: 'all',
            priority: 20,
          },
          // Animation libraries
          animations: {
            test: /[\\/]node_modules[\\/](framer-motion|lottie-react)[\\/]/,
            name: 'animations',
            chunks: 'all',
            priority: 15,
          },
          // UI libraries
          ui: {
            test: /[\\/]node_modules[\\/](lucide-react|@headlessui|@heroicons)[\\/]/,
            name: 'ui',
            chunks: 'all',
            priority: 15,
          },
          // Query and state management
          state: {
            test: /[\\/]node_modules[\\/](@tanstack|zustand)[\\/]/,
            name: 'state',
            chunks: 'all',
            priority: 15,
          },
          // Code highlighting and markdown
          content: {
            test: /[\\/]node_modules[\\/](react-syntax-highlighter|react-markdown|remark-gfm)[\\/]/,
            name: 'content',
            chunks: 'all',
            priority: 15,
          },
        },
      }

      // Tree shaking optimization
      config.optimization.usedExports = true
      config.optimization.sideEffects = false
    }

    return config
  },
  // Ensure proper routing
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: process.env.REACT_APP_BACKEND_URL + '/api/:path*'
      }
    ]
  }
}

module.exports = nextConfig