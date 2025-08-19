import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { AnimationProvider } from '../lib/AnimationProvider'
import QueryProvider from '../components/QueryProvider'
import ServiceWorkerProvider from '../components/ServiceWorkerProvider'
import AccessibilityProvider from '../components/AccessibilityProvider'
// Temporarily removed to fix runtime errors
// import StoreProvider from '../components/StoreProvider'
import NotificationSystem from '../components/NotificationSystem'
import { BackupPanel, UndoRedoToolbar, HistoryViewer } from '../components/StateManagement'

const inter = Inter({ 
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap',
})

export const metadata: Metadata = {
  title: 'MasterX - Quantum Intelligence Platform',
  description: 'Advanced AI conversation platform with quantum intelligence capabilities',
  keywords: ['AI', 'Machine Learning', 'Quantum Intelligence', 'Chat', 'Education'],
  authors: [{ name: 'MasterX AI Team' }],
  creator: 'MasterX',
  publisher: 'MasterX',
  robots: 'index, follow',
  manifest: '/manifest.json',
  openGraph: {
    title: 'MasterX - Quantum Intelligence Platform',
    description: 'Experience the future of AI conversation with our quantum intelligence platform',
    type: 'website',
    locale: 'en_US',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'MasterX - Quantum Intelligence Platform',
    description: 'Experience the future of AI conversation with our quantum intelligence platform',
  },
  icons: {
    icon: '/favicon.ico',
    apple: '/apple-touch-icon.png',
  },
}

export const viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
  themeColor: '#6366f1',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body className={`${inter.className} antialiased`}>
        <QueryProvider>
          <div id="root">
            {children}
          </div>
        </QueryProvider>
      </body>
    </html>
  )
}
