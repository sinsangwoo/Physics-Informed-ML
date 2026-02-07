import { ReactNode } from 'react'
import { Activity, Zap, Code2 } from 'lucide-react'

interface LayoutProps {
  children: ReactNode
}

export function Layout({ children }: LayoutProps) {
  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/50 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Activity className="w-8 h-8 text-primary-500" />
              <div>
                <h1 className="text-xl font-bold">Physics-Informed ML</h1>
                <p className="text-xs text-slate-400">Real-Time Neural PDE Solver</p>
              </div>
            </div>
            
            <div className="flex items-center gap-6 text-sm">
              <div className="flex items-center gap-2">
                <Zap className="w-4 h-4 text-yellow-500" />
                <span className="text-slate-400">1000x Speedup</span>
              </div>
              <div className="flex items-center gap-2">
                <Code2 className="w-4 h-4 text-blue-500" />
                <span className="text-slate-400">FNO + PINN</span>
              </div>
              <a 
                href="https://github.com/sinsangwoo/Physics-Informed-ML" 
                target="_blank"
                rel="noopener noreferrer"
                className="text-slate-400 hover:text-slate-100 transition-colors"
              >
                GitHub
              </a>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 container mx-auto px-4 py-6">
        {children}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800 bg-slate-900/30">
        <div className="container mx-auto px-4 py-4 text-center text-sm text-slate-500">
          <p>Built with React + Three.js | Powered by Neural Operators</p>
        </div>
      </footer>
    </div>
  )
}
