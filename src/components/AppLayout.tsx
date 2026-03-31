import { NavLink, Outlet, useLocation } from 'react-router-dom';
import { Fingerprint, BarChart3, FlaskConical, Plane, Shield, Wrench, Camera, Menu, X } from 'lucide-react';
import { useState, useEffect } from 'react';

const NAV = [
  { to: '/report', icon: BarChart3, label: 'Bias Report' },
  { to: '/lab', icon: FlaskConical, label: 'Lab' },
  { to: '/airport', icon: Plane, label: 'Airport' },
  { to: '/scan', icon: Camera, label: 'Scan Face' },
  { to: '/eu-ai-act', icon: Shield, label: 'EU AI Act' },
  { to: '/mitigation', icon: Wrench, label: 'Fix Bias' },
];

export default function AppLayout() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const location = useLocation();

  // Close sidebar on route change (mobile)
  useEffect(() => {
    setSidebarOpen(false);
  }, [location.pathname]);

  return (
    <div className="flex min-h-screen bg-observatory-bg">
      {/* Mobile overlay */}
      {sidebarOpen && (
        <div className="fixed inset-0 bg-black/60 z-40 md:hidden" onClick={() => setSidebarOpen(false)} />
      )}

      {/* Sidebar */}
      <aside className={`
        fixed md:sticky top-0 h-screen z-50 w-64 border-r border-observatory-border/50 flex flex-col shrink-0 bg-observatory-surface/95 backdrop-blur-xl
        transition-transform duration-200 ease-out
        ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}
        md:translate-x-0
      `}>
        <div className="flex items-center justify-between border-b border-observatory-border/30">
          <NavLink to="/" className="flex items-center gap-3 px-5 py-5 flex-1">
            <div className="w-9 h-9 rounded-xl bg-observatory-accent/15 flex items-center justify-center shrink-0">
              <Fingerprint className="w-5 h-5 text-observatory-accent" />
            </div>
            <div>
              <span className="font-mono font-bold text-sm gradient-text">Fingerprint²</span>
              <p className="text-[10px] text-observatory-text-dim font-mono">Bias Benchmarking</p>
            </div>
          </NavLink>
          <button onClick={() => setSidebarOpen(false)} className="md:hidden p-3 text-observatory-text-dim">
            <X className="w-5 h-5" />
          </button>
        </div>

        <nav className="flex flex-col gap-0.5 px-3 py-4 flex-1">
          <p className="text-[10px] font-mono text-observatory-text-dim px-3 mb-2 uppercase tracking-widest">Modules</p>
          {NAV.map(n => (
            <NavLink
              key={n.to}
              to={n.to}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all ${
                  isActive
                    ? 'bg-observatory-accent/10 text-observatory-accent'
                    : 'text-observatory-text-muted hover:text-observatory-text hover:bg-observatory-surface-alt/50'
                }`
              }
            >
              <n.icon className="w-4 h-4 shrink-0" />
              <span>{n.label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="px-5 py-4 border-t border-observatory-border/30">
          <div className="flex items-center gap-2 text-[10px] text-observatory-text-dim font-mono">
            <Shield className="w-3 h-3 text-observatory-success" />
            EU AI Act Compliant
          </div>
        </div>
      </aside>

      {/* Main content */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Mobile header */}
        <header className="md:hidden sticky top-0 z-30 flex items-center gap-3 px-4 py-3 bg-observatory-surface/90 backdrop-blur-lg border-b border-observatory-border/30">
          <button onClick={() => setSidebarOpen(true)} className="p-1.5 rounded-lg hover:bg-observatory-surface-alt text-observatory-text-muted">
            <Menu className="w-5 h-5" />
          </button>
          <div className="flex items-center gap-2">
            <Fingerprint className="w-4 h-4 text-observatory-accent" />
            <span className="font-mono font-bold text-xs gradient-text">Fingerprint²</span>
          </div>
        </header>

        <main className="flex-1 overflow-y-auto">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
