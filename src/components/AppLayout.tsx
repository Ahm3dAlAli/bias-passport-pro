import { NavLink, Outlet } from 'react-router-dom';
import { Fingerprint, BarChart3, FlaskConical, Plane, CreditCard, Shield, Wrench, ScanLine } from 'lucide-react';

const NAV = [
  { to: '/report', icon: BarChart3, label: 'Report' },
  { to: '/lab', icon: FlaskConical, label: 'Lab' },
  { to: '/airport', icon: Plane, label: 'Airport' },
  { to: '/eid', icon: ScanLine, label: 'E-ID' },
  { to: '/banking', icon: CreditCard, label: 'Banking' },
  { to: '/eu-ai-act', icon: Shield, label: 'EU AI' },
  { to: '/mitigation', icon: Wrench, label: 'Fix Bias' },
];

export default function AppLayout() {
  return (
    <div className="flex min-h-screen bg-observatory-bg">
      {/* Sidebar */}
      <aside className="w-16 md:w-56 glass flex flex-col items-center md:items-stretch py-4 px-1 md:px-3 border-r border-observatory-border shrink-0 sticky top-0 h-screen overflow-y-auto">
        <NavLink to="/" className="flex items-center gap-2 px-2 py-3 mb-4">
          <div className="w-8 h-8 rounded-lg bg-observatory-accent/20 flex items-center justify-center shrink-0">
            <Fingerprint className="w-5 h-5 text-observatory-accent" />
          </div>
          <span className="hidden md:block font-mono font-bold text-sm gradient-text">Fingerprint²</span>
        </NavLink>
        <nav className="flex flex-col gap-1 w-full">
          {NAV.map(n => (
            <NavLink
              key={n.to}
              to={n.to}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all ${
                  isActive
                    ? 'bg-observatory-accent/15 text-observatory-accent glow-accent'
                    : 'text-observatory-text-muted hover:text-observatory-text hover:bg-observatory-surface-alt'
                }`
              }
            >
              <n.icon className="w-4 h-4 shrink-0" />
              <span className="hidden md:block">{n.label}</span>
            </NavLink>
          ))}
        </nav>
        <div className="mt-auto pt-4 hidden md:block">
          <div className="text-[10px] text-observatory-text-dim text-center font-mono">
            EU AI Act Compliant
          </div>
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 overflow-y-auto">
        <Outlet />
      </main>
    </div>
  );
}
