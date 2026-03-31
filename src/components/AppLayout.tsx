import { NavLink, Outlet } from 'react-router-dom';
import { Fingerprint, BarChart3, FlaskConical, Plane, Shield, Wrench, Camera } from 'lucide-react';

const NAV = [
  { to: '/report', icon: BarChart3, label: 'Bias Report' },
  { to: '/lab', icon: FlaskConical, label: 'Lab' },
  { to: '/airport', icon: Plane, label: 'Airport' },
  { to: '/scan', icon: Camera, label: 'Scan Face' },
  { to: '/eid', icon: ScanLine, label: 'E-ID' },
  { to: '/eu-ai-act', icon: Shield, label: 'EU AI Act' },
  { to: '/mitigation', icon: Wrench, label: 'Fix Bias' },
];

export default function AppLayout() {
  return (
    <div className="flex min-h-screen bg-observatory-bg">
      <aside className="w-64 border-r border-observatory-border/50 flex flex-col shrink-0 sticky top-0 h-screen bg-observatory-surface/50">
        <NavLink to="/" className="flex items-center gap-3 px-5 py-5 border-b border-observatory-border/30">
          <div className="w-9 h-9 rounded-xl bg-observatory-accent/15 flex items-center justify-center shrink-0">
            <Fingerprint className="w-5 h-5 text-observatory-accent" />
          </div>
          <div>
            <span className="font-mono font-bold text-sm gradient-text">Fingerprint²</span>
            <p className="text-[10px] text-observatory-text-dim font-mono">Bias Benchmarking</p>
          </div>
        </NavLink>

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

      <main className="flex-1 overflow-y-auto">
        <Outlet />
      </main>
    </div>
  );
}
