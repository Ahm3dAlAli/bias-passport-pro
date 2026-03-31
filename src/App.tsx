import { Suspense, lazy, type ReactNode } from 'react';
import { Routes, Route } from 'react-router-dom';
import AppLayout from './components/AppLayout';
import RouteErrorBoundary from './components/RouteErrorBoundary';

const LandingPage = lazy(() => import('./pages/LandingPage'));
const BiasReportPage = lazy(() => import('./pages/BiasReportPage'));
const LabPage = lazy(() => import('./pages/LabPage'));
const AirportPage = lazy(() => import('./pages/AirportPage'));

const EUAIActPage = lazy(() => import('./pages/EUAIActPage'));
const MitigationPage = lazy(() => import('./pages/MitigationPage'));
const ScanPage = lazy(() => import('./pages/ScanPage'));

function RouteFallback() {
  return (
    <div className="min-h-screen bg-observatory-bg text-observatory-text flex items-center justify-center px-6">
      <div className="card max-w-md w-full text-center">
        <div className="card-header">Loading page</div>
        <p className="text-sm text-observatory-text-muted">Preparing the Fingerprint² module…</p>
      </div>
    </div>
  );
}

function RouteShell({ children }: { children: ReactNode }) {
  return (
    <RouteErrorBoundary>
      <Suspense fallback={<RouteFallback />}>{children}</Suspense>
    </RouteErrorBoundary>
  );
}

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<RouteShell><LandingPage /></RouteShell>} />
      <Route element={<AppLayout />}>
        <Route path="/scan" element={<RouteShell><ScanPage /></RouteShell>} />
        <Route path="/report" element={<RouteShell><BiasReportPage /></RouteShell>} />
        <Route path="/lab" element={<RouteShell><LabPage /></RouteShell>} />
        <Route path="/airport" element={<RouteShell><AirportPage /></RouteShell>} />
        <Route path="/eid" element={<RouteShell><EIDPage /></RouteShell>} />
        <Route path="/eu-ai-act" element={<RouteShell><EUAIActPage /></RouteShell>} />
        <Route path="/mitigation" element={<RouteShell><MitigationPage /></RouteShell>} />
      </Route>
    </Routes>
  );
}
