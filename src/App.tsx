import { Suspense, lazy, type ReactNode } from 'react';
import { Routes, Route, useLocation } from 'react-router-dom';
import AppLayout from './components/AppLayout';
import RouteErrorBoundary from './components/RouteErrorBoundary';
import LandingPage from './pages/LandingPage';
import BiasReportPage from './pages/BiasReportPage';
import LabPage from './pages/LabPage';
import EUAIActPage from './pages/EUAIActPage';
import MitigationPage from './pages/MitigationPage';

const AirportPage = lazy(() => import('./pages/AirportPage'));
const ScanPage = lazy(() => import('./pages/ScanPage'));

function RouteFallback() {
  return (
    <div className="flex min-h-[40vh] items-center justify-center px-6 text-center text-sm text-observatory-text-muted">
      Loading module…
    </div>
  );
}

function RouteShell({ children }: { children: ReactNode }) {
  const location = useLocation();
  return (
    <RouteErrorBoundary resetKey={location.pathname}>
      <Suspense fallback={<RouteFallback />}>
        {children}
      </Suspense>
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
        
        <Route path="/eu-ai-act" element={<RouteShell><EUAIActPage /></RouteShell>} />
        <Route path="/mitigation" element={<RouteShell><MitigationPage /></RouteShell>} />
      </Route>
    </Routes>
  );
}
