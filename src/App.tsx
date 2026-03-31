import { type ReactNode } from 'react'; // v2
import { Routes, Route, useLocation } from 'react-router-dom';
import AppLayout from './components/AppLayout';
import RouteErrorBoundary from './components/RouteErrorBoundary';
import LandingPage from './pages/LandingPage';
import BiasReportPage from './pages/BiasReportPage';
import LabPage from './pages/LabPage';
import AirportPage from './pages/AirportPage';
import EUAIActPage from './pages/EUAIActPage';
import MitigationPage from './pages/MitigationPage';
import ScanPage from './pages/ScanPage';

function RouteShell({ children }: { children: ReactNode }) {
  const location = useLocation();
  return (
    <RouteErrorBoundary resetKey={location.pathname}>
      {children}
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
