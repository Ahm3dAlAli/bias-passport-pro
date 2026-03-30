import { Routes, Route } from 'react-router-dom';
import AppLayout from './components/AppLayout';
import LandingPage from './pages/LandingPage';
import BiasReportPage from './pages/BiasReportPage';
import LabPage from './pages/LabPage';
import AirportPage from './pages/AirportPage';
import EIDPage from './pages/EIDPage';
import BankingPage from './pages/BankingPage';
import EUAIActPage from './pages/EUAIActPage';
import MitigationPage from './pages/MitigationPage';

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route element={<AppLayout />}>
        <Route path="/report" element={<BiasReportPage />} />
        <Route path="/lab" element={<LabPage />} />
        <Route path="/airport" element={<AirportPage />} />
        <Route path="/eid" element={<EIDPage />} />
        <Route path="/banking" element={<BankingPage />} />
        <Route path="/eu-ai-act" element={<EUAIActPage />} />
        <Route path="/mitigation" element={<MitigationPage />} />
      </Route>
    </Routes>
  );
}
