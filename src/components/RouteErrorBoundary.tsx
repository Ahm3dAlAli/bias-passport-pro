import { AlertTriangle, RefreshCw } from 'lucide-react';
import { Component, type ErrorInfo, type ReactNode } from 'react';

interface RouteErrorBoundaryProps {
  children: ReactNode;
}

interface RouteErrorBoundaryState {
  hasError: boolean;
}

export default class RouteErrorBoundary extends Component<RouteErrorBoundaryProps, RouteErrorBoundaryState> {
  state: RouteErrorBoundaryState = { hasError: false };

  static getDerivedStateFromError(): RouteErrorBoundaryState {
    return { hasError: true };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Route render failed:', error, errorInfo);
  }

  handleReload = () => {
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-observatory-bg text-observatory-text flex items-center justify-center px-6">
          <div className="card max-w-md w-full text-center">
            <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-2xl bg-observatory-danger/10">
              <AlertTriangle className="h-6 w-6 text-observatory-danger" />
            </div>
            <h1 className="text-xl font-semibold text-observatory-text">This page failed to load</h1>
            <p className="mt-2 text-sm text-observatory-text-muted">
              The rest of the app is protected now, so one broken page will not blank the entire website.
            </p>
            <button
              type="button"
              onClick={this.handleReload}
              className="mt-5 inline-flex items-center gap-2 rounded-xl bg-observatory-accent px-4 py-2.5 text-sm font-semibold text-observatory-bg transition-all hover:bg-observatory-accent-glow"
            >
              <RefreshCw className="h-4 w-4" />
              Reload page
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}