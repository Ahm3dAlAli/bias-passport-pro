// Direct edge function caller that doesn't depend on the Supabase client initializing
const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL || "https://iwgtanivfzhosqwxrujf.supabase.co";
const SUPABASE_KEY = import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY || "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Iml3Z3Rhbml2Znpob3Nxd3hydWpmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzQ4ODcwOTcsImV4cCI6MjA5MDQ2MzA5N30.2BUpEqzPqaIK_hciN1u7NUWpWEHvDlYtS1UcAEBf_rg";

export async function invokeBiasScan(body: { image: string; model?: string }) {
  const url = `${SUPABASE_URL}/functions/v1/bias-scan`;
  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${SUPABASE_KEY}`,
      "apikey": SUPABASE_KEY,
    },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Edge function error: ${res.status}`);
  }

  return res.json();
}
