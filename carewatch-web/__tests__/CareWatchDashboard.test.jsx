import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import "@testing-library/jest-dom";
import CareWatchDashboard from "@/components/CareWatchDashboard";

jest.mock("next/navigation", () => ({ useRouter: () => ({}) }));

beforeEach(() => {
  global.fetch = jest.fn(() =>
    Promise.resolve({ json: () => Promise.resolve({}) })
  );
  // Reset consent before each test so the modal appears fresh
  localStorage.clear();
});
afterEach(() => jest.clearAllMocks());

// ── Consent modal tests ──────────────────────────────────────────────────────

test("renders consent modal before dashboard on first load", () => {
  render(<CareWatchDashboard />);
  expect(screen.getByText("DATA COLLECTION CONSENT")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /I AGREE/i })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /I DO NOT AGREE/i })).toBeInTheDocument();
  // Dashboard-specific controls should NOT be visible before consent
  expect(screen.queryByText(/EVENT FEED/i)).not.toBeInTheDocument();
});

test("dashboard renders after user clicks I Agree", async () => {
  const user = userEvent.setup();
  render(<CareWatchDashboard />);
  await user.click(screen.getByRole("button", { name: /I AGREE/i }));
  await waitFor(() =>
    expect(screen.getByText("CARE")).toBeInTheDocument()
  );
  expect(screen.getByText("MEDS")).toBeInTheDocument();
});

test("consent is remembered in localStorage", async () => {
  const user = userEvent.setup();
  render(<CareWatchDashboard />);
  await user.click(screen.getByRole("button", { name: /I AGREE/i }));
  expect(localStorage.getItem("carewatch_consent")).toBe("granted");
});

test("decline locks access until user accepts", async () => {
  const user = userEvent.setup();
  render(<CareWatchDashboard />);
  await user.click(screen.getByRole("button", { name: /I DO NOT AGREE/i }));
  expect(localStorage.getItem("carewatch_consent")).toBe("declined");
  expect(screen.getByText(/ACCESS DISABLED/i)).toBeInTheDocument();
  expect(screen.queryByText(/EVENT FEED/i)).not.toBeInTheDocument();

  await user.click(screen.getByRole("button", { name: /ACCEPT CONSENT/i }));
  await waitFor(() => expect(screen.getByText("CARE")).toBeInTheDocument());
});

test("skips consent modal if already consented (localStorage set)", () => {
  localStorage.setItem("carewatch_consent", "granted");
  render(<CareWatchDashboard />);
  expect(screen.getByText("CARE")).toBeInTheDocument();
  expect(screen.queryByText("DATA COLLECTION CONSENT")).not.toBeInTheDocument();
});

// ── Existing dashboard tests (require consent first) ─────────────────────────

// Helper: render the dashboard with consent pre-set
function renderWithConsent() {
  localStorage.setItem("carewatch_consent", "granted");
  return render(<CareWatchDashboard />);
}

test("renders CARE MEDS branding on load", () => {
  renderWithConsent();
  expect(screen.getByText("CARE")).toBeInTheDocument();
  expect(screen.getByText("MEDS")).toBeInTheDocument();
});

test("renders event feed after consent", () => {
  renderWithConsent();
  expect(screen.getByText(/EVENT FEED/i)).toBeInTheDocument();
});

test("fallback to demo data when API fails", async () => {
  global.fetch = jest.fn(() => Promise.reject(new Error("network down")));
  renderWithConsent();
  await waitFor(() =>
    expect(screen.getByText(/No live data — API unreachable/i)).toBeInTheDocument()
  );
  expect(screen.getByText("DEMO")).toBeInTheDocument();
});
