import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import "@testing-library/jest-dom";
import CareWatchDashboard from "@/components/CareWatchDashboard";

jest.mock("next/navigation", () => ({ useRouter: () => ({}) }));

beforeEach(() => {
  global.fetch = jest.fn(() =>
    Promise.resolve({ json: () => Promise.resolve({}) })
  );
});
afterEach(() => jest.clearAllMocks());

test("renders CAREWATCH branding on load", () => {
  render(<CareWatchDashboard />);
  expect(screen.getByText("CARE")).toBeInTheDocument();
  expect(screen.getByText("WATCH")).toBeInTheDocument();
});

test("renders in normal mode by default", () => {
  render(<CareWatchDashboard />);
  expect(screen.getByText("NORMAL DAY")).toBeInTheDocument();
});

test("crisis mode shows alert banner", async () => {
  const user = userEvent.setup();
  render(<CareWatchDashboard />);
  await user.click(screen.getByText("CRISIS MODE"));
  await waitFor(() =>
    expect(screen.getByText(/CRITICAL ALERT/i)).toBeInTheDocument()
  );
});

test("acknowledging crisis banner hides it", async () => {
  const user = userEvent.setup();
  render(<CareWatchDashboard />);
  await user.click(screen.getByText("CRISIS MODE"));
  await waitFor(() => screen.getByText(/CRITICAL ALERT/i));
  await user.click(screen.getByText("ACKNOWLEDGE"));
  await waitFor(() =>
    expect(screen.queryByText(/CRITICAL ALERT/i)).not.toBeInTheDocument()
  );
});

test("fallback to demo data when fetch returns empty", async () => {
  render(<CareWatchDashboard />);
  // Demo data has eating as current activity in NORMAL mode
  await waitFor(() =>
    expect(screen.getByText("EATING")).toBeInTheDocument()
  );
});
