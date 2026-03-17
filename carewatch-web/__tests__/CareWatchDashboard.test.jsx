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

test("renders CARE MEDS branding on load", () => {
  render(<CareWatchDashboard />);
  expect(screen.getByText("CARE")).toBeInTheDocument();
  expect(screen.getByText("MEDS")).toBeInTheDocument();
});

test("renders in normal mode by default", () => {
  render(<CareWatchDashboard />);
  expect(screen.getByText("NORMAL DAY")).toBeInTheDocument();
});

test("crisis mode shows alert banner", async () => {
  const user = userEvent.setup();
  render(<CareWatchDashboard />);
  await user.click(screen.getByText("MISSED DOSE"));
  await waitFor(() =>
    expect(screen.getByText(/CRITICAL ALERT/i)).toBeInTheDocument()
  );
});

test("acknowledging crisis banner hides it", async () => {
  const user = userEvent.setup();
  render(<CareWatchDashboard />);
  await user.click(screen.getByText("MISSED DOSE"));
  await waitFor(() => screen.getByText(/CRITICAL ALERT/i));
  await user.click(screen.getByText("ACKNOWLEDGE"));
  await waitFor(() =>
    expect(screen.queryByText(/CRITICAL ALERT/i)).not.toBeInTheDocument()
  );
});

test("fallback to demo data when API fails", async () => {
  global.fetch = jest.fn(() => Promise.reject(new Error("network down")));
  render(<CareWatchDashboard />);
  await waitFor(() =>
    expect(screen.getByText(/No live data — API unreachable/i)).toBeInTheDocument()
  );
  expect(screen.getByText("DEMO")).toBeInTheDocument();
});
