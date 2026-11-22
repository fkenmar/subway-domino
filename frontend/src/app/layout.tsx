import type { Metadata, Viewport } from "next";
import type { ReactNode } from "react";
import "./globals.css";

const description =
  "Break the real New York City subway, then hand control to a reinforcement-learning dispatcher and watch it contain the meltdown — live, on MTA open data.";

export const metadata: Metadata = {
  title: "Subway Domino — Human vs. AI NYC subway control",
  description,
  applicationName: "Subway Domino",
  openGraph: {
    title: "Subway Domino",
    description,
    url: "https://kenmar.dev/subway",
    siteName: "kenmar.dev",
    type: "website",
  },
  robots: { index: true, follow: true },
};

export const viewport: Viewport = {
  themeColor: "#070b11",
  colorScheme: "dark",
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
};

export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
