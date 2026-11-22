import type { Metadata, Viewport } from "next";
import type { ReactNode } from "react";
import { DotGothic16 } from "next/font/google";
import "./globals.css";

/**
 * Dot-matrix LED face (OFL) for the countdown-board digits only. DotGothic16 is a dense pixel
 * matrix where every glyph is unambiguous ("." is a real full stop, "0" is not a diamond, "±"
 * and "−" are in the font), unlike the sparser Doto used first.
 */
const led = DotGothic16({ subsets: ["latin"], weight: "400", variable: "--font-led", display: "swap" });

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
  themeColor: "#000000",
  colorScheme: "dark",
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
};

export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="en" className={led.variable}>
      <body>{children}</body>
    </html>
  );
}
